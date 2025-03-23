import gc
import os
import torch as t
import numpy as np
import logging
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import einops
import transformer_lens
import sae_lens
import time
from huggingface_hub import list_repo_files

# ----------------------- WANDA Pruning Function -----------------------
@t.no_grad()
def wanda(W, X_norm, sparse_ratio):
    W_metric = W.abs() * X_norm
    _, sorted_idx = W_metric.sort(dim=1)
    pruned_idx = sorted_idx[:, :int(W.shape[1] * sparse_ratio)]
    W_clone = W.detach().clone()
    W_clone.scatter_(dim=1, index=pruned_idx, src=t.zeros_like(pruned_idx, dtype=W.dtype))
    del W_metric, sorted_idx, pruned_idx
    return W_clone


# ----------------------- Update SAE Weights -----------------------
def update_sae_weights_with_wanda(sae, original_W_enc, original_W_dec, X_enc, X_dec, sparse_ratio):
    W_enc = wanda(original_W_enc, X_enc, sparse_ratio)
    sae.W_enc = t.nn.Parameter(W_enc)
    del W_enc

    W_dec = wanda(original_W_dec, X_dec, sparse_ratio)
    sae.W_dec = t.nn.Parameter(W_dec)
    del W_dec
    
    return sae

# ----------------------- Compute Reconstruction Loss -----------------------
@t.no_grad()
def compute_reconstruction_loss(model, sae, token_batches):    
    hook_name = sae.cfg.hook_name

    def reconstr_hook(activation, hook, sae_out):
        return sae_out

    total_loss = 0.0
    count = 0

    for batch in tqdm(token_batches, desc="Computing loss"):
        logits, cache = model.run_with_cache(batch, prepend_bos=True)
        del logits
        feature_acts = sae.encode(cache[hook_name])
        del cache
        
        sae_out = sae.decode(feature_acts)
        del feature_acts
        
        loss = model.run_with_hooks(
            batch,
            fwd_hooks=[(hook_name, partial(reconstr_hook, sae_out=sae_out))],
            return_type="loss"
        ).item()
        total_loss += loss
        count += 1

        del sae_out
        t.cuda.empty_cache()

    return round(total_loss / count, 3) if count > 0 else 0.0


# ----------------------- Norm Cache Function -----------------------
@t.no_grad()
def compute_norms(model, sae, loader, kind="enc"):
    hook_name = sae.cfg.hook_name
    norms = []

    for batch in tqdm(loader, desc="X norms"):
        _, cache = model.run_with_cache_with_saes(batch, saes=[sae])

        key = f"{hook_name}.hook_sae_acts_pre" if kind == "enc" else f"{hook_name}.hook_sae_recons"
        X = einops.rearrange(cache[key], "b h d -> (b h) d")
        norms.append(X.norm(p=2, dim=0))

        del cache
        t.cuda.empty_cache()
    return t.stack(norms).mean(dim=0)


# ----------------------- Collate Function -----------------------
def collate_fn(batch):
    batch = [t.squeeze(tokens, dim=0) for tokens in batch]
    return pad_sequence(batch, batch_first=True, padding_value=0)

class DataClass(t.utils.data.Dataset):
    def __init__(self, dataset, model, max_length=1024):
        self.dataset = dataset
        self.model = model
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx.item()]['text']
        tokens = self.model.to_tokens(text)
        tokens = tokens[:, :self.max_length]
        return tokens

# ----------------------- Main -----------------------
if __name__ == "__main__":
    # Parse Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="openwebtext")
    parser.add_argument("--device", type=int, default="0")
    args = parser.parse_args()

    # Configs
    dataset_name, device = args.dataset, f'cuda:{args.device}'
    # dataset_name, device1, device2 = args.dataset, f'cuda:{args.device}', 'cpu'
    model_name = 'gemma-2-2b'
    # hook_ids = ["hook_resid_post", "hook_mlp_out", "attn.hook_z"] 
    hook_ids = ["attn.hook_z"]

    release = {
        "hook_resid_post":"gemma-scope-2b-pt-res-canonical", 
        "hook_mlp_out": "gemma-scope-2b-pt-mlp-canonical", 
        "attn.hook_z": "gemma-scope-2b-pt-att-canonical"
    }

    files_in_the_repo = list_repo_files("suchitg/sae-compression-gemma-2-2b-pruned-sae")

    for hook_id in hook_ids:
        log_dir = f'logs/{hook_id}'
        os.makedirs(log_dir, exist_ok=True)

        # Logging
        logging.basicConfig(
            filename=os.path.join(log_dir, f'{dataset_name}_log.txt'),
            level=logging.INFO,
            format='%(message)s',
        )
        logging.info(f"Using device: {device}")


        # Load model
        t.set_grad_enabled(False)
        model: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained(model_name, device=device)
        model.load_state_dict(t.load(f'/home/gupte.31/COLM/sae-compression/gemma2b/pruned/gemma-2-2b_wanda.pth'))
        model.to(device)

        logging.info("Model loaded")
        # Load dataset
        raw_dataset = transformer_lens.utils.get_dataset(dataset_name)
        data = DataClass(raw_dataset, model)
        del raw_dataset

        # Split dataset
        total_samples = 128 + 32
        indices = t.randperm(len(data))[:total_samples]
        subset = Subset(data, indices)
        train_set, val_set = random_split(subset, [128, 32])
        del subset
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
        del data, train_set, val_set


        # Layer-wise pruning
        for layer in tqdm(range(model.cfg.n_layers), desc="Layer-wise pruning"):
            pretrained_release = release[hook_id]
            pretrained_sae_id = f"layer_{layer}/width_16k/canonical"
            logging.info(f'Loading pretrained SAE: {pretrained_release}/{pretrained_sae_id}')

            sae = sae_lens.SAE.from_pretrained(pretrained_release, pretrained_sae_id, device=device)[0]

            # val_loss = compute_reconstruction_loss(model, sae, val_loader)
            # logging.info(f'Before pruning layer {layer} | Val loss: {val_loss} |')

            # Save original weights
            original_W_enc = sae.W_enc.detach().clone()
            original_W_dec = sae.W_dec.detach().clone()

            # Cache norms once
            X_enc = compute_norms(model, sae, train_loader, kind="enc")
            X_dec = compute_norms(model, sae, train_loader, kind="dec")
            sparse_ratios = np.array([0.75, 0.50, 0.25])
            # sparse_ratios = np.array([0.5]) # For testing - comment out for full run

            for ratio in sparse_ratios:
                filename = f'{dataset_name}_{ratio}_blocks.{layer}.{hook_id}/cfg.json'
                if filename not in files_in_the_repo:
                    sae = update_sae_weights_with_wanda(sae, original_W_enc, original_W_dec, X_enc, X_dec, ratio)
                    # val_loss = compute_reconstruction_loss(model, sae, val_loader)
                    # logging.info(f"Layer {layer} | Sparse Ratio: {ratio} | Val Loss: {val_loss}")

                    # Save pruned SAE
                    save_path = f'/local/scratch/suchit/COLM/pruned_saes/{model_name}/wanda/{dataset_name}/{hook_id}_ratio={ratio}'
                    os.makedirs(save_path, exist_ok=True)
                    t.save(sae.state_dict(), f'{save_path}/blocks.{layer}.{hook_id}.pth')

                    sae_lens.upload_saes_to_huggingface({f"{dataset_name}_{ratio}_blocks.{layer}.{hook_id}":sae}, hf_repo_id="suchitg/sae-compression-gemma-2-2b-pruned-sae")
                    # del val_loss

            time.sleep(2)
            print("Resume after 30 seconds")
            del X_enc, X_dec, sae

            gc.collect()
            t.cuda.empty_cache()

    
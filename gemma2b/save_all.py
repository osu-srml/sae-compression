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

# ----------------------- WANDA Pruning Function -----------------------
def wanda(W, X_norm, sparse_ratio):
    with t.no_grad():
        W_metric = W.abs() * X_norm
        _, sorted_idx = W_metric.sort(dim=1)
        pruned_idx = sorted_idx[:, :int(W.shape[1] * sparse_ratio)]
        W_clone = W.detach().clone()
        W_clone.scatter_(dim=1, index=pruned_idx, src=t.zeros_like(pruned_idx, dtype=W.dtype))
    return W_clone

# ----------------------- Update SAE Weights -----------------------
def prune_sae_batch(model, sae, token_batches, sparse_ratio):
    """Prune SAE using batched token inputs with given sparsity ratio."""
    with t.no_grad():
        for batch in tqdm(token_batches, desc=f"Pruning SAE (sparsity {sparse_ratio})"):
            _, cache = model.run_with_cache_with_saes(batch, saes=[sae])
            X_enc = einops.rearrange(cache[f"{sae.cfg.hook_name}.hook_sae_acts_pre"], "b h d -> (b h) d").norm(p=2, dim=0)
            X_dec = einops.rearrange(cache[f"{sae.cfg.hook_name}.hook_sae_recons"], "b h d -> (b h) d").norm(p=2, dim=0)

            for name, param in sae.named_parameters():
                if 'W' in name:
                    if 'enc' in name:
                        sae.W_enc = t.nn.Parameter(wanda(param, X_enc, sparse_ratio))
                    else:
                        sae.W_dec = t.nn.Parameter(wanda(param, X_dec, sparse_ratio))
            del cache, X_enc, X_dec, _
            gc.collect()
            t.cuda.empty_cache()
    return sae

# ----------------------- Collate Function -----------------------
def collate_fn(batch):
    batch = [t.squeeze(tokens, dim=0) for tokens in batch]
    return pad_sequence(batch, batch_first=True, padding_value=0)

# ----------------------- Dataset Wrapper -----------------------
class DataClass(t.utils.data.Dataset):
    def __init__(self, dataset, model, max_length=1024):
        self.dataset = dataset
        self.model = model
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx.item()]['text']
        tokens = self.model.to_tokens(text)[:self.max_length]
        return tokens

# ----------------------- Main -----------------------
if __name__ == "__main__":
    # Parse Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="openwebtext")
    parser.add_argument("--device", type=int, default="5")
    args = parser.parse_args()

    # Configs
    dataset_name, device_id = args.dataset, f'cuda:{args.device}'
    model_name = 'gemma-2-2b'
    hook_ids = ["hook_resid_post", "hook_mlp_out", "hook_attn_out"] 
    release = {
        "hook_resid_post": "gemma-scope-2b-pt-res-canonical", 
        "hook_mlp_out": "gemma-scope-2b-pt-mlp-canonical", 
        "attn.hook_z": "gemma-scope-2b-pt-att-canonical"
    }

    # Set device and load model once
    device = t.device(device_id)
    t.set_grad_enabled(False)
    model: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained(model_name, device=device)
    model.load_state_dict(t.load(f'/home/gupte.31/COLM/sae-compression/gemma2b/pruned/gemma-2-2b_wanda.pth'))

    # Load dataset once
    raw_dataset = transformer_lens.utils.get_dataset(dataset_name)
    data = DataClass(raw_dataset, model, model.cfg.n_ctx)
    del raw_dataset

    # Split dataset
    total_samples = 128 + 2
    indices = t.randperm(len(data))[:total_samples]
    subset = Subset(data, indices)
    train_set, val_set = random_split(subset, [128, 2])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Process each hook_id using the same model, offloading when not in use
    for hook_id in hook_ids:
        print(f"Processing hook_id: {hook_id}")
        for layer in tqdm(range(model.cfg.n_layers), desc=f"Layer-wise pruning for {hook_id}"):
            pretrained_release = release.get(hook_id, None)
            if pretrained_release is None:
                print(f"No pretrained release found for hook_id: {hook_id}")
                continue
            pretrained_sae_id = f"layer_{layer}/width_16k/canonical"

            # Using a single sparse ratio for testing (modify as needed)
            sparse_ratios = np.array([0.5])  # For testing - comment out for full run

            for ratio in sparse_ratios:
                # Load SAE and perform pruning
                sae = sae_lens.SAE.from_pretrained(pretrained_release, pretrained_sae_id, device=str(device))[0]
                sae.eval()
                sae = prune_sae_batch(model, sae, train_loader, sparse_ratio=ratio)

                # Offload SAE to CPU to free GPU memory before saving
                sae = sae.to('cpu')
                save_path = f'/local/scratch/suchit/COLM/pruned_saes/{model_name}/wanda/{dataset_name}/{hook_id}_ratio={ratio}'
                os.makedirs(save_path, exist_ok=True)
                t.save(sae.state_dict(), f'{save_path}/blocks.{layer}.{hook_id}.pth')

                del sae
                gc.collect()
                t.cuda.empty_cache()

        # Offload model to CPU after finishing this hook_id to free up GPU memory
        model = model.to('cpu')
        gc.collect()
        t.cuda.empty_cache()
        # If there are more hook_ids, move model back to GPU for the next round
        if hook_id != hook_ids[-1]:
            model = model.to(device)

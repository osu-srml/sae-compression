import gc
import math
import os
import sys
from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np
import torch as t
from datasets import load_dataset
import transformer_lens
import sae_lens

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import einops
import logging
import argparse
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset


# Wanda Pruning Function
def wanda(W, X_norm, sparse_ratio):
    with t.no_grad():
        W_metric = W.abs() * X_norm
        _, sorted_idx = W_metric.sort(dim=1)
        pruned_idx = sorted_idx[:, :int(W.shape[1] * sparse_ratio)]
        
        W_clone = W.detach().clone()
        W_clone.scatter_(dim=1, index=pruned_idx, src=t.zeros_like(pruned_idx, dtype=W.dtype))

    return W_clone

# Prune SAE Function
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
            del cache
            t.cuda.empty_cache()
    return sae

# Compute Reconstruction Loss 
def compute_reconstruction_loss(model, sae, token_batches):
    hook_name = sae.cfg.hook_name
    def reconstr_hook(activation, hook, sae_out):
        return sae_out
    
    total_loss = 0.0
    count = 0

    with t.no_grad():
        for batch in tqdm(token_batches, desc=f"Computing loss"):
            _, cache = model.run_with_cache(batch, prepend_bos=True)

            feature_acts = sae.encode(cache[hook_name])
            sae_out = sae.decode(feature_acts)

            total_loss += model.run_with_hooks(batch, fwd_hooks=[(hook_name, partial(reconstr_hook, sae_out=sae_out))], return_type="loss").item()
            count += 1
    
            del cache
            t.cuda.empty_cache()

    avg_loss = total_loss / count if count > 0 else 0.0
    return round(avg_loss, 4)



if __name__ == "__main__":
    # Set Seed
    t.manual_seed(0)

    # Parse Arguments
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="openwebtext")
    args.add_argument("--prune", type=str, default="wanda")
    args.add_argument("--device", type=str, default="1")

    args = args.parse_args()

    dataset_name, pruning_method, device = args.dataset, args.prune, f'cuda:{args.device}'

    # Set up logging
    logging.basicConfig(
        filename=f'logs/{dataset_name}_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    # Set Device
    device = t.device(device)
    t.set_grad_enabled(False)

    # print("Using device:", device)
    logging.info(f"Using device: {device}")
    gpt2_pruned: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained("gpt2-small", device=device)
    gpt2_pruned.load_state_dict(t.load(f'pruned/pruned_gpt2_{pruning_method}.pth'))
    logging.info(f"Loaded wanda pruned gpt-2 small")

    def collate_fn(batch):
        """Pads token sequences in a batch to the maximum length in the batch."""
        batch = [t.squeeze(tokens, dim=0) for tokens in batch]  # Remove extra dim if shape is [1, N]
        batch_padded = pad_sequence(batch, batch_first=True, padding_value=0)  # Pad shorter sequences
        return batch_padded
    
    # Load Dataset
    dataset = transformer_lens.utils.get_dataset(dataset_name)

    # Define Dataset Class
    class DataClass(t.utils.data.Dataset):
        def __init__(self, dataset, max_length=128):
            self.dataset = dataset
            self.max_length = max_length

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            # print(idx.item(), type(idx.item()))
            # exit()
            text = self.dataset[idx.item()]['text']
            tokens = gpt2_pruned.to_tokens(text)[:self.max_length]
            return tokens
        
    # Initialize Dataset
    data = DataClass(dataset)
    logging.info(f"Loaded calibration dataset: {dataset_name}")

    # # Split into Train (80%), Val (10%), Test (10%)
    # train_size = 128 # Calibration dataset size as specified in wanda paper
    # val_size = int(0.5 * len(data))
    # test_size = len(data) - train_size - val_size
    # train_set, val_set, test_set = random_split(data, [train_size, val_size, test_size])

    # # Create DataLoaders
    # batch_size = 8
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_set, batch_size=16, shuffle=False, collate_fn=collate_fn)
    # test_loader = DataLoader(test_set, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Define dataset sizes
    train_size = 128 # Calibration dataset size as specified in wanda paper
    val_size = 1024
    test_size = 128
    total_needed = train_size + val_size + test_size

    # Randomly select 192 samples from the dataset
    subset_indices = t.randperm(len(data))[:total_needed]  # Randomly permute indices and take 192
    subset_data = Subset(data, subset_indices)  # Create a subset

    # Split into Train (128), Val (32), Test (32)
    train_set, val_set, test_set = random_split(subset_data, [train_size, val_size, test_size])

    # Create DataLoaders
    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    results = list()

    # Prune GPT-2 SAE Layer-wise
    for layer in tqdm(range(gpt2_pruned.cfg.n_layers)):
        hf_repo_id = 'suchitg/sae_test'
        hf_sae_id = f"blocks.{layer}.attn.hook_z-attn-sae-v1"
        hf_sae = sae_lens.SAE.from_pretrained(hf_repo_id, hf_sae_id, device=str(device))[0]
        hf_val_loss = compute_reconstruction_loss(gpt2_pruned, hf_sae, val_loader)
        hf_test_loss = compute_reconstruction_loss(gpt2_pruned, hf_sae, test_loader)

        logging.info(f'HuggingFace SAE trained on the pruned model, Validation loss: {hf_val_loss}, and Test loss: {hf_test_loss}')
        results.append(["SAE trained on pruned gpt2-small ", layer, hf_val_loss, hf_test_loss, 0.0])
        del hf_sae, hf_val_loss, hf_test_loss

        pretrained_release = "gpt2-small-hook-z-kk"
        pretrained_sae_id = f"blocks.{layer}.hook_z"
        pretrained_sae = sae_lens.SAE.from_pretrained(pretrained_release, pretrained_sae_id, device=str(device))[0]
        pretrained_val_loss = compute_reconstruction_loss(gpt2_pruned, pretrained_sae, val_loader)
        pretrained_test_loss = compute_reconstruction_loss(gpt2_pruned, pretrained_sae, test_loader)


        logging.info(f'Before pruning pretrained SAE, Validation loss: {pretrained_val_loss}, and Test loss: {pretrained_test_loss}')
        results.append(["Pretrained SAE", layer, pretrained_val_loss, pretrained_test_loss, 0.0])
        
        logging.info(f'Pruning for layer: {layer}')
        logging.info(f'Pretrained SAE location: {pretrained_release}/{pretrained_sae_id}')
        logging.info(f'HuggingFace SAE location: {hf_repo_id}/{hf_sae_id}')

        # Hyperparameter Tuning: Finding Best sparse_ratio
        best_ratio = None
        best_val_loss = float("inf")
        sparse_ratios = np.round(np.arange(0.99, 0.24, -0.02), 2)

        for sparse_ratio in sparse_ratios:
            pruned_sae = prune_sae_batch(gpt2_pruned, pretrained_sae, train_loader, sparse_ratio=sparse_ratio)
            val_loss = compute_reconstruction_loss(gpt2_pruned, pruned_sae, val_loader)

            logging.info(f"After pruning layer {layer} of pretrained SAE with sparse ratio: {sparse_ratio}, Validation Loss: {val_loss}")
            results.append(["Pruned SAE", layer, val_loss, None, sparse_ratio])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ratio = sparse_ratio

            pretrained_sae = sae_lens.SAE.from_pretrained(pretrained_release, pretrained_sae_id, device=str(device))[0]
            logging.info('Resetting pretrained SAE')
            del pruned_sae
            t.cuda.empty_cache()
        
        logging.info(f"For layer {layer}, the best Sparse Ratio: {best_ratio} with validation loss: {best_val_loss}")

        best_pruned_sae = prune_sae_batch(gpt2_pruned, pretrained_sae, train_loader, sparse_ratio=best_ratio)
        best_test_loss = compute_reconstruction_loss(gpt2_pruned, best_pruned_sae, test_loader)

        logging.info(f'After pruning pretrained SAE, Validation loss: {best_val_loss}, and Test loss: {best_test_loss}')
        results.append(["Best sparse ratio pruned SAE", layer, best_val_loss, best_test_loss, best_ratio])

        # Save Pruned SAE
        save_path = f'/local/scratch/suchit/COLM/{pruning_method}/{dataset_name}'
        os.makedirs(save_path, exist_ok=True)
        t.save(best_pruned_sae.state_dict(), f'{save_path}/pruned_attn_sae_layer={layer}_BSR={best_ratio}_val=1024.pth')


        del pretrained_sae, best_pruned_sae
        gc.collect()
        t.cuda.empty_cache()
        
    pd.DataFrame(results, columns=["Model", "Layer", "Validation Loss", "Test Loss", "Sparse Ratio"]).to_csv(f'logs/results_{dataset_name}.csv', index=False)
    




            
            

            































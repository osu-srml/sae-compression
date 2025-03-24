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
    args.add_argument("--device", type=str, default="4")

    args = args.parse_args()
    datasets = ["openwebtext", "wiki", "c4", "pile"]
    for dataset_name in datasets:
        pruning_method, device = args.prune, f'cuda:{args.device}'

        # Set Device
        device = t.device(device)
        t.set_grad_enabled(False)

        print("Using device:", device)
        gpt2_pruned: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained("gpt2-small", device=device)
        gpt2_pruned.load_state_dict(t.load(f'pruned/pruned_gpt2_{pruning_method}.pth'))
        
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

        # Define dataset sizes
        train_size = 128 # Calibration dataset size as specified in wanda paper
        val_size = 1024
        test_size = 128
        total_needed = train_size + val_size + test_size

        # Randomly select samples from the dataset
        subset_indices = t.randperm(len(data))[:total_needed]  # Randomly permute indices and take 192
        subset_data = Subset(data, subset_indices)  # Create a subset
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

            results.append(["SAE trained on pruned gpt2-small", "MATS", "standard", "30K", layer, hf_val_loss, hf_test_loss])
            del hf_sae, hf_val_loss, hf_test_loss
            gc.collect()
            t.cuda.empty_cache()

            hf_sae_id = f"blocks.{layer}.attn.hook_z-attn-sae-v-my_cfg"
            hf_sae = sae_lens.SAE.from_pretrained(hf_repo_id, hf_sae_id, device=str(device))[0]
            hf_val_loss = compute_reconstruction_loss(gpt2_pruned, hf_sae, val_loader)
            hf_test_loss = compute_reconstruction_loss(gpt2_pruned, hf_sae, test_loader)

            results.append(["SAE trained on pruned gpt2-small", "Custom", "standard", "30K", layer, hf_val_loss, hf_test_loss])
            del hf_sae, hf_val_loss, hf_test_loss
            gc.collect()
            t.cuda.empty_cache()


            hf_sae_id = f"blocks.{layer}.attn.hook_z-attn-sae-v-my_cfg_epochs=50K"
            hf_sae = sae_lens.SAE.from_pretrained(hf_repo_id, hf_sae_id, device=str(device))[0]
            hf_val_loss = compute_reconstruction_loss(gpt2_pruned, hf_sae, val_loader)
            hf_test_loss = compute_reconstruction_loss(gpt2_pruned, hf_sae, test_loader)

            results.append(["SAE trained on pruned gpt2-small", "Custom", "standard", "50K", layer, hf_val_loss, hf_test_loss])
            del hf_sae, hf_val_loss, hf_test_loss
            gc.collect()
            t.cuda.empty_cache()

            if layer in [0, 3, 5]:
                hf_sae_id = f"blocks.{layer}.attn.hook_z-attn-sae-v-my_cfg_gated_arch"
                hf_sae = sae_lens.SAE.from_pretrained(hf_repo_id, hf_sae_id, device=str(device))[0]
                hf_val_loss = compute_reconstruction_loss(gpt2_pruned, hf_sae, val_loader)
                hf_test_loss = compute_reconstruction_loss(gpt2_pruned, hf_sae, test_loader)

                results.append(["SAE trained on pruned gpt2-small", "Custom", "gated", "30K", layer, hf_val_loss, hf_test_loss])
                del hf_sae, hf_val_loss, hf_test_loss

                gc.collect()
                t.cuda.empty_cache()
            
        pd.DataFrame(results, columns=["Model", "Config", "Architecture", "Epochs", "Layer", "Validation Loss", "Test Loss"]).to_csv(f'logs/losses_{dataset_name}.csv', index=False)
        




            
            

            































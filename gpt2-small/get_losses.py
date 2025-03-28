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
import re

def extract_layer_data(text, layer_number, hook_name):
    pattern = rf"Loading pretrained SAE:.*?blocks\.{layer_number}\.{hook_name}(.*?)(?=Loading pretrained SAE:|$)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        print(f"Layer {layer_number} not found.")
        return []

    layer_block = match.group(1)
    entries = re.findall(r"Layer \d+ \| Sparse Ratio: ([\d.]+) \| Val Loss: ([\d.]+)", layer_block)
    
    return [(float(sr), float(vl)) for sr, vl in entries]

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
    args.add_argument("--device", type=str, default="0")
    args.add_argument("--hookid", type=str, default="0")

    args = args.parse_args()

    dataset_name, device, hookid = args.dataset, f'cuda:{args.device}', args.hookid

    hook_files = {
        'attn': 'hook_z',
        'mlp': 'hook_mlp_out',
        'resid': 'hook_resid_post',
    }
    hook_repos = {
        'attn': 'gpt2-small-hook-z-kk',
        'mlp': 'gpt2-small-mlp-out-v5-32k',
        'resid': 'gpt2-small-resid-post-v5-32k',
    }

    # Set Device
    device = t.device(device)
    t.set_grad_enabled(False)

    print("Using device:", device)
    gpt2_pruned: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained("gpt2-small", device=device)
    gpt2_pruned.load_state_dict(t.load('/home/gupte.31/COLM/sae-compression/gpt2-small/pruned/gpt2-small_wanda.pth'))
    
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
    del data, subset_data, train_set, test_set

    # Create DataLoaders
    batch_size = 16
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    results = list()

    # Prune GPT-2 SAE Layer-wise
    for layer in tqdm(range(gpt2_pruned.cfg.n_layers)):     
        repo_id = hook_repos[hookid]
        sae_id = f"blocks.{layer}.{hook_files[hookid]}"
        sae = sae_lens.SAE.from_pretrained(repo_id, sae_id, device=str(device))[0]
        val_loss = compute_reconstruction_loss(gpt2_pruned, sae, val_loader)

        results.append(["Pretrained", layer, val_loss, 0.0])
        del sae, val_loss
        gc.collect()
        t.cuda.empty_cache()

        repo_id = "suchitg/sae-compression-gpt-2-small-trained-sae-openwebtext-wanda"
        if hookid == 'attn':
            sae_id = f"blocks.{layer}.attn.hook_z-attn-sae-v-my_cfg"
        else:
            sae_id = f"gpt2-small-blocks.{layer}.{hook_files[hookid]}-standard-mycfg"
        sae = sae_lens.SAE.from_pretrained(repo_id, sae_id, device=str(device))[0]
        val_loss = compute_reconstruction_loss(gpt2_pruned, sae, val_loader)

        results.append(["Trained", layer, val_loss, 0.0])
        del sae, val_loss
        gc.collect()
        t.cuda.empty_cache()

        with open(f"/home/gupte.31/COLM/sae-compression/gpt2-small/logs/{hook_files[hookid]}/openwebtext_log.txt", "r") as file:
            log_text = file.read()

        read_txt = extract_layer_data(log_text, layer, hook_files[hookid])
        for sr, vl in read_txt:
            print(f"Sparse Ratio: {sr}, Val Loss: {vl}")
            results.append(["Pruned", layer, vl, sr])

    print(f"Finished processing layer {layer}")
    pd.DataFrame(results, columns=["SAE Variant", "Layer", "Validation Loss", "Sparsity"]).to_csv(f'logs/{hook_files[hookid]}/losses_{dataset_name}.csv', index=False)
    




            
            

            































import math, einops, sae_lens, transformer_lens, torch
from tqdm import tqdm
import torch.nn as nn

# Import get_loaders function from data module within the same directory 
from data import get_loaders
from collections import defaultdict
import fnmatch
import numpy as np
import pandas as pd
import torch as t
from datasets import load_dataset
import transformer_lens
import sae_lens


def wanda(W, X_norm, sparse_ratio=0.5):
    W_metric = W.abs() * X_norm
    _, sorted_idx = W_metric.sort(dim=1)
    pruned_idx = sorted_idx[:, :int(W.shape[1] * sparse_ratio)]
    
    W_clone = W.detach().clone()    
    W_clone = W_clone.scatter(dim=1, index=pruned_idx, src=t.zeros_like(pruned_idx, dtype=W.dtype))
    
    return W_clone

def prune_wanda(model, tokens):
    wts_act = {
        'attn.W_Q': 'attn.hook_q',
        'attn.W_K': 'attn.hook_k',
        'attn.W_V': 'attn.hook_v',
        'attn.W_O': 'hook_attn_out',
        'mlp.W_in': 'mlp.hook_pre',
        'mlp.W_out': 'hook_mlp_out'
    }
    
    for layer in range(model.cfg.n_layers):
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        
        for wt, act in wts_act.items():
            W = model.get_parameter(f'blocks.{layer}.{wt}')
            X = cache[f'blocks.{layer}.{act}']

            if W.dim() == 3:
                if 'W_O' in wt:
                    X_norm = X.norm(p=2, dim=0)
                    W_new = W.clone()  # Avoid modifying in-place
                    for head in range(W.shape[0]):
                        W_new[head] = wanda(W_new[head], X_norm, sparse_ratio=0.2)
                 
                else:
                    W_new = W.clone()
                    for head in range(W.shape[0]):
                        X_norm = X[:, head, :].norm(p=2, dim=0)
                        W_new[head] = wanda(W_new[head], X_norm, sparse_ratio=0.2)
                 
            else:
                X_norm = X.norm(p=2, dim=0)
                W_new = wanda(W.clone(), X_norm, sparse_ratio=0.5)

            del X
            with t.no_grad():
                W.copy_(W_new)
        del logits, cache

    return model

class SparseGPT:
    def __init__(self, W):
        self.dev = W.device
        self.W = W
        self.rows, self.cols = W.shape
        self.H = torch.zeros((self.cols, self.cols), device=self.dev, dtype=torch.float32)
        self.n_samples = 0

    def add_batch(self, X): 
        tmp = X.shape[0]
        if self.n_samples > 0:
            self.H.mul_(self.n_samples / (self.n_samples + tmp)) 
        self.n_samples += tmp

        scale_factor = math.sqrt(2 / max(self.n_samples, 1))
        X = scale_factor * X.float()

        self.H.add_(X.T @ X) 

    
    def faster_prune(self, W, sparsity=0.2, blocksize=128, percdamp=0.01):
        W = W.float()

        H = self.H
        del self.H

        dead = torch.any(H == 0, dim=0)
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.cols, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        torch.cuda.empty_cache()
        mask = None

        for i1 in range(0, self.cols, blocksize):
            i2 = min(i1 + blocksize, self.cols)
            count = i2 - i1

            W1 = W[:, i1:i2]
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if mask is not None:
                mask1 = mask[:, i1:i2]
            else:  
                tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                mask1 = tmp <= thresh

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        self.W = W
        
    def free(self):
        self.H = None
        torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(model, tokens):
    print("Starting pruning ...")
    wts_act = {
    'attn.W_Q': 'attn.hook_q',
    'attn.W_K': 'attn.hook_k',
    'attn.W_V': 'attn.hook_v',
    'attn.W_O': 'hook_attn_out',
    'mlp.W_in': 'mlp.hook_pre',
    'mlp.W_out': 'hook_mlp_out'
    }

    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    for layer in range(model.cfg.n_layers):
        layer_cache = {k: v for k, v in cache.items() if f'blocks.{layer}.' in k}

        for wt, act in wts_act.items():
            W = model.get_parameter(f'blocks.{layer}.{wt}')
            X = layer_cache[f'blocks.{layer}.{act}']

            if W.dim() == 2:
                sparsegpt_object = SparseGPT(W)
                sparsegpt_object.add_batch(X)
                sparsegpt_object.faster_prune(W)
                W.copy_(sparsegpt_object.W)
                sparsegpt_object.free()

            else:
                if 'W_O' in wt:
                    for head in range(W.shape[0]):
                        sparsegpt_object = SparseGPT(W[head])
                        sparsegpt_object.add_batch(X)
                        sparsegpt_object.faster_prune(W[head])
                        W[head].copy_(sparsegpt_object.W)
                        sparsegpt_object.free()


                else:
                    for head in range(W.shape[0]):
                        sparsegpt_object = SparseGPT(W[head])
                        sparsegpt_object.add_batch(X[:, head, :])
                        sparsegpt_object.faster_prune(W[head])
                        W[head].copy_(sparsegpt_object.W)
                        sparsegpt_object.free()

            del sparsegpt_object
            torch.cuda.empty_cache()

        del layer_cache
    del cache
    torch.cuda.empty_cache()


def load_dataset(model, dataset_name='openwebtext'):
    dataset = transformer_lens.utils.get_dataset(dataset_name)
    class DataClass(torch.utils.data.Dataset):
        def __init__(self, dataset, max_length=1024):
            self.dataset = dataset
            self.max_length = 1024

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            text = self.dataset[idx]['text']
            tokens = model.to_tokens(text)
            tokens = tokens[:self.max_length]
            return tokens
    
    data = DataClass(dataset)
    calibration_data = []
    for batch in tqdm(data):
        if batch.shape[1] == 1024:
            calibration_data.append(batch)
    del data, dataset

    return torch.cat(calibration_data, dim=0)[:8, :]


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.cfg.n_ctx, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    del testloader, _
    torch.cuda.empty_cache()
    return ppl_test 


# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.cfg.n_ctx

    # List to store negative log likelihoods
    nlls = []
    # print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        # if i % 50 == 0:
        #     print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.cfg.n_ctx):(j * model.cfg.n_ctx)].to(device)
        inputs = inputs.reshape(j-i, model.cfg.n_ctx)

        # Forward pass through the model
        lm_logits = model(inputs, return_type="logits")

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.cfg.n_ctx * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.cfg.n_ctx))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()



if __name__ == '__main__':
    gpt2 = sae_lens.HookedSAETransformer.from_pretrained("gpt2-small", device="cuda:3")
    pruned_gpt2 = sae_lens.HookedSAETransformer.from_pretrained("gpt2-small", device="cuda:4")

    dataset = load_dataset(pruned_gpt2)
    prune_sparsegpt(pruned_gpt2, dataset)


    W_Q1 = pruned_gpt2.W_Q.clone().to("cpu")
    W_Q2 = gpt2.W_Q.clone().to("cpu")

    # Compute the mean absolute difference
    diff = torch.abs(W_Q1 - W_Q2).mean().item()

    print(diff)
    print(f'Original GPT-2 perplexity: {eval_ppl(gpt2, gpt2.tokenizer, device="cuda:3")}')
    print(f'SparseGPT Pruned GPT-2 perplexity: {eval_ppl(pruned_gpt2, pruned_gpt2.tokenizer, device="cuda:4")}')




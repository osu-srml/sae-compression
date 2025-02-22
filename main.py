import gc
import itertools
import math
import os
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias


import numpy as np
import pandas as pd
import torch as t
from datasets import load_dataset
import transformer_lens
import sae_lens

import einops
import circuitsvis as cv
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, IFrame, clear_output, display
from jaxtyping import Float, Int
from openai import OpenAI
from rich import print as rprint
from rich.table import Table
from tabulate import tabulate

from tqdm import tqdm

def magnitude(W, sparse_ratio=0.5):
    W_abs = W.abs()
    k = int(W_abs.numel() * sparse_ratio)
    _, indices = W_abs.view(-1).topk(k)
    mask = t.zeros_like(W_abs)
    mask.view(-1)[indices] = 1
    return mask*W

def prune_magnitude(model):
    wts = ['W_Q', 'W_K', 'W_V', 'W_O', 'W_in', 'W_out']
    for name, param in model.named_parameters():
        # print(f"Layer: {name}, Shape: {param.shape}")
        if name.split('.')[-1] in wts:
            if param.dim() == 3:    
                for i in range(param.shape[0]):
                    param[i] = magnitude(param[i])
            else:
                param = magnitude(param)
    
    return model

def wanda(W, X_norm, sparse_ratio=0.5):
    W_metric = W.abs() * X_norm
    _, sorted_idx = W_metric.sort(dim=1)
    pruned_idx = sorted_idx[:, :int(W.shape[1] * sparse_ratio)]
    
    W_clone = W.detach().clone()    
    W_clone.scatter_(dim=1, index=pruned_idx, src=t.zeros_like(pruned_idx, dtype=W.dtype))
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
                    for head in range(W.shape[0]):
                        W[head] = wanda(W[head], X_norm, sparse_ratio=0.5)
                        
                else:
                    for head in range(W.shape[0]):
                        X_norm = X[:, head, :].norm(p=2, dim=0)
                        W[head] = wanda(W[head], X_norm, sparse_ratio=0.5)
            else:
                X_norm = X.norm(p=2, dim=0)
                W = wanda(W, X_norm, sparse_ratio=0.5)
            
    return model



if __name__ == "__main__":
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    t.set_grad_enabled(False)
    print(device)

    gpt2: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained("gpt2-small", device=device)
    pruned_gpt2: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained("gpt2-small", device=device)
    dataset = transformer_lens.utils.get_dataset('openwebtext')

    class OpenWebText(t.utils.data.Dataset):
        def __init__(self, dataset, max_length=1024):
            self.dataset = dataset
            self.max_length = 1024

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            text = self.dataset[idx]['text']
            tokens = gpt2.to_tokens(text)
            tokens = tokens[:self.max_length]
            return tokens
        
    openwebtext = OpenWebText(dataset)
    for batch in tqdm(openwebtext):
        pruned_gpt2 = prune_wanda(pruned_gpt2, batch)
    
    t.save(pruned_gpt2.state_dict(), 'pruned_gpt2_wanda.pth')

    # pruned_gpt2 = prune_magnitude(pruned_gpt2)
    # t.save(pruned_gpt2.state_dict(), 'pruned_gpt2_magnitude.pth')

    example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
    example_answer = " Mary"
    print(transformer_lens.utils.test_prompt(example_prompt, example_answer, gpt2, prepend_bos=True))

    example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
    example_answer = " Mary"
    print(transformer_lens.utils.test_prompt(example_prompt, example_answer, pruned_gpt2, prepend_bos=True))

    print(transformer_lens.evals.sanity_check(gpt2))
    print(transformer_lens.evals.sanity_check(pruned_gpt2))



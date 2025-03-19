import gc
import torch as t
import transformer_lens
import sae_lens
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

@t.no_grad()
def wanda(W, X_norm, sparse_ratio=0.5):
    W_metric = W.abs() * X_norm
    _, sorted_idx = W_metric.sort(dim=1)
    pruned_idx = sorted_idx[:, :int(W.shape[1] * sparse_ratio)]
    
    W_clone = W.detach().clone()    
    W_clone.scatter_(dim=1, index=pruned_idx, src=t.zeros_like(pruned_idx, dtype=W.dtype))
    return W_clone

@t.no_grad()
def prune_wanda(model, tokens):
    wts_act = {
    'attn.W_Q': 'attn.hook_q',
    # 'attn.W_K': 'attn.hook_k',
    # 'attn.W_V': 'attn.hook_v',
    'attn._W_K': 'attn.hook_k',
    'attn._W_V': 'attn.hook_v',
    'attn.W_O': 'hook_attn_out',
    'mlp.W_in': 'mlp.hook_pre',
    'mlp.W_out': 'hook_mlp_out'
    }
    for layer in range(model.cfg.n_layers):
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        del logits
        for wt, act in wts_act.items():
            W = model.get_parameter(f'blocks.{layer}.{wt}')
            X = cache[f'blocks.{layer}.{act}']
            t.cuda.empty_cache()
            gc.collect()

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

        del cache
        t.cuda.empty_cache()
        gc.collect()
                        
    return model





if __name__ == "__main__":
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    t.set_grad_enabled(False)
    print(device)
    pruned_gemma2b: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained("gemma-2-2b", device="cuda", n_devices=4)
    dataset = transformer_lens.utils.get_dataset('openwebtext')

    class OpenWebText(t.utils.data.Dataset):
        def __init__(self, dataset, max_length=1024):
            self.dataset = dataset
            self.max_length = max_length

        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            text = self.dataset[idx]['text']
            tokens = pruned_gemma2b.to_tokens(text)
            tokens = tokens[:self.max_length]
            return tokens
        
    openwebtext = OpenWebText(dataset)

    t.cuda.empty_cache()
    del pruned_gemma2b
    gc.collect()

    # # MAGNITUDE pruning
    # pruned_gemma2b: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained("gemma-2-2b", device="cuda", n_devices=4)
    # pruned_gemma2b = prune_magnitude(pruned_gemma2b)
    # t.save(pruned_gemma2b.state_dict(), 'pruned/gemma-2-2b_magnitude.pth')

    # # WANDA pruning
    # pruned_gemma2b: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained("gemma-2-2b", device="cuda", n_devices=4)
    # i = 0
    # with t.no_grad():
    #     for batch in tqdm(openwebtext):
    #         if i == 128:
    #             break
    #         t.cuda.empty_cache()
    #         gc.collect()
    #         pruned_gemma2b = prune_wanda(pruned_gemma2b, batch)
    #         i = i + 1
    
    # t.save(pruned_gemma2b.state_dict(), 'pruned/gemma-2-2b_wanda.pth')
    # t.cuda.empty_cache()
    # del pruned_gemma2b
    # gc.collect()


    # MAGNITUDE pruning
    pruned_gemma2b: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained("gemma-2b", device="cuda", n_devices=4)
    pruned_gemma2b = prune_magnitude(pruned_gemma2b)
    t.save(pruned_gemma2b.state_dict(), 'pruned/gemma-2b_magnitude.pth')

    # WANDA pruning
    pruned_gemma2b: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained("gemma-2b", device="cuda", n_devices=4)
    i = 0
    with t.no_grad():
        for batch in tqdm(openwebtext):
            if i == 128:
                break
            t.cuda.empty_cache()
            gc.collect()
            pruned_gemma2b = prune_wanda(pruned_gemma2b, batch)
            i = i + 1
    
    t.save(pruned_gemma2b.state_dict(), 'pruned/gemma-2b_wanda.pth')
    t.cuda.empty_cache()
    del pruned_gemma2b
    gc.collect()


    # MAGNITUDE pruning
    pruned_gemma2b: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained("gemma-2b-it", device="cuda", n_devices=4)
    pruned_gemma2b = prune_magnitude(pruned_gemma2b)
    t.save(pruned_gemma2b.state_dict(), 'pruned/gemma-2b-it_magnitude.pth')

    # WANDA pruning
    pruned_gemma2b: sae_lens.HookedSAETransformer = sae_lens.HookedSAETransformer.from_pretrained("gemma-2b-it", device="cuda", n_devices=4)
    i = 0
    with t.no_grad():
        for batch in tqdm(openwebtext):
            if i == 128:
                break
            t.cuda.empty_cache()
            gc.collect()
            pruned_gemma2b = prune_wanda(pruned_gemma2b, batch)
            i = i + 1
    
    t.save(pruned_gemma2b.state_dict(), 'pruned/gemma-2b-it_wanda.pth')
    t.cuda.empty_cache()
    del pruned_gemma2b
    gc.collect()





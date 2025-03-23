import os
import torch as t
import sae_lens
from tqdm import tqdm

if __name__ == "__main__":
    ratios = [0.5, 0.25, 0.75]
    datasets = ["wiki", "pile", "c4", "openwebtext"]
    hook_ids = ["hook_resid_post", "hook_mlp_out", "attn.hook_z"] 
    release = {
        "hook_resid_post":"gemma-scope-2b-pt-res-canonical", 
        "hook_mlp_out": "gemma-scope-2b-pt-mlp-canonical", 
        "attn.hook_z": "gemma-scope-2b-pt-att-canonical"
    }
    path = "/local/scratch/suchit/COLM/pruned_saes/gemma-2-2b/wanda"
    for dataset in datasets:
        for ratio in ratios:
            for hook_id in hook_ids:
                # Layer-wise pruning
                layers = [
                    [0, 21, 22, 23, 24, 25, 10, 20], 
                    [1, 2, 3, 4, 5, 6, 7, 8, 9], 
                    [11, 12, 13, 14, 15, 16, 17, 18, 19]
                ]
                for layer_batch in tqdm(layers, desc="Layer-wise pruning"):
                    sae_dict = {}

                    for layer in layer_batch:
                        pretrained_release = release[hook_id]
                        pretrained_sae_id = f"layer_{layer}/width_16k/canonical"
            
                        sae = sae_lens.SAE.from_pretrained(pretrained_release, pretrained_sae_id, device="cpu")[0]
                        sae.load_state_dict(t.load(f'{path}/{dataset}/{hook_id}_ratio={ratio}/blocks.{layer}.{hook_id}.pth'))

                        sae_dict[f'blocks.{layer}.{hook_id}'] = sae

                    sae_lens.upload_sae(sae_dict, hf_repo_id=f'suchitg/sae-compression-gemma-2-2b-pruned-sae-{dataset}-{ratio}')
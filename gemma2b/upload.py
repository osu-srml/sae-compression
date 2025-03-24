import os
import torch as t
import sae_lens
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="openwebtext")
    parser.add_argument("--device", type=int, default="0")
    args = parser.parse_args()

    ratios = [0.5, 0.25, 0.75]
    dataset = args.dataset
    hook_ids = ["hook_resid_post", "hook_mlp_out", "attn.hook_z"] 
    release = {
        "hook_resid_post":"gemma-scope-2b-pt-res-canonical", 
        "hook_mlp_out": "gemma-scope-2b-pt-mlp-canonical", 
        "attn.hook_z": "gemma-scope-2b-pt-att-canonical"
    }
    path = "/local/scratch/suchit/COLM/pruned_saes/gemma-2-2b/wanda"
    for ratio in ratios:
        for hook_id in hook_ids:
            # Layer-wise pruning
                sae_dict = {}
                for layer in tqdm(range(26), desc="Uploading layers"):
                    pretrained_release = release[hook_id]
                    pretrained_sae_id = f"layer_{layer}/width_16k/canonical"
        
                    sae = sae_lens.SAE.from_pretrained(pretrained_release, pretrained_sae_id, device=f"cuda:{str(args.device)}")[0]
                    sae.load_state_dict(t.load(f'{path}/{dataset}/{hook_id}_ratio={ratio}/blocks.{layer}.{hook_id}.pth'))

                    sae_dict[f'blocks.{layer}.{hook_id}'] = sae

                sae_lens.upload_saes_to_huggingface(sae_dict, hf_repo_id=f'suchitg/sae-compression-gemma-2-2b-pruned-sae-{dataset}-{ratio}')
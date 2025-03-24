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

    # ratios = [0.5, 0.25, 0.75]
    dataset = args.dataset
    hook_ids = ["hook_resid_post", "hook_mlp_out", "hook_z"] 
    # hook_ids = ["hook_resid_post", "attn.hook_z"] 
    # hook_ids = ["hook_mlp_out"] 
    release = {
        "hook_resid_post": "gpt2-small-resid-post-v5-32k", 
        "hook_mlp_out": "gpt2-small-mlp-out-v5-32k", 
        "hook_z": "gpt2-small-hook-z-kk"
    }
    path = "/local/scratch/suchit/COLM/pruned_saes/gpt2-small/wanda"
    # for hook_id in hook_ids:
        # Layer-wise pruning
            # sae_dict = {}
            # sae_dict_0_5 = {}
            # for layer in tqdm(range(12), desc="Uploading layers"):
            #     pretrained_release = release[hook_id]
            #     pretrained_sae_id = f"blocks.{layer}.{hook_id}"
    
            #     sae = sae_lens.SAE.from_pretrained(pretrained_release, pretrained_sae_id, device=f"cuda:{str(args.device)}")[0]
            #     sae.load_state_dict(t.load(f'{path}/{dataset}/{hook_id}/blocks.{layer}.{hook_id}.pth'))
            #     sae_dict[f'blocks.{layer}.{hook_id}'] = sae

            #     # sae = sae_lens.SAE.from_pretrained(pretrained_release, pretrained_sae_id, device=f"cuda:{str(args.device)}")[0]
            #     # if "hook_z" in pretrained_sae_id:
            #     #     sae.load_state_dict(t.load(f'{path}/{dataset}/{hook_id}_ratio=0.5/blocks.{layer}.{hook_id}_0.5.pth'))
            #     # else:
            #     #     sae.load_state_dict(t.load(f'{path}/{dataset}/{hook_id}_ratio=0.5/blocks.{layer}.{hook_id}.pth'))
            #     sae = sae_lens.SAE.from_pretrained(pretrained_release, pretrained_sae_id, device=f"cuda:{str(args.device)}")[0]
            #     sae.load_state_dict(t.load(f'{path}/{dataset}/{hook_id}_ratio=0.5/blocks.{layer}.{hook_id}.pth'))
            #     sae_dict_0_5[f'blocks.{layer}.{hook_id}'] = sae

            # sae_lens.upload_saes_to_huggingface(sae_dict, hf_repo_id=f'suchitg/sae-compression-gpt-2-small-pruned-sae-{dataset}')
            # sae_lens.upload_saes_to_huggingface(sae_dict_0_5, hf_repo_id=f'suchitg/sae-compression-gpt-2-small-pruned-sae-{dataset}_0.5')


    sae_dict_0_25 = {}
    sae_dict_0_75 = {}
    for hook_id in hook_ids:
        for layer in tqdm(range(12), desc="Uploading layers"):
            pretrained_release = release[hook_id]
            pretrained_sae_id = f"blocks.{layer}.{hook_id}"

            sae = sae_lens.SAE.from_pretrained(pretrained_release, pretrained_sae_id, device=f"cuda:{str(args.device)}")[0]
            sae.load_state_dict(t.load(f'{path}/{dataset}/{hook_id}_ratio=0.25/blocks.{layer}.{hook_id}.pth'))

            if "hook_z" in pretrained_sae_id:
                sae_dict_0_25[f'blocks.{layer}.attn.{hook_id}'] = sae
            else:
                sae_dict_0_25[f'blocks.{layer}.{hook_id}'] = sae


            sae = sae_lens.SAE.from_pretrained(pretrained_release, pretrained_sae_id, device=f"cuda:{str(args.device)}")[0]
            sae.load_state_dict(t.load(f'{path}/{dataset}/{hook_id}_ratio=0.75/blocks.{layer}.{hook_id}.pth'))
            if "hook_z" in pretrained_sae_id:
                sae_dict_0_75[f'blocks.{layer}.attn.{hook_id}'] = sae
            else:
                sae_dict_0_75[f'blocks.{layer}.{hook_id}'] = sae

    sae_lens.upload_saes_to_huggingface(sae_dict_0_25, hf_repo_id=f'suchitg/sae-compression-gpt-2-small-pruned-sae-{dataset}_0.25')
    sae_lens.upload_saes_to_huggingface(sae_dict_0_75, hf_repo_id=f'suchitg/sae-compression-gpt-2-small-pruned-sae-{dataset}_0.75')
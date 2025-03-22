Pretrained SAES:

1) hook_resid_post: https://jbloomaus.github.io/SAELens/sae_table/#gpt2-small-resid-post-v5-32k
   Huggingface Repo: jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs
    
    from sae_lens import SAE

    release = "gpt2-small-resid-post-v5-32k"
    sae_id = "blocks.0.hook_resid_post"
    sae = SAE.from_pretrained(release, sae_id)[0]


2) hook_mlp_out: https://jbloomaus.github.io/SAELens/sae_table/#gpt2-small-mlp-out-v5-32k
   Huggingface Repo: jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs

    from sae_lens import SAE

    release = "gpt2-small-mlp-out-v5-32k"
    sae_id = "blocks.0.hook_mlp_out"
    sae = SAE.from_pretrained(release, sae_id)[0]


3) hook_z: https://jbloomaus.github.io/SAELens/sae_table/#gpt2-small-hook-z-kk
   Huggingface Repo: Huggingface Repo: ckkissane/attn-saes-gpt2-small-all-layers

    from sae_lens import SAE

    release = "gpt2-small-hook-z-kk"
    sae_id = "blocks.0.hook_z"
    sae = SAE.from_pretrained(release, sae_id)[0]


4) hook_attn_out: https://jbloomaus.github.io/SAELens/sae_table/#gpt2-small-attn-out-v5-32k
   Huggingface Repo: jbloom/GPT2-Small-OAI-v5-32k-attn-out-SAEs
    
    from sae_lens import SAE

    release = "gpt2-small-attn-out-v5-32k"
    sae_id = "blocks.0.hook_attn_out"
    sae = SAE.from_pretrained(release, sae_id)[0]
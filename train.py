import torch as t
import sae_lens
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner, upload_saes_to_huggingface
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()


    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f'Device using:{device}')

    model = sae_lens.HookedSAETransformer.from_pretrained("gpt2-small", device=device)
    model.load_state_dict(t.load('pruned/pruned_gpt2_wanda.pth'))

    total_training_steps = 30_000  # probably we should do more
    batch_size = 4096
    total_training_tokens = total_training_steps * batch_size

    lr_warm_up_steps = 0
    lr_decay_steps = total_training_steps // 5  # 20% of training
    l1_warm_up_steps = total_training_steps // 20  # 5% of training

    layer = args.layer
    # cfg = LanguageModelSAERunnerConfig(
    #     #
    #     # Data Generating Function (Model + Training Distibuion)
    #     model_name="gpt2-small",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
    #     hook_name=f"blocks.{layer}.attn.hook_z",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
    #     hook_layer=layer,  # Only one layer in the model.
    #     d_in=model.cfg.d_head * model.cfg.n_heads,
    #     dataset_path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
    #     is_dataset_tokenized=True,
    #     streaming=True,  # we could pre-download the token dataset if it was small.
    #     train_batch_size_tokens=batch_size,
    #     context_size=model.cfg.n_ctx,
    #     prepend_bos=True,
    #     #
    #     # SAE architecture
    #     # architecture="gated",
    #     architecture="standard",
    #     expansion_factor=16,
    #     b_dec_init_method="zeros",
    #     apply_b_dec_to_input=True,
    #     normalize_sae_decoder=False,
    #     scale_sparsity_penalty_by_decoder_norm=True,
    #     decoder_heuristic_init=True,
    #     init_encoder_as_decoder_transpose=True,
    #     #
    #     # Activations store
    #     n_batches_in_buffer=64,
    #     training_tokens=total_training_tokens,
    #     store_batch_size_prompts=16,
    #     #
    #     # Training hyperparameters (standard)
    #     lr=1e-4,
    #     adam_beta1=0.9,
    #     adam_beta2=0.999,
    #     lr_scheduler_name="constant",
    #     lr_warm_up_steps=lr_warm_up_steps,  # avoids large number of initial dead features
    #     lr_decay_steps=lr_decay_steps,
    #     #
    #     # Training hyperparameters (SAE-specific)
    #     l1_coefficient=2,
    #     l1_warm_up_steps=l1_warm_up_steps,
    #     use_ghost_grads=False,  # we don't use ghost grads anymore
    #     feature_sampling_window=1000,  # how often we resample dead features
    #     dead_feature_window=500,  # size of window to assess whether a feature is dead
    #     dead_feature_threshold=1e-4,  # threshold for classifying feature as dead, over window
    #     #
    #     # Logging / evals
    #     log_to_wandb=True,  # always use wandb unless you are just testing code.
    #     # wandb_project="sae_compression",
    #     wandb_project="sae_test",
    #     wandb_log_frequency=30,
    #     eval_every_n_wandb_logs=20,
    #     #
    #     # Misc.
    #     device=str(device),
    #     seed=42,
    #     n_checkpoints=5,
    #     checkpoint_path="checkpoints",
    #     dtype="float32",
    # )

    cfg = LanguageModelSAERunnerConfig(
        #
        # Data Generating Function (Model + Training Distibuion)
        model_name="gpt2-small",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
        hook_name=f"blocks.{layer}.attn.hook_z",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
        hook_layer=layer,  # Only one layer in the model.
        d_in=model.cfg.d_head * model.cfg.n_heads,
        d_sae=24576,
        dataset_path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        is_dataset_tokenized=True,
        streaming=True,  # we could pre-download the token dataset if it was small.
        train_batch_size_tokens=batch_size,
        context_size=128,
        prepend_bos=True,
        #
        # SAE architecture
        # architecture="gated",
        architecture="standard",
        # expansion_factor=32,
        b_dec_init_method="zeros",
        apply_b_dec_to_input=True,
        normalize_sae_decoder=False,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,
        #
        # Activations store
        n_batches_in_buffer=64,
        training_tokens=total_training_tokens,
        store_batch_size_prompts=16,
        #
        # Training hyperparameters (standard)
        lr=0.0012,
        adam_beta1=0.0,
        adam_beta2=0.9099,
        lr_scheduler_name="constant",
        lr_warm_up_steps=lr_warm_up_steps,  # avoids large number of initial dead features
        lr_decay_steps=lr_decay_steps,
        #
        # Training hyperparameters (SAE-specific)
        l1_coefficient=0.5,
        l1_warm_up_steps=l1_warm_up_steps,
        use_ghost_grads=False,  # we don't use ghost grads anymore
        feature_sampling_window=1000,  # how often we resample dead features
        dead_feature_window=500,  # size of window to assess whether a feature is dead
        dead_feature_threshold=1e-4,  # threshold for classifying feature as dead, over window
        #
        # Logging / evals
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        # wandb_project="sae_compression",
        wandb_project="sae_test",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
        #
        # Misc.
        device=str(device),
        seed=42,
        # n_checkpoints=5,
        # checkpoint_path=None,
        dtype="float32",
    )

    t.set_grad_enabled(True)
    # runner = SAETrainingRunner(cfg, override_model=model)
    runner = SAETrainingRunner(cfg)
    sae = runner.run()

    # hf_repo_id = "suchitg/sae_wanda"
    hf_repo_id = "suchitg/sae_test"
    sae_id = f"{cfg.hook_name}-attn-sae-v2"
    upload_saes_to_huggingface({sae_id: sae}, hf_repo_id=hf_repo_id)

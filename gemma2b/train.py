import torch as t
import sae_lens
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner, upload_saes_to_huggingface
from tqdm import tqdm
import argparse
import logging

logging.getLogger("sae_lens").setLevel(logging.CRITICAL)

if __name__ == "__main__":
    # Parse Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    layer, device_id = args.layer, args.device
    
    device = t.device(f"cuda:{str(device_id)}" if t.cuda.is_available() else "cpu")
    # device = t.device(f"cuda:1" if t.cuda.is_available() else "cpu")
    print(f'Device using:{device}')

    model = sae_lens.HookedSAETransformer.from_pretrained("gemma-2-2b", device=device)
    model.load_state_dict(t.load('pruned/gemma-2-2b_wanda.pth'))
    hooks = ['hook_resid_post', 'hook_mlp_out', 'attn.hook_z']
    # layers = [24]


      
    # for layer in layers:
    for hook in hooks:
        if "attn" in hook:
            din = 2048
        else:
            din = 2304

        print(f'Layer: {layer}, Hook: {hook}')
        cfg = LanguageModelSAERunnerConfig(
            architecture="jumprelu",
            model_name="gemma-2-2b",
            model_class_name="HookedTransformer",
            # hook_name="blocks.10.hook_resid_post",
            hook_name=f"blocks.{layer}.{hook}",
            hook_eval="NOT_IN_USE",
            # hook_layer=10,
            hook_layer=layer,
            hook_head_index=None,
            dataset_path="monology/pile-uncopyrighted",
            dataset_trust_remote_code=True,
            streaming=True,
            context_size=1024,
            d_in=din,
            d_sae=16384,
            b_dec_init_method="zeros",
            activation_fn="relu",
            normalize_sae_decoder=False,
            apply_b_dec_to_input=False,
            decoder_orthogonal_init=False,
            decoder_heuristic_init=True,
            init_encoder_as_decoder_transpose=True,
            # n_batches_in_buffer=20 --Default
            n_batches_in_buffer=128,
            # training_tokens=2000000 --Default
            # store_batch_size_prompts=32 --Default
            train_batch_size_tokens=4096,
            normalize_activations="none",
            device="cuda:1",
            act_store_device="cpu",
            seed=42,
            dtype="float32",
            prepend_bos=True,
            jumprelu_init_threshold=0.001,
            jumprelu_bandwidth=0.001,
            autocast=True,
            adam_beta1=0.0,
            adam_beta2=0.999,
            # mse_loss_normalization=None --Default
            # l1_coefficient=0.001 --Default
            # lp_norm=1 --Default
            # scale_sparsity_penalty_by_decoder_norm=False --Default
            # l1_warm_up_steps=0 --Default
            # lr=7e-5,
            # lr_scheduler_name="constant", --Default
            # lr_warm_up_steps=10000,
            lr_warm_up_steps=0,
            # lr_end=None --Default
            # lr_decay_steps=0 --Default
            lr_decay_steps=10000,
            # use_ghost_grads=False --Default
            # feature_sampling_window=2000 --Default
            # dead_feature_window=1000 --Default
            # dead_feature_threshold=1e-8 --Default
            run_name=f"blocks.{layer}.{hook}",
            log_to_wandb=True,
            wandb_project="sae-compression-gemma-2-2b",
            wandb_log_frequency=30,
            eval_every_n_wandb_logs=20,
            n_checkpoints=0,
            checkpoint_path="/local/scratch/suchit/COLM/checkpoints",
        )

        t.set_grad_enabled(True)
        runner = SAETrainingRunner(cfg, override_model=model)
        sae = runner.run()

        # hf_repo_id = "suchitg/sae_test"
        hf_repo_id = "suchitg/sae-compression-gemma-2-2b"
        sae_id = f"{cfg.hook_name}"
        upload_saes_to_huggingface({sae_id: sae}, hf_repo_id=hf_repo_id)

        del runner, sae
        t.cuda.empty_cache()


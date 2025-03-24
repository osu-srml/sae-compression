import sae_bench.evals.core.main as core

# selected_saes = [
#     ('gpt2-small-hook-z-kk', "blocks.0.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.1.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.2.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.3.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.4.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.5.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.6.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.7.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.8.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.9.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.10.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.11.hook_z"),
#    ]

# datasets = [
#     "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
#     "apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2",
#     "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
#     "lighteval/MATH"
#     ]

# # # pruned_on = ['openwebtext', 'pile', 'c4', 'wiki']
# # pruned_on = ['pile', 'c4', 'wiki']

# # i = 1
# # for dataset in datasets:
# #     for pruned in pruned_on:
# #         _ = core.multiple_evals(
# #             selected_saes=selected_saes,
# #             n_eval_reconstruction_batches=200,
# #             n_eval_sparsity_variance_batches=2000,
# #             eval_batch_size_prompts=32,
# #             compute_featurewise_density_statistics=True,
# #             compute_featurewise_weight_based_metrics=True,
# #             exclude_special_tokens_from_reconstruction=True,
# #             dataset=dataset,
# #             context_size=128,
# #             output_folder=f"pruned/core_with_LOADED_sae_pruned_on_{pruned}_{i}",
# #             verbose=True,
# #             load_sae_from_disk=True,
# #             pruned_on=pruned
# #             )
# #     i = i + 1




# selected_saes = [
#    ('gpt2-small-hook-z-kk', "blocks.0.hook_z"),
#    ('gpt2-small-hook-z-kk', "blocks.1.hook_z"),
#    ('gpt2-small-hook-z-kk', "blocks.2.hook_z"),
#    ('gpt2-small-hook-z-kk', "blocks.3.hook_z"),
#    ('gpt2-small-hook-z-kk', "blocks.4.hook_z"),
#    ('gpt2-small-hook-z-kk', "blocks.5.hook_z"),
#    ('gpt2-small-hook-z-kk', "blocks.6.hook_z"),
#    ('gpt2-small-hook-z-kk', "blocks.7.hook_z"),
#    ('gpt2-small-hook-z-kk', "blocks.8.hook_z"),
#    ('gpt2-small-hook-z-kk', "blocks.9.hook_z"),
#    ('gpt2-small-hook-z-kk', "blocks.10.hook_z"),
#    ('gpt2-small-hook-z-kk', "blocks.11.hook_z"),
#    ]

# # i = 1
# # for dataset in datasets:
# #     for pruned in pruned_on:
# #         _ = core.multiple_evals(
# #             selected_saes=selected_saes,
# #             n_eval_reconstruction_batches=200,
# #             n_eval_sparsity_variance_batches=2000,
# #             eval_batch_size_prompts=32,
# #             compute_featurewise_density_statistics=True,
# #             compute_featurewise_weight_based_metrics=True,
# #             exclude_special_tokens_from_reconstruction=True,
# #             dataset=dataset,
# #             context_size=128,
# #             output_folder=f"pruned/core_with_sae_pruned_on_{pruned}_{i}",
# #             verbose=True,
# #             )
# #     i = i + 1

# _ = core.multiple_evals(
#     selected_saes=selected_saes,
#     n_eval_reconstruction_batches=200,
#     n_eval_sparsity_variance_batches=2000,
#     eval_batch_size_prompts=32,
#     compute_featurewise_density_statistics=True,
#     compute_featurewise_weight_based_metrics=True,
#     exclude_special_tokens_from_reconstruction=True,
#     dataset="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
#     context_size=128,
#     output_folder=f"pruned/core_with_pretrained_sae_1",
#     verbose=True,
#     )



# selected_saes = [
#    ('suchitg/sae_test', "blocks.0.attn.hook_z-attn-sae-v-my_cfg"),
#    ('suchitg/sae_test', "blocks.1.attn.hook_z-attn-sae-v-my_cfg"),
#    ('suchitg/sae_test', "blocks.2.attn.hook_z-attn-sae-v-my_cfg"),
#    ('suchitg/sae_test', "blocks.3.attn.hook_z-attn-sae-v-my_cfg"),
#    ('suchitg/sae_test', "blocks.4.attn.hook_z-attn-sae-v-my_cfg"),
#    ('suchitg/sae_test', "blocks.5.attn.hook_z-attn-sae-v-my_cfg"),
#    ('suchitg/sae_test', "blocks.6.attn.hook_z-attn-sae-v-my_cfg"),
#    ('suchitg/sae_test', "blocks.7.attn.hook_z-attn-sae-v-my_cfg"),
#    ('suchitg/sae_test', "blocks.8.attn.hook_z-attn-sae-v-my_cfg"),
#    ('suchitg/sae_test', "blocks.9.attn.hook_z-attn-sae-v-my_cfg"),
#    ('suchitg/sae_test', "blocks.10.attn.hook_z-attn-sae-v-my_cfg"),
#    ('suchitg/sae_test', "blocks.11.attn.hook_z-attn-sae-v-my_cfg")
#    ]

# _ = core.multiple_evals(
#     selected_saes=selected_saes,
#     n_eval_reconstruction_batches=200,
#     n_eval_sparsity_variance_batches=2000,
#     eval_batch_size_prompts=32,
#     compute_featurewise_density_statistics=True,
#     compute_featurewise_weight_based_metrics=True,
#     exclude_special_tokens_from_reconstruction=True,
#     dataset="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
#     context_size=128,
#     output_folder=f"pruned/core_with_sae_trained_on_pruned_gpt2_1",
#     verbose=True,
#     )


import torch
import sae_bench.custom_saes.run_all_evals_custom_saes as run_all_evals_custom_saes

model_name = 'gemma-2-2b'
llm_batch_size = 1024
torch_dtype = torch.float32
str_dtype = torch_dtype.__str__().split(".")[-1]
save_activations = False
RANDOM_SEED = 42

device = 'cuda:0'
eval_types = [
    "absorption",
    "core",
    "scr",
    "tpp",
    "sparse_probing",
]

# # Load from disk
# output_folder = 'pruned_sae'
# selected_saes = [
#     ('gpt2-small-hook-z-kk', "blocks.0.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.1.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.2.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.3.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.4.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.5.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.6.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.7.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.8.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.9.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.10.hook_z"),
#     ('gpt2-small-hook-z-kk', "blocks.11.hook_z"),
#    ]

# _ = run_all_evals_custom_saes.run_evals(
#     model_name,
#     selected_saes,
#     llm_batch_size,
#     str_dtype,
#     device,
#     eval_types,
#     api_key=None,
#     force_rerun=False,
#     save_activations=save_activations,
#     output_folder = output_folder,
#     load_sae_from_disk=True,
#     pruned_on='openwebtext'
# )

# Pretrained
output_folder = 'pretrained_sae'
selected_saes = [
   ("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical"),
   ]

_ = run_all_evals_custom_saes.run_evals(
    model_name,
    selected_saes,
    llm_batch_size,
    str_dtype,
    device,
    eval_types,
    api_key=None,
    force_rerun=False,
    save_activations=save_activations,
    output_folder = output_folder
)


# Trained on pruned
output_folder = 'sae_on_pruned'
selected_saes = [
   ("suchitg/sae-compression-gemma-2-2b", "blocks.12.hook_resid_post"),
   ]

_ = run_all_evals_custom_saes.run_evals(
    model_name,
    selected_saes,
    llm_batch_size,
    str_dtype,
    device,
    eval_types,
    api_key=None,
    force_rerun=False,
    save_activations=save_activations,
    output_folder = output_folder
)



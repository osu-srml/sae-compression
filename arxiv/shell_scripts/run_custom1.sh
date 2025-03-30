#!/bin/bash
model_name="gemma-2-2b"

# User configuration
sae_regex_patterns=(
    # "suchitg/sae-compression-gemma-2-2b-pruned-sae-pile-0.25"
    "suchitg/sae-compression-gemma-2-2b-pruned-sae-pile-0.5"
    # "suchitg/sae-compression-gemma-2-2b-trained-sae-pile-wanda"
    # "gemma-scope-2b-pt-att-canonical"
    # "gemma-scope-2b-pt-mlp-canonical"
    # "gemma-scope-2b-pt-res-canonical"
)



# Create array of patterns
declare -a sae_block_patterns=(
    # layer_12/width_16k/canonical
    "blocks.12.hook_resid_post"
    # "blocks.12.attn.hook_z"
    "blocks.12.hook_mlp_out"
)




# for sae_regex_pattern in "${sae_regex_patterns[@]}"; do
#     for sae_block_pattern in "${sae_block_patterns[@]}"; do
#         echo "Starting pattern ${sae_block_pattern}..."
#         python sae_bench/evals/absorption/main.py \
#             --sae_regex_pattern "${sae_regex_pattern}" \
#             --sae_block_pattern "${sae_block_pattern}" \
#             --output_folder /local/scratch/suchit/COLM/eval_results/absorption \
#             --model_name ${model_name} || {
#                 echo "Pattern ${sae_block_pattern} failed, continuing to next pattern..."
#                 continue
#             }
#         echo "Completed pattern ${sae_block_pattern}"
#     done
# done

# for sae_regex_pattern in "${sae_regex_patterns[@]}"; do
#     for sae_block_pattern in "${sae_block_patterns[@]}"; do
#         echo "Starting core eval for pattern ${sae_block_pattern}..."
#         python sae_bench/evals/core/main.py "${sae_regex_pattern}" "${sae_block_pattern}" \
#         --batch_size_prompts 16 \
#         --n_eval_sparsity_variance_batches 2000 \
#         --n_eval_reconstruction_batches 200 \
#         --output_folder /local/scratch/suchit/COLM/eval_results/core \
#         --exclude_special_tokens_from_reconstruction --verbose --llm_dtype bfloat16 || {
#             echo "Core eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
#             continue
#         }
#         echo "Completed core eval for pattern ${sae_block_pattern}"
#     done
# done

# for sae_regex_pattern in "${sae_regex_patterns[@]}"; do
#     for sae_block_pattern in "${sae_block_patterns[@]}"; do
#         echo "Starting SCR eval for pattern ${sae_block_pattern}..."
#         python sae_bench/evals/scr_and_tpp/main.py \
#         --sae_regex_pattern "${sae_regex_pattern}" \
#         --sae_block_pattern "${sae_block_pattern}" \
#         --model_name ${model_name} \
#         --output_folder /local/scratch/suchit/COLM/eval_results/scr \
#         --artifacts_path /local/scratch/suchit/COLM/artifacts \
#         --perform_scr true \
#         --clean_up_activations || {
#             echo "SCR eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
#             continue
#         }
#         echo "Completed SCR eval for pattern ${sae_block_pattern}"
#     done
# done


# for sae_regex_pattern in "${sae_regex_patterns[@]}"; do
#     for sae_block_pattern in "${sae_block_patterns[@]}"; do
#         echo "Starting TPP eval for pattern ${sae_block_pattern}..."
#         python sae_bench/evals/scr_and_tpp/main.py \
#         --sae_regex_pattern "${sae_regex_pattern}" \
#         --sae_block_pattern "${sae_block_pattern}" \
#         --model_name ${model_name} \
#         --output_folder /local/scratch/suchit/COLM/eval_results/tpp \
#         --artifacts_path /local/scratch/suchit/COLM/artifacts \
#         --perform_scr false \
#         --clean_up_activations || {
#             echo "TPP eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
#             continue
#         }
#         echo "Completed TPP eval for pattern ${sae_block_pattern}"
#     done
# done

# for sae_regex_pattern in "${sae_regex_patterns[@]}"; do
#     for sae_block_pattern in "${sae_block_patterns[@]}"; do
#         echo "Starting sparse probing for pattern ${sae_block_pattern}..."
#         python sae_bench/evals/sparse_probing/main.py \
#         --sae_regex_pattern "${sae_regex_pattern}" \
#         --sae_block_pattern "${sae_block_pattern}" \
#         --model_name ${model_name} \
#         --output_folder /local/scratch/suchit/COLM/eval_results/sparse_probing \
#         --artifacts_path /local/scratch/suchit/COLM/artifacts \
#         --clean_up_activations || {
#             echo "Sparse probing for pattern ${sae_block_pattern} failed, continuing to next pattern..."
#             continue
#         }
#         echo "Completed sparse probing for pattern ${sae_block_pattern}"
#     done
# done


for sae_regex_pattern in "${sae_regex_patterns[@]}"; do
    for sae_block_pattern in "${sae_block_patterns[@]}"; do
        echo "Starting ravel for pattern ${sae_block_pattern}..."
        python sae_bench/evals/ravel/main.py \
        --sae_regex_pattern "${sae_regex_pattern}" \
        --sae_block_pattern "${sae_block_pattern}" \
        --output_folder /local/scratch/suchit/COLM/${model_name}/eval_results/ravel \
        --artifacts_path /local/scratch/suchit/COLM/${model_name}/artifacts \
        --model_name ${model_name} || {
            echo "Ravel for pattern ${sae_block_pattern} failed, continuing to next pattern..."
            continue
        }
        echo "Completed ravel for pattern ${sae_block_pattern}"
    done
done


# User configuration
# sae_regex_pattern="gemma-scope-2b-pt-res-canonical"
sae_regex_patterns=(
  "gemma-scope-2b-pt-att-canonical"
#   "gemma-scope-2b-pt-mlp-canonical"
#   "gemma-scope-2b-pt-res-canonical"
)
model_name="gemma-2-2b"
# model_name_it="gemma-2-2b-it"

# DEVICE="$1"

# # Create array of patterns
# declare -a sae_block_patterns=(
#     # ".*layer_5.*(16k).*"
#     ".*layer_12.*(16k).canonical"
#     # ".*layer_15.*(16k).canonical"
#     # ".*layer_19.*(16k).*"
# )

# Create array of patterns
declare -a sae_block_patterns=(
    ".*layer_12.*(16k).canonical"
)

# for sae_block_pattern in "${sae_block_patterns[@]}"; do
#     echo "Starting pattern ${sae_block_pattern}..."
#     python sae_bench/evals/absorption/main.py \
#         --sae_regex_pattern "${sae_regex_pattern}" \
#         --sae_block_pattern "${sae_block_pattern}" \
#         --output_folder /local/scratch/suchit/COLM/eval_results \
#         --model_name ${model_name} || {
#             echo "Pattern ${sae_block_pattern} failed, continuing to next pattern..."
#             continue
#         }
#     echo "Completed pattern ${sae_block_pattern}"
# done

# for sae_block_pattern in "${sae_block_patterns[@]}"; do
#     echo "Starting pattern ${sae_block_pattern}..."
#     python sae_bench/evals/autointerp/main.py \
#         --sae_regex_pattern "${sae_regex_pattern}" \
#         --sae_block_pattern "${sae_block_pattern}" \
#         --model_name ${model_name} || {
#             echo "Pattern ${sae_block_pattern} failed, continuing to next pattern..."
#             continue
#         }
#     echo "Completed pattern ${sae_block_pattern}"
# done

# for sae_regex_pattern in "${sae_regex_patterns[@]}"; do
#     for sae_block_pattern in "${sae_block_patterns[@]}"; do
#         echo "Starting core eval for pattern ${sae_block_pattern}..."
#         python sae_bench/evals/core/main.py "${sae_regex_pattern}" "${sae_block_pattern}" \
#         --batch_size_prompts 16 \
#         --n_eval_sparsity_variance_batches 2000 \
#         --n_eval_reconstruction_batches 200 \
#         --output_folder /local/scratch/suchit/COLM/eval_results/core \
#         --exclude_special_tokens_from_reconstruction --verbose --llm_dtype float32 || {
#             echo "Core eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
#             continue
#         }
#         echo "Completed core eval for pattern ${sae_block_pattern}"
#     done
# done

# for sae_block_pattern in "${sae_block_patterns[@]}"; do
#     echo "Starting SCR eval for pattern ${sae_block_pattern}..."
#     python sae_bench/evals/scr_and_tpp/main.py \
#     --sae_regex_pattern "${sae_regex_pattern}" \
#     --sae_block_pattern "${sae_block_pattern}" \
#     --model_name ${model_name} \
#     --output_folder /local/scratch/suchit/COLM/eval_results \
#     --artifacts_path /local/scratch/suchit/COLM/artifacts \
#     --perform_scr true \
#     --clean_up_activations || {
#         echo "SCR eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
#         continue
#     }
#     echo "Completed SCR eval for pattern ${sae_block_pattern}"
# done

# for sae_block_pattern in "${sae_block_patterns[@]}"; do
#     echo "Starting TPP eval for pattern ${sae_block_pattern}..."
#     python sae_bench/evals/scr_and_tpp/main.py \
#     --sae_regex_pattern "${sae_regex_pattern}" \
#     --sae_block_pattern "${sae_block_pattern}" \
#     --model_name ${model_name} \
#     --output_folder /local/scratch/suchit/COLM/eval_results \
#     --artifacts_path /local/scratch/suchit/COLM/artifacts \
#     --perform_scr false \
#     --clean_up_activations || {
#         echo "TPP eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
#         continue
#     }
#     echo "Completed TPP eval for pattern ${sae_block_pattern}"
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

# for sae_block_pattern in "${sae_block_patterns[@]}"; do
#     echo "Starting ravel for pattern ${sae_block_pattern}..."
#     python sae_bench/evals/ravel/main.py \
#     --sae_regex_pattern "${sae_regex_pattern}" \
#     --sae_block_pattern "${sae_block_pattern}" \
#     --output_folder /local/scratch/suchit/COLM/${model_name}/eval_results/ravel \
#     --artifacts_path /local/scratch/suchit/COLM/${model_name}/artifacts \
#     --model_name ${model_name} || {
#         echo "Ravel for pattern ${sae_block_pattern} failed, continuing to next pattern..."
#         continue
#     }
#     echo "Completed ravel for pattern ${sae_block_pattern}"
# done


# for sae_block_pattern in "${sae_block_patterns[@]}"; do
#     echo "Starting unlearning for pattern ${sae_block_pattern}..."
#     python sae_bench/evals/unlearning/main.py \
#     --sae_regex_pattern "${sae_regex_pattern}" \
#     --sae_block_pattern "${sae_block_pattern}" \
#     --model_name ${model_name_it} || {
#         echo "Unlearning for pattern ${sae_block_pattern} failed, continuing to next pattern..."
#         continue
#     }
#     echo "Completed unlearning for pattern ${sae_block_pattern}"
# done

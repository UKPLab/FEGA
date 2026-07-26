# import os

# import torch

# import sae_bench.custom_saes.custom_sae_config as custom_sae_config
# import sae_bench.custom_saes.relu_sae as relu_sae
# import sae_bench.custom_saes.run_all_evals_custom_saes as run_all_evals_custom_saes
# import sae_bench.evals.core.main as core
# import sae_bench.evals.sparse_probing.main as sparse_probing
# import sae_bench.sae_bench_utils.general_utils as general_utils
# from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex

# DEVICE="cuda:1" #set device based on availability
# REPO_ID= "canrager/saebench_gemma-2-2b_width-2pow16_date-0107" #using gemma-2-2b SAEs of width 65k
# MODEL_NAME="google/gemma-2-2b"
# RANDOM_SEED = 42

# output_folders = {
#     "absorption": "custom_eval_results/absorption",
#     "sparse_probing": "custom_eval_results/sparse_probing",
#     "unlearning": "custom_eval_results/unlearning",
#     "ravel": "custom_eval_results/ravel",
#     "scr": "custom_eval_results/scr",
#     "tpp": "custom_eval_results/tpp"
# }

# eval_types = [
#     "absorption",
#     "sparse_probing",
#     "unlearning",
#     "ravel",
#     "scr",
#     "tpp"
# ]

# for key in eval_types:
#     #create the output folders for every key if they don't exist
#     continue

# llm_batch_size = 512
# torch_dtype = torch.float32
# str_dtype = torch_dtype.__str__().split(".")[-1]

# save_activations = False

# hook_layer = 12
# hook_name = f"blocks.{hook_layer}.hook_resid_post"
# sae=None

# for sae_type in "batch_top_k", "gated", "jump_relu", "matryoshka", "p_anneal", "standard_new", "top_k":
#     checkpoint = f"gemma-2-2b_{sae_type}_width-2pow16_date-0107/resid_post_layer_12/trainer_5/ae.pt"

#     if sae_type=="batch_top_k":
#         continue
#     elif sae_type=="gated":
#         continue
#     elif sae_type=="jump_relu":
#         continue
#     elif sae_type=="matryoshka":
#         continue
#     elif sae_type=="p_anneal":
#         continue
#     elif sae_type=="standard_new":
#         sae = relu_sae.load_dictionary_learning_relu_sae(
#             REPO_ID, checkpoint, MODEL_NAME, DEVICE, torch_dtype, layer=hook_layer
#             )
        
#     elif sae_type=="top_k":
#         continue
#     else:
#         raise ValueError("SAE type not implemented!")

# #Some sanity checks
# print(f"sae dtype: {sae.dtype}, device: {sae.device}")
# d_sae, d_in = sae.W_dec.data.shape
# assert d_sae >= d_in
# print(f"d_in: {d_in}, d_sae: {d_sae}")

import os
import torch
import argparse
import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    run_evals,
    get_all_hf_repo_autoencoders,
)

def main():
    """
    Main function to run SAEBench evaluations on all specified SAE architectures
    for Gemma-2-2B. This script iterates through each architecture, finds all
    corresponding SAEs in the repository, and runs the full evaluation suite.
    """
    parser = argparse.ArgumentParser(description="Run SAEBench evaluations for Gemma-2-2B SAEs.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the evaluations on.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the language model. Default is tuned for Gemma-2-2B on a 24GB GPU.")
    args = parser.parse_args()

    # --- Configuration ---
    REPO_ID = "canrager/saebench_gemma-2-2b_width-2pow16_date-0107"
    MODEL_NAME = "gemma-2-2b"
    RANDOM_SEED = 42
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    str_dtype = "bfloat16" if torch_dtype == torch.bfloat16 else "float32"
    
    eval_types = [
        "absorption",
        "sparse_probing",
        "unlearning",
        "ravel",
        "scr",
        "tpp"
]
    
    sae_architectures = {
        "ReLU": "standard_new",
        "TopK": "top_k",
        "BatchTopK": "batch_top_k",
        "Gated": "gated",
        "JumpReLU": "jump_relu",
        "Matryoshka": "matryoshka",
        "P-Anneal": "p_anneal",
    }

    # --- Get API Key for AutoInterp ---
    api_key = None
    if "autointerp" in eval_types:
        try:
            with open("openai_api_key.txt") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            print("Warning: openai_api_key.txt not found. Skipping autointerp evaluation.")
            eval_types.remove("autointerp")

    # --- Fetch all SAE locations from the Hugging Face Repo ---
    print(f"Fetching all SAE locations from repo: {REPO_ID}")
    all_sae_locations = get_all_hf_repo_autoencoders(REPO_ID)
    print(f"Found a total of {len(all_sae_locations)} SAE configurations.")

    # --- Main Evaluation Loop ---
    for arch_name, arch_pattern in sae_architectures.items():
        print("-----------------------------------------------------")
        print(f"Starting evaluation for architecture: {arch_name}")
        print("-----------------------------------------------------")
        
        # Filter the locations for the current architecture and layer 12
        include_keywords = [arch_pattern, "resid_post_layer_12"]
        exclude_keywords = ["checkpoints"]
        
        arch_sae_locations = general_utils.filter_keywords(
            all_sae_locations,
            exclude_keywords=exclude_keywords,
            include_keywords=include_keywords,
        )
        
        if not arch_sae_locations:
            print(f"No SAEs found for architecture '{arch_name}'. Skipping.")
            continue
            
        print(f"Found {len(arch_sae_locations)} SAEs to evaluate for '{arch_name}'.")

        # --- Run Evaluations for the current architecture ---
        run_evals(
            repo_id=REPO_ID,
            model_name=MODEL_NAME,
            sae_locations=arch_sae_locations,
            llm_batch_size=args.batch_size,
            llm_dtype=str_dtype,
            device=args.device,
            eval_types=eval_types,
            api_key=api_key,
            random_seed=RANDOM_SEED,
            force_rerun=False, # Set to True to overwrite existing results
        )
        
        print(f"Finished evaluation for architecture: {arch_name}\n")

    print("All evaluations complete!")
    print(f"Results saved in the 'eval_results/' directory.")

if __name__ == "__main__":
    main()
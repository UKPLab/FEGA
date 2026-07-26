import argparse
import os
from pathlib import Path

import sae_bench.evals.ravel.main as ravel
import sae_bench.sae_bench_utils.general_utils as general_utils
import torch
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    get_all_hf_repo_autoencoders,
    load_dictionary_learning_sae,
)
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig

torch._dynamo.config.suppress_errors = True
os.environ["TORCH_LOGS"] = "recompiles"  # see what's changing
torch._dynamo.config.cache_size_limit = 2048
torch._dynamo.config.recompile_limit = 64


REPO_ROOT = Path(__file__).resolve().parent.parent
RAVEL_RESULTS = Path("data/instance_eval_results/ravel")
RAVEL_ARTIFACTS = Path("data/artifacts/ravel")
RAVEL_MDBMS = Path("data/mdbm/ravel")
DOWNLOAD_SAES = Path("data/downloaded_saes")


def main():
    """
    Main function to run SAEBench evaluations and save per-instance results
    for detailed failure analysis.
    """
    os.chdir(REPO_ROOT)

    # --- Performance Optimization ---
    # Enable TensorFloat32 for better performance on compatible GPUs.
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(
        description="Run SAEBench evaluations for Gemma-2-2B SAEs with instance-level outputs."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run the evaluations on."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for the language model."
    )
    args = parser.parse_args()

    # --- Configuration ---
    REPO_ID = "canrager/saebench_gemma-2-2b_width-2pow16_date-0107"
    MODEL_NAME = "gemma-2-2b"
    TRAINER = "trainer_2"
    RANDOM_SEED = 42

    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    str_dtype = "bfloat16" if torch_dtype == torch.bfloat16 else "float32"

    # Specify which evaluations to run
    eval_types = [
        "ravel",
        # "absorption",
        # "sparse_probing",
        # "unlearning",
        # "scr",
        # "tpp"
    ]

    sae_architectures = {
        # "ReLU": "standard_new",
        "TopK": "top_k",
        # "BatchTopK": "batch_top_k",
        # "Gated": "gated",
        # "JumpReLU": "jump_relu",
        # "Matryoshka": "matryoshka_batch_top_k",
        # "P-Anneal": "p_anneal",
    }

    # --- Setup Output Directories ---
    output_folders = {
        eval_type: Path("data/instance_eval_results") / eval_type
        for eval_type in eval_types
    }
    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)

    # --- Fetch all SAE locations from the Hugging Face Repo ---
    print(f"Fetching all SAE locations from repo: {REPO_ID}")
    all_sae_locations = get_all_hf_repo_autoencoders(
        REPO_ID, download_location=str(DOWNLOAD_SAES)
    )
    print(f"Found a total of {len(all_sae_locations)} SAE configurations.")

    # --- Main Evaluation Loop ---
    for arch_name, arch_pattern in sae_architectures.items():
        print("-----------------------------------------------------")
        print(f"Starting evaluation for architecture: {arch_name}")
        print("-----------------------------------------------------")

        include_keywords = [arch_pattern, "resid_post_layer_12", TRAINER]
        exclude_keywords = ["checkpoints"]
        # Prevent the TopK substring from selecting broader BatchTopK paths.
        if arch_name == "TopK":
            exclude_keywords.append("batch_top_k")

        arch_sae_locations = general_utils.filter_keywords(
            all_sae_locations,
            exclude_keywords=exclude_keywords,
            include_keywords=include_keywords,
        )

        if not arch_sae_locations:
            print(f"No SAEs found for architecture '{arch_name}'. Skipping.")
            continue

        print(f"Found {len(arch_sae_locations)} SAEs to evaluate for '{arch_name}'.")

        # Loop through each individual SAE for the current architecture
        for i, sae_location in enumerate(arch_sae_locations):
            print(
                f"\n--- Evaluating SAE {i + 1}/{len(arch_sae_locations)}: {sae_location} ---"
            )

            sae = load_dictionary_learning_sae(
                repo_id=REPO_ID,
                location=sae_location,
                model_name=MODEL_NAME,
                device=args.device,
                dtype=torch_dtype,
                download_location=str(DOWNLOAD_SAES),
            )
            unique_sae_id = f"{REPO_ID.split('/')[1]}_{sae_location.replace('/', '_')}"
            selected_saes = [(unique_sae_id, sae)]

            # Run each specified evaluation for the currently loaded SAE
            for eval_type in eval_types:
                try:
                    print(f"  -> Running '{eval_type}' evaluation...")

                    if eval_type == "ravel":
                        cfg = RAVELEvalConfig(
                            model_name=MODEL_NAME,
                            random_seed=RANDOM_SEED,
                            llm_batch_size=args.batch_size // 4,
                            llm_dtype=str_dtype,
                            artifact_dir=str(RAVEL_ARTIFACTS),
                            mdbm_dir=str(RAVEL_MDBMS),
                        )
                        ravel.run_eval(
                            cfg,
                            selected_saes,
                            args.device,
                            str(output_folders[eval_type]),
                            force_rerun=True,
                            artifacts_path="data/artifacts",
                        )
                        print("RAVEL EVAL COMPLETED !!!")

                    # Add other elif blocks here for other eval types if needed...

                except Exception as e:
                    print(f"  -> ERROR running '{eval_type}' for {sae_location}: {e}")

            del sae
            torch.cuda.empty_cache()
            # Comment break if you want to run ravel eval for all SAE checkpoints with varying sparsity levels
            break

        print(f"Finished evaluation for architecture: {arch_name}\n")

    print("All evaluations complete!")
    print(f"Results saved in '{RAVEL_RESULTS}'.")


if __name__ == "__main__":
    main()

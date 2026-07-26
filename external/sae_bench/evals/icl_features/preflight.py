from __future__ import annotations

import argparse
import importlib
import json
from importlib import metadata
from pathlib import Path
from typing import Any

import sae_bench.custom_saes.base_sae as base_sae
import sae_bench.sae_bench_utils.general_utils as general_utils
import torch
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    load_dictionary_learning_sae,
)
from sae_bench.evals.icl_features.statistics import (
    one_sample_mean_degradation_test,
)
from sae_bench.sae_bench_utils.activation_collection import LLM_NAME_TO_DTYPE
from transformer_lens import HookedTransformer


def _parse_sae_spec(value: str) -> tuple[str, str, str]:
    parts = value.split("::")
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "Expected LABEL::LOCATION::EXPECTED_TRAINER_CLASS"
        )
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the GPU, dependencies, Gemma model, and custom SAEs."
    )
    parser.add_argument("--model-name", default="gemma-2-2b")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--sae-spec",
        action="append",
        type=_parse_sae_spec,
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--download-saes-dir",
        type=Path,
        default=Path("data/downloaded_saes"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _dependency_versions() -> dict[str, str]:
    modules = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "transformers": "transformers",
        "transformer_lens": "transformer-lens",
        "huggingface_hub": "huggingface-hub",
        "yaml": "PyYAML",
        "nltk": "nltk",
        "umap": "umap-learn",
        "spherecluster": "spherecluster",
    }
    versions = {}
    for module_name, display_name in modules.items():
        if module_name == "spherecluster":
            # Upstream spherecluster imports a removed private sklearn module
            # from its package initializer. FEGA uses its compatible vendored
            # implementation, so verify that implementation separately and
            # read the installed distribution version without importing it.
            importlib.import_module("fega.core.vmf.utils._spherecluster._vmfm")
            versions[display_name] = metadata.version(display_name)
            continue
        module = importlib.import_module(module_name)
        versions[display_name] = str(getattr(module, "__version__", "unknown"))
    from nltk.corpus import brown

    if not brown.words()[:1]:
        raise RuntimeError("The NLTK Brown corpus is installed but empty")
    return versions


def _load_and_check_sae(
    *,
    repo_id: str,
    label: str,
    location: str,
    expected_trainer_class: str,
    model_name: str,
    device: str,
    dtype: torch.dtype,
    download_saes_dir: Path,
) -> dict[str, Any]:
    repo_download_dir = download_saes_dir / repo_id.replace("/", "_")
    config_path = base_sae.resolve_repo_file(
        repo_id=repo_id,
        filename=f"{location}/config.json",
        local_dir=str(repo_download_dir),
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    trainer_class = str(config["trainer"]["trainer_class"])
    if trainer_class != expected_trainer_class:
        raise ValueError(
            f"{label}: expected {expected_trainer_class}, found {trainer_class}"
        )
    sae = load_dictionary_learning_sae(
        repo_id=repo_id,
        location=location,
        model_name=model_name,
        device=device,
        dtype=dtype,
        download_location=str(download_saes_dir),
    )
    with torch.no_grad():
        source = torch.zeros((1, int(sae.W_dec.shape[1])), device=device, dtype=dtype)
        encoded = sae.encode(source)
        decoded = sae.decode(encoded)
    result = {
        "label": label,
        "location": location,
        "trainer_class": trainer_class,
        "hook_name": str(sae.cfg.hook_name),
        "hook_layer": int(sae.cfg.hook_layer),
        "d_sae": int(sae.W_dec.shape[0]),
        "d_in": int(sae.W_dec.shape[1]),
        "encode_shape": list(encoded.shape),
        "decode_shape": list(decoded.shape),
    }
    del sae, source, encoded, decoded
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the GPU smoke test cannot run")
    dtype_name = LLM_NAME_TO_DTYPE.get(args.model_name, "float32")
    dtype = general_utils.str_to_dtype(dtype_name)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "device": args.device,
        "cuda_device_name": torch.cuda.get_device_name(args.device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "dependencies": _dependency_versions(),
        "model_name": args.model_name,
        "sae_repo_id": args.repo_id,
    }
    significance_check = one_sample_mean_degradation_test(
        [0.90, 0.91, 0.92, 0.93],
        reference_accuracy=1.0,
    )
    if significance_check is None or not (
        0.0 <= significance_check["p_value_one_sided"] < 0.01
    ):
        raise RuntimeError(
            f"SciPy aggregate significance self-check failed: {significance_check}"
        )
    payload["scipy_significance_check"] = significance_check

    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        dtype=dtype,
    )
    model.eval()
    tokens = model.to_tokens("The answer is", prepend_bos=True).to(args.device)
    with torch.no_grad():
        logits = model(tokens, return_type="logits")
    payload["model_check"] = {
        "tokens_shape": list(tokens.shape),
        "logits_shape": list(logits.shape),
        "finite_logits": bool(torch.isfinite(logits).all().item()),
    }
    del model, tokens, logits
    torch.cuda.empty_cache()

    payload["saes"] = [
        _load_and_check_sae(
            repo_id=args.repo_id,
            label=label,
            location=location,
            expected_trainer_class=expected,
            model_name=args.model_name,
            device=args.device,
            dtype=dtype,
            download_saes_dir=args.download_saes_dir,
        )
        for label, location, expected in args.sae_spec
    ]
    payload["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Preflight passed; wrote {args.output}")


if __name__ == "__main__":
    main()

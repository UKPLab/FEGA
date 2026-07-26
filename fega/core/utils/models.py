from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_bench.evals.ravel.main import LLM_NAME_MAP
from sae_bench.sae_bench_utils import general_utils


def _infer_location_from_release(
    release: str, repo_id: str, download_location: str | Path | None = None
) -> str | None:
    """Infer dictionary-learning SAE location from a release identifier."""
    import sae_bench.custom_saes.run_all_evals_dictionary_learning_saes as dict_saes

    repo_base = repo_id.split("/")[-1]
    tail = release[len(repo_base) + 1 :] if release.startswith(f"{repo_base}_") else release
    candidates = dict_saes.get_all_hf_repo_autoencoders(
        repo_id,
        download_location=str(download_location or "data/downloaded_saes"),
    )
    best = None
    for loc in candidates:
        loc_key = loc.replace("/", "_")
        if tail.endswith(loc_key):
            best = loc
            break
        if loc_key in tail:
            best = loc
    return best


def _resolve_sae_source(
    sae_source: str | None, local_checkpoint_path: str | Path | None
) -> str:
    source = (sae_source or "auto").strip().lower()
    if source not in {"auto", "sae_lens", "local_checkpoint"}:
        raise ValueError(
            f"Unsupported sae_source={sae_source!r}. Expected one of auto, sae_lens, local_checkpoint."
        )
    if source == "auto":
        return "local_checkpoint" if local_checkpoint_path is not None else "sae_lens"
    return source


def _hf_model_name(model_name: str) -> str:
    return LLM_NAME_MAP.get(model_name, model_name)


def _normalize_model_name_for_compare(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    return _hf_model_name(model_name)


def _validate_expected_sae_cfg(
    sae_cfg_dict: Dict[str, Any] | None,
    sae,
) -> None:
    if not sae_cfg_dict:
        return

    checks = {
        "hook_layer": (sae_cfg_dict.get("hook_layer"), int(getattr(sae.cfg, "hook_layer"))),
        "hook_name": (sae_cfg_dict.get("hook_name"), str(getattr(sae.cfg, "hook_name"))),
        "d_in": (sae_cfg_dict.get("d_in"), int(getattr(sae.cfg, "d_in"))),
        "d_sae": (sae_cfg_dict.get("d_sae"), int(getattr(sae.cfg, "d_sae"))),
    }
    for key, (expected, found) in checks.items():
        if expected is None:
            continue
        if key in {"hook_layer", "d_in", "d_sae"}:
            expected_value = int(expected)
        else:
            expected_value = str(expected)
        if expected_value != found:
            raise ValueError(
                f"SAE mismatch for {key}: expected {expected!r} from reference metadata, found {found!r}."
            )

    expected_model = _normalize_model_name_for_compare(
        str(sae_cfg_dict.get("model_name")) if sae_cfg_dict.get("model_name") else None
    )
    found_model = _normalize_model_name_for_compare(str(getattr(sae.cfg, "model_name")))
    if expected_model is not None and expected_model != found_model:
        raise ValueError(
            "SAE model mismatch against reference metadata: "
            f"expected {expected_model!r}, found {found_model!r}."
        )


def _validate_eval_and_model_compat(eval_config, model, sae) -> None:
    expected_model = _normalize_model_name_for_compare(str(eval_config.model_name))
    sae_model = _normalize_model_name_for_compare(str(getattr(sae.cfg, "model_name")))
    if expected_model != sae_model:
        raise ValueError(
            "eval_config.model_name mismatch with SAE config: "
            f"eval={expected_model!r}, sae={sae_model!r}."
        )
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(
            "Loaded model config does not expose `hidden_size` for SAE compatibility check."
        )
    sae_d_in = int(getattr(sae.cfg, "d_in"))
    if int(hidden_size) != sae_d_in:
        raise ValueError(
            f"Model hidden_size ({int(hidden_size)}) does not match SAE d_in ({sae_d_in})."
        )


def load_model_and_sae(
    eval_config,
    device: str,
    cache_dir: str | None = None,
    sae_repo_id: str | None = None,
    sae_release_id: str | None = None,
    sae_id_override: str | None = None,
    download_location: str | Path | None = None,
    sae_cfg_dict: Dict[str, Any] | None = None,
    sae_source: str | None = None,
    local_checkpoint_path: str | Path | None = None,
    local_resolved_config_path: str | Path | None = None,
    model_revision: str | None = None,
):
    """Load a compatible model, tokenizer, and SAE on the requested device.

    ``model_revision`` optionally pins both Hugging Face model-owned resources
    while leaving all existing callers on their prior unpinned behavior.
    """
    # Resolve the configured SAE source before loading any remote resources.
    source = _resolve_sae_source(sae_source, local_checkpoint_path)
    if source == "local_checkpoint":
        raise ValueError(
            "sae_source='local_checkpoint' is not supported in this FEGA pass. "
            "Use a SAE Lens release, or use sae_lens_id='custom_sae' with sae_repo_id "
            "so FEGA can load the dictionary-learning SAE through SAE Bench."
        )

    model_name = _hf_model_name(eval_config.model_name)
    llm_dtype = general_utils.str_to_dtype(eval_config.llm_dtype)
    model_kwargs = {"attn_implementation": "eager"} if "gemma" in model_name else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=llm_dtype,
        cache_dir=str(cache_dir) if cache_dir else None,
        revision=model_revision,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir) if cache_dir else None,
        revision=model_revision,
    )

    release = sae_release_id or getattr(eval_config, "sae_lens_release_id", None)
    sae_id_arg = sae_id_override or getattr(eval_config, "sae_lens_id", None)
    if release is None or sae_id_arg is None:
        raise ValueError(
            "SAE identifiers not provided; expected sae_lens_release_id and sae_lens_id."
        )

    if sae_id_arg == "custom_sae":
        if not sae_repo_id or sae_cfg_dict is None:
            raise ValueError(
                "custom_sae references require sae_repo_id and SAE config metadata "
                "so FEGA can use SAE Bench's dictionary-learning loader."
            )
        import sae_bench.custom_saes.run_all_evals_dictionary_learning_saes as dict_saes

        inferred_loc = _infer_location_from_release(
            release, sae_repo_id, download_location=download_location
        )
        if not inferred_loc:
            raise ValueError(
                f"Could not infer SAE location for release '{release}' in repo '{sae_repo_id}'"
            )
        hook_layer = sae_cfg_dict.get("hook_layer")
        sae = dict_saes.load_dictionary_learning_sae(
            repo_id=sae_repo_id,
            location=inferred_loc,
            model_name=eval_config.model_name,
            device=device,
            dtype=llm_dtype,
            layer=hook_layer,
            download_location=str(download_location or "data/downloaded_saes"),
        )
    else:
        _, sae, _ = general_utils.load_and_format_sae(release, sae_id_arg, device)

    _validate_expected_sae_cfg(sae_cfg_dict, sae)
    _validate_eval_and_model_compat(eval_config, model, sae)
    sae = sae.to(device=device, dtype=llm_dtype)
    return model, tokenizer, sae


def load_mdbm_mask(weight_path: Path) -> torch.Tensor:
    """Load binary mask from an MDBM checkpoint without instantiating the full model."""
    if not weight_path.exists():
        raise FileNotFoundError(f"MDBM weights not found at {weight_path}")
    try:
        blob = torch.load(weight_path, map_location="cpu", weights_only=True)
    except Exception:
        # Trusted local checkpoint: allowlist MDBM/MDAS classes and retry with weights_only=False
        from torch.serialization import add_safe_globals

        from sae_bench.evals.ravel import mdbm as mdbm_mod

        add_safe_globals([mdbm_mod.MDBM, mdbm_mod.MDAS])
        blob = torch.load(weight_path, map_location="cpu", weights_only=False)
    if hasattr(blob, "binary_mask"):
        return getattr(blob, "binary_mask").float()
    if isinstance(blob, dict):
        if "binary_mask" in blob:
            return blob["binary_mask"].float()
        state = blob.get("state_dict") or blob
        for key in ["binary_mask", "binary_mask_a", "binary_mask_b"]:
            if key in state:
                return state[key].float()
        # fallback: scan for any tensor key containing binary_mask
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and "binary_mask" in k:
                return v.float()
    raise ValueError(f"Could not find binary_mask in MDBM checkpoint at {weight_path}")

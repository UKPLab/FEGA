from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import sae_bench.custom_saes.run_all_evals_dictionary_learning_saes as dict_saes


def _load_model_utils():
    models_path = Path(__file__).resolve().parents[3] / "fega/core/utils/models.py"
    spec = importlib.util.spec_from_file_location("fega_model_utils_test", models_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


model_utils = _load_model_utils()


class DummyModel:
    def __init__(self, hidden_size: int = 768):
        self.config = SimpleNamespace(hidden_size=hidden_size)


class DummySAE:
    def __init__(
        self,
        *,
        model_name: str = "gpt2",
        hook_layer: int = 8,
        hook_name: str = "blocks.8.hook_resid_pre",
        d_in: int = 768,
        d_sae: int = 12288,
    ):
        self.cfg = SimpleNamespace(
            model_name=model_name,
            hook_layer=hook_layer,
            hook_name=hook_name,
            d_in=d_in,
            d_sae=d_sae,
        )

    def to(self, device=None, dtype=None):
        del device, dtype
        return self


@pytest.fixture
def eval_config():
    return SimpleNamespace(model_name="gpt2", llm_dtype="float32")


def _patch_hf_loaders(monkeypatch, hidden_size: int = 768) -> None:
    monkeypatch.setattr(
        model_utils.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: DummyModel(hidden_size=hidden_size)),
    )
    monkeypatch.setattr(
        model_utils.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: object()),
    )


def _sae_cfg_dict() -> dict[str, object]:
    return {
        "model_name": "gpt2",
        "hook_layer": 8,
        "hook_name": "blocks.8.hook_resid_pre",
        "d_in": 768,
        "d_sae": 12288,
    }


def _patch_dictionary_loader(monkeypatch, sae: DummySAE) -> dict[str, object]:
    calls = {}
    monkeypatch.setattr(
        model_utils,
        "_infer_location_from_release",
        lambda release, repo_id, download_location=None: "resid_post_layer_8/trainer_0",
    )

    def fake_loader(**kwargs):
        calls.update(kwargs)
        return sae

    monkeypatch.setattr(
        dict_saes,
        "load_dictionary_learning_sae",
        fake_loader,
    )
    return calls


def test_load_model_and_sae_validates_reference_sae_cfg(
    monkeypatch, eval_config
) -> None:
    _patch_hf_loaders(monkeypatch)
    _patch_dictionary_loader(monkeypatch, DummySAE(d_in=768))

    with pytest.raises(ValueError, match="SAE mismatch for d_in"):
        model_utils.load_model_and_sae(
            eval_config=eval_config,
            device="cpu",
            sae_cfg_dict={
                **_sae_cfg_dict(),
                "d_in": 1024,
            },
            sae_release_id="repo_release_resid_post_layer_8_trainer_0",
            sae_id_override="custom_sae",
            sae_repo_id="owner/repo",
        )


def test_load_model_and_sae_validates_hidden_size(monkeypatch, eval_config) -> None:
    _patch_hf_loaders(monkeypatch, hidden_size=1024)
    _patch_dictionary_loader(monkeypatch, DummySAE(d_in=768))

    with pytest.raises(ValueError, match="hidden_size"):
        model_utils.load_model_and_sae(
            eval_config=eval_config,
            device="cpu",
            sae_cfg_dict=_sae_cfg_dict(),
            sae_release_id="repo_release_resid_post_layer_8_trainer_0",
            sae_id_override="custom_sae",
            sae_repo_id="owner/repo",
        )


def test_load_model_and_sae_pins_model_and_tokenizer_revision(
    monkeypatch, eval_config
) -> None:
    """Forward the frozen revision to both Hugging Face checkpoint loaders."""
    # Capture both loader calls while keeping SAE loading on the existing test path.
    calls = {}

    def load_model(*args, **kwargs):
        """Record model-loader keywords and return a compatible dummy model."""
        # Preserve the call for assertions after the loader completes.
        calls["model"] = kwargs
        return DummyModel()

    def load_tokenizer(*args, **kwargs):
        """Record tokenizer-loader keywords and return a minimal dummy object."""
        # Preserve the call for assertions after the loader completes.
        calls["tokenizer"] = kwargs
        return object()

    monkeypatch.setattr(
        model_utils.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(load_model),
    )
    monkeypatch.setattr(
        model_utils.AutoTokenizer,
        "from_pretrained",
        staticmethod(load_tokenizer),
    )
    _patch_dictionary_loader(monkeypatch, DummySAE())

    model_utils.load_model_and_sae(
        eval_config=eval_config,
        device="cpu",
        sae_cfg_dict=_sae_cfg_dict(),
        sae_release_id="repo_release_resid_post_layer_8_trainer_0",
        sae_id_override="custom_sae",
        sae_repo_id="owner/repo",
        model_revision="frozen-revision",
    )

    assert calls["model"]["revision"] == "frozen-revision"
    assert calls["tokenizer"]["revision"] == "frozen-revision"

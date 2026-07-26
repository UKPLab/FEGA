import json
from pathlib import Path
from types import SimpleNamespace

import sae_bench.custom_saes.base_sae as base_sae
import sae_bench.evals.icl_features.preflight as preflight
import torch
from nltk.corpus import brown


class _FakeSae:
    W_dec = torch.zeros((6, 4))
    cfg = SimpleNamespace(hook_name="blocks.2.hook_resid_post", hook_layer=2)

    def encode(self, source: torch.Tensor) -> torch.Tensor:
        return torch.zeros((source.shape[0], 6))

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        return torch.zeros((encoded.shape[0], 4))


def test_preflight_uses_local_first_sae_resolver(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"trainer": {"trainer_class": "StandardTrainer"}}),
        encoding="utf-8",
    )
    download_root = tmp_path / "data/downloaded_saes"
    repo_cache = download_root / "owner_repo"
    resolver_kwargs = {}

    def fake_resolver(**kwargs):
        resolver_kwargs.update(kwargs)
        return str(config_path)

    def fail_hub_lookup(**_kwargs):
        raise AssertionError("preflight should use the local-first SAE resolver")

    def fake_loader(**kwargs):
        assert kwargs["download_location"] == str(download_root)
        return _FakeSae()

    monkeypatch.setattr(base_sae, "resolve_repo_file", fake_resolver)
    monkeypatch.setattr(preflight, "hf_hub_download", fail_hub_lookup, raising=False)
    monkeypatch.setattr(preflight, "load_dictionary_learning_sae", fake_loader)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    result = preflight._load_and_check_sae(
        repo_id="owner/repo",
        label="relu",
        location="location/trainer_0",
        expected_trainer_class="StandardTrainer",
        model_name="gemma-2-2b",
        device="cpu",
        dtype=torch.float32,
        download_saes_dir=download_root,
    )

    assert resolver_kwargs == {
        "repo_id": "owner/repo",
        "filename": "location/trainer_0/config.json",
        "local_dir": str(repo_cache),
    }
    assert result["encode_shape"] == [1, 6]
    assert result["decode_shape"] == [1, 4]


def test_preflight_checks_fega_vendored_spherecluster(monkeypatch) -> None:
    imported = []
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda name: imported.append(name) or SimpleNamespace(__version__="test"),
    )
    monkeypatch.setattr(preflight.metadata, "version", lambda _name: "test")
    monkeypatch.setitem(vars(brown), "words", lambda: ["token"])

    versions = preflight._dependency_versions()

    assert "fega.core.vmf.utils._spherecluster._vmfm" in imported
    assert versions["spherecluster"] == "test"

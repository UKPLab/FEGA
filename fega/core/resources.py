from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fega.config_schema import FEGAPipelineConfig
from fega.core.config import FEGAConfig
from fega.core.utils import load_mdbm_mask, load_model_and_sae
from fega.core.utils.ravel import ReplayContext, mdbm_weight_path
from fega.paths import run_root


def resolve_mdbm_path(
    config: FEGAPipelineConfig,
    entity_class: str,
    attribute: str,
    *,
    weight_path_override: Path | None = None,
    ctx: ReplayContext | None = None,
) -> Path:
    """Resolve the effective MDBM checkpoint path for an entity/attribute pair."""
    if weight_path_override:
        return weight_path_override
    if config.mdbm_weight_path:
        return config.mdbm_weight_path
    replay_ctx = (
        ctx if ctx is not None else ReplayContext.from_file(config.reference_json)
    )
    return mdbm_weight_path(
        replay_ctx,
        entity_class,
        attribute,
        replay_ctx.reference_path,
        mdbm_root_override=config.mdbm_root,
    )


@dataclass
class ModelResources:
    """Lazy, reusable handles for model/tokenizer/SAE and MDBM masks."""

    config: FEGAPipelineConfig
    model_revision: str | None = None
    _model: Any | None = None
    _tokenizer: Any | None = None
    _sae: Any | None = None
    _mdbm_masks: dict[tuple[str, str, str], Any] = field(default_factory=dict)
    _ctx: ReplayContext | None = None
    _eval_config: Any | None = None
    _json_cache: dict[str, Any] = field(default_factory=dict)

    def get_model_and_sae(self):
        """Load or return cached (model, tokenizer, sae) on the configured device."""
        if self._model is None or self._tokenizer is None or self._sae is None:
            if self.config.source_kind == "induction":
                from fega.core.data_prep.induction import (
                    resolve_induction_model_sae_spec,
                )

                spec = resolve_induction_model_sae_spec(self.config)
                eval_cfg = spec.eval_config
                sae_release_id = spec.sae_release_id
                sae_id_override = spec.sae_id_override
                sae_repo_id = spec.sae_repo_id
                sae_cfg_dict = spec.sae_cfg_dict
            else:
                ctx = self._ensure_replay_context()
                eval_cfg = self._ensure_eval_config()
                sae_release_id = ctx.sae_lens_release_id
                sae_id_override = ctx.sae_lens_id
                sae_repo_id = self.config.sae_repo_id
                sae_cfg_dict = ctx.sae_cfg_dict
            model, tokenizer, sae = load_model_and_sae(
                eval_cfg,
                self.config.device,
                cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
                sae_release_id=sae_release_id,
                sae_id_override=sae_id_override,
                download_location=self.config.download_saes_dir,
                sae_repo_id=sae_repo_id,
                sae_cfg_dict=sae_cfg_dict,
                sae_source=self.config.sae_source,
                local_checkpoint_path=self.config.local_sae_checkpoint_path,
                local_resolved_config_path=self.config.local_sae_resolved_config_path,
                model_revision=self.model_revision,
            )
            model.eval()
            sae.eval()
            self._model, self._tokenizer, self._sae = model, tokenizer, sae
        return self._model, self._tokenizer, self._sae

    def get_tokenizer(self):
        """Return a cached tokenizer, loading model/SAE bundle if needed."""
        if self._tokenizer is None:
            _, tokenizer, _ = self.get_model_and_sae()
            return tokenizer
        return self._tokenizer

    def get_mdbm_mask(
        self,
        entity_class: str,
        attribute: str,
        weight_path: Path | None = None,
    ):
        """Load or cache the MDBM mask tensor for an entity/attribute."""
        resolved_path = self._resolve_mdbm_path(entity_class, attribute, weight_path)
        key = (entity_class, attribute, str(resolved_path))
        if key not in self._mdbm_masks:
            self._mdbm_masks[key] = load_mdbm_mask(resolved_path)
        return self._mdbm_masks[key]

    def _resolve_mdbm_path(
        self,
        entity_class: str,
        attribute: str,
        weight_path_override: Path | None = None,
    ):
        return resolve_mdbm_path(
            self.config,
            entity_class,
            attribute,
            weight_path_override=weight_path_override,
            ctx=self._ensure_replay_context(),
        )

    def get_cached_json(self, path: Path):
        """Return cached JSON content if available."""
        return self._json_cache.get(str(path.resolve()))

    def cache_json(self, path: Path, payload: Any) -> None:
        """Cache JSON-able payload keyed by absolute path."""
        self._json_cache[str(path.resolve())] = payload

    def _ensure_replay_context(self) -> ReplayContext:
        if self._ctx is None:
            self._ctx = ReplayContext.from_file(self.config.reference_json)
        return self._ctx

    def _ensure_eval_config(self):
        if self._eval_config is None:
            cfg = FEGAConfig.from_reference(
                self.config.reference_json,
                device=self.config.device,
                output_dir=run_root(self.config),
                entity_attribute_selection=self.config.entity_attribute_selection,
                random_seed=self.config.seed.global_,
                llm_batch_size_override=self.config.llm_batch_size_override,
            )
            self._eval_config = cfg.eval_config
        return self._eval_config

import json
from pathlib import Path
from typing import Dict, List, Mapping

import torch

from .misc import resolve_path


class ChunkProcessor:
    """Chunked activation/meta writer with shared streaming helper."""

    def __init__(
        self,
        out_dir: Path,
        chunk_size: int | None,
        tensor_chunk_tpl: str,
        meta_chunk_tpl: str,
        single_file: bool,
    ):
        """Initialize a chunk writer; buffers live in memory until flush."""
        self.out_dir = out_dir
        self.chunk_size = chunk_size
        self.tensor_chunk_tpl = tensor_chunk_tpl
        self.meta_chunk_tpl = meta_chunk_tpl
        self.single_file = single_file
        self.buffers: Dict[str, list] = {"x": [], "z": [], "meta": []}
        self.manifest_entries: List[Dict] = []
        self.chunk_idx = 0

    def add(
        self,
        rep: torch.Tensor,
        z: torch.Tensor,
        logit: torch.Tensor | None,
        meta_rec: Dict,
        *,
        readouts: Mapping[str, torch.Tensor] | None = None,
    ):
        """Buffer one example and flush if the chunk is full."""
        self.buffers["x"].append(rep.cpu())
        self.buffers["z"].append(z.cpu())
        tensor_readouts = dict(readouts or {})
        if logit is not None and "logits" not in tensor_readouts:
            tensor_readouts["logits"] = logit
        for name, tensor in tensor_readouts.items():
            self.buffers.setdefault(name, []).append(tensor.cpu())
        self.buffers["meta"].append(meta_rec)
        if self.chunk_size and len(self.buffers["x"]) >= self.chunk_size:
            self.flush()

    def flush(self):
        """Write buffered tensors/meta to disk and reset buffers."""
        if not self.buffers["x"]:
            return
        chunk_count = len(self.buffers["x"])
        indices = [m["index"] for m in self.buffers["meta"]]
        tensor_fname = (
            "activations_tensors.pt"
            if self.single_file
            else self.tensor_chunk_tpl.format(self.chunk_idx)
        )
        meta_fname = (
            "activations_meta.jsonl"
            if self.single_file
            else self.meta_chunk_tpl.format(self.chunk_idx)
        )
        tensor_path = self.out_dir / tensor_fname
        meta_path_local = self.out_dir / meta_fname

        payload: Dict[str, torch.Tensor | list] = {"index": indices}
        for key, values in self.buffers.items():
            if key == "meta":
                continue
            payload[key] = torch.stack(values)
        torch.save(payload, tensor_path)

        with open(meta_path_local, "w") as mf:
            for rec in self.buffers["meta"]:
                mf.write(json.dumps(rec) + "\n")

        self.manifest_entries.append(
            {
                "chunk": self.chunk_idx,
                "count": chunk_count,
                "start_index": indices[0],
                "end_index": indices[-1],
                "tensors": str(tensor_path),
                "meta": str(meta_path_local),
            }
        )
        self.chunk_idx += 1
        for buf in self.buffers.values():
            buf.clear()

    @staticmethod
    def stream(manifest_path: Path, base_dir: Path):
        """Yield (tensors_path, meta_path) pairs from manifest (chunked or single-file)."""
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        manifest_base = manifest_path.parent
        chunks = manifest.get("chunks", [])
        if not chunks and (
            int(manifest.get("chunk_count") or 0) == 0
            or int(manifest.get("total_records") or 0) == 0
        ):
            return
        if not chunks:
            tensors_candidate = manifest.get("tensors") or "activations_tensors.pt"
            meta_candidates = []
            if "meta" in manifest:
                meta_candidates.append(manifest["meta"])
            meta_candidates.extend(["activations_meta.jsonl", "activations_meta.json"])
            tensors_path = resolve_path(
                base_dir, tensors_candidate, secondary_base=manifest_base
            )
            meta_path = None
            for cand in meta_candidates:
                try:
                    meta_path = resolve_path(
                        base_dir, cand, secondary_base=manifest_base
                    )
                    break
                except FileNotFoundError:
                    continue
            if meta_path is None:
                raise FileNotFoundError(
                    f"Could not resolve meta file. Tried {meta_candidates} under base {base_dir}"
                )
            yield tensors_path, meta_path
            return
        for entry in chunks:
            tensors_path = resolve_path(
                base_dir, entry["tensors"], secondary_base=manifest_base
            )
            meta_path = resolve_path(
                base_dir, entry["meta"], secondary_base=manifest_base
            )
            yield tensors_path, meta_path

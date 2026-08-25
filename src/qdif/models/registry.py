"""Local checkpoint discovery.

Scans the places a machine typically keeps models -- the Hugging Face hub cache, any
mounted external volume, LM Studio's store -- so an experiment can run offline and so
`qdif inspect` can report what is already on disk before anything is downloaded.

Read-only: nothing here writes to or deletes a model directory.

Set `QDIF_MODEL_ROOTS` (colon-separated) to add your own locations, or `HF_HOME` to
point at a non-default cache.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _default_roots() -> list[Path]:
    """Where to look for local checkpoints, in priority order.

    Portable by construction: the HF cache (honouring `HF_HOME`), LM Studio's store,
    any extra roots the user names in `QDIF_MODEL_ROOTS` (colon-separated), and every
    mounted volume's common cache layouts. Nothing machine-specific is hard-coded --
    this repository was developed with an external drive called SHUTTLE, and that name
    should not leak into anyone else's setup.
    """
    roots: list[Path] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots += [Path(hf_home) / "hub", Path(hf_home) / "huggingface" / "hub"]
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for extra in filter(None, os.environ.get("QDIF_MODEL_ROOTS", "").split(":")):
        roots.append(Path(extra))
    volumes = Path("/Volumes")
    if volumes.is_dir():
        try:
            for vol in sorted(volumes.iterdir()):
                if not vol.is_dir():
                    continue
                for sub in ("huggingface/hub", "hub", "lmstudio-models", "mtplx-models",
                            "omlx", "unsloth"):
                    roots.append(vol / sub)
        except OSError:
            pass
    roots.append(Path.home() / ".lmstudio" / "models")
    return roots


#: Roots searched for checkpoints, in priority order. Missing paths are skipped.
SEARCH_ROOTS: list[Path] = _default_roots()

_FORMAT_MARKERS = (
    ("gguf", ("*.gguf",)),
    ("safetensors", ("*.safetensors",)),
)


@dataclass
class LocalModel:
    repo_id: str
    path: Path
    fmt: str  # "safetensors" | "gguf" | "unknown"
    size_bytes: int
    model_type: str | None = None
    num_hidden_layers: int | None = None
    hidden_size: int | None = None
    quantization: str | None = None
    has_vision: bool = False
    has_mtp: bool = False

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9

    @property
    def is_trainable_hf(self) -> bool:
        """True if this is a full-precision HF checkpoint we can load into torch.

        GGUF and pre-quantized MLX exports are usable for baseline inference but not
        as a training backbone in this harness.
        """
        return self.fmt == "safetensors" and self.quantization is None


def _dir_size(path: Path, cap_files: int = 4000) -> int:
    total = 0
    for i, p in enumerate(path.rglob("*")):
        if i > cap_files:
            break
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
            elif p.is_symlink():
                total += p.resolve().stat().st_size
        except OSError:
            continue
    return total


def _read_config(path: Path) -> dict:
    cfg = path / "config.json"
    if not cfg.exists():
        return {}
    try:
        return json.loads(cfg.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _describe(repo_id: str, path: Path) -> LocalModel | None:
    fmt = "unknown"
    for name, patterns in _FORMAT_MARKERS:
        if any(next(path.glob(pat), None) is not None for pat in patterns):
            fmt = name
            break
    if fmt == "unknown" and not (path / "config.json").exists():
        return None

    cfg = _read_config(path)
    text_cfg = cfg.get("text_config", cfg)
    quant = None
    if "quantization" in cfg:
        q = cfg["quantization"]
        quant = f"{q.get('bits', '?')}bit" if isinstance(q, dict) else str(q)
    elif "quantization_config" in cfg:
        q = cfg["quantization_config"]
        quant = str(q.get("quant_method", q)) if isinstance(q, dict) else str(q)

    return LocalModel(
        repo_id=repo_id,
        path=path,
        fmt=fmt,
        size_bytes=_dir_size(path),
        model_type=cfg.get("model_type"),
        num_hidden_layers=text_cfg.get("num_hidden_layers"),
        hidden_size=text_cfg.get("hidden_size"),
        quantization=quant,
        has_vision="vision_config" in cfg,
        has_mtp=bool(text_cfg.get("mtp_num_hidden_layers")),
    )


def _iter_candidates(root: Path):
    """Yield (repo_id, snapshot_dir) pairs for both hub-cache and flat layouts."""
    if not root.exists():
        return
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.name.startswith("models--"):
            repo_id = entry.name[len("models--") :].replace("--", "/")
            snaps = entry / "snapshots"
            if snaps.is_dir():
                for snap in sorted(snaps.iterdir()):
                    if snap.is_dir():
                        yield repo_id, snap
            continue
        # Flat layout: <root>/<org>/<model> or <root>/<model>
        if (entry / "config.json").exists() or next(entry.glob("*.gguf"), None):
            yield entry.name, entry
            continue
        try:
            for sub in sorted(entry.iterdir()):
                if sub.is_dir() and (
                    (sub / "config.json").exists() or next(sub.glob("*.gguf"), None)
                ):
                    yield f"{entry.name}/{sub.name}", sub
        except OSError:
            continue


def scan_local_models(name_filter: str = "qwen", roots: list[Path] | None = None) -> list[LocalModel]:
    """Find local checkpoints whose repo id contains `name_filter` (case-insensitive)."""
    found: list[LocalModel] = []
    seen: set[Path] = set()
    for root in roots or SEARCH_ROOTS:
        for repo_id, path in _iter_candidates(root):
            if name_filter and name_filter.lower() not in repo_id.lower():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            desc = _describe(repo_id, path)
            if desc is not None:
                found.append(desc)
    return sorted(found, key=lambda m: (m.repo_id, str(m.path)))


def resolve_model_path(
    repo_id: str,
    local_path: str | None = None,
    local_files_only: bool = True,
    require_trainable: bool = True,
) -> Path:
    """Resolve a config's model reference to a concrete directory on disk.

    Order: explicit `local_path` -> exact repo-id match in a local root ->
    HF hub download (only if `local_files_only` is False).
    """
    if local_path:
        p = Path(os.path.expanduser(local_path))
        if not p.exists():
            raise FileNotFoundError(f"model.local_path does not exist: {p}")
        return p

    candidates = [m for m in scan_local_models(name_filter="") if m.repo_id == repo_id]
    if require_trainable:
        candidates = [m for m in candidates if m.is_trainable_hf] or candidates
    if candidates:
        return candidates[0].path

    if local_files_only:
        near = [m.repo_id for m in scan_local_models(name_filter="qwen3.5")]
        raise FileNotFoundError(
            f"{repo_id!r} was not found locally and model.local_files_only=True.\n"
            f"Local Qwen3.5-family checkpoints available: {sorted(set(near)) or '(none)'}\n"
            f"Either point model.local_path at one of these, or set "
            f"model.local_files_only=false to allow a download."
        )

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id))

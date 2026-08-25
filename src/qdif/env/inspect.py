"""Environment inspection: hardware, backends, local checkpoints, baseline servers."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

OPTIONAL_PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "accelerate",
    "datasets",
    "mlx",
    "mlx-lm",
    "peft",
    "unsloth",
    "bitsandbytes",
)


@dataclass
class HardwareInfo:
    platform: str
    machine: str
    processor: str
    python: str
    total_memory_gb: float
    memory_kind: str
    torch_device: str = "cpu"
    mps_available: bool = False
    cuda_available: bool = False
    cuda_device: str | None = None
    mps_recommended_working_set_gb: float | None = None


def _sysctl(key: str) -> str | None:
    if not shutil.which("sysctl"):
        return None
    try:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_info() -> HardwareInfo:
    is_mac = platform.system() == "Darwin"
    total_bytes = 0
    try:
        import psutil

        total_bytes = psutil.virtual_memory().total
    except ImportError:
        raw = _sysctl("hw.memsize")
        total_bytes = int(raw) if raw else 0

    info = HardwareInfo(
        platform=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        processor=_sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown",
        python=sys.version.split()[0],
        total_memory_gb=total_bytes / 1e9,
        memory_kind="unified (shared CPU/GPU)" if is_mac else "system RAM",
    )

    try:
        import torch

        info.mps_available = bool(torch.backends.mps.is_available())
        info.cuda_available = bool(torch.cuda.is_available())
        if info.cuda_available:
            info.cuda_device = torch.cuda.get_device_name(0)
            info.torch_device = "cuda"
            info.memory_kind = "VRAM"
        elif info.mps_available:
            info.torch_device = "mps"
            # Metal's practical ceiling, not the machine's installed RAM.
            info.mps_recommended_working_set_gb = round(total_bytes * 0.75 / 1e9, 1)
    except ImportError:
        pass
    return info


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in OPTIONAL_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def transformers_qwen_support() -> dict[str, bool]:
    """Which Qwen architectures the installed transformers can actually instantiate."""
    support: dict[str, bool] = {}
    for name in ("qwen3", "qwen3_next", "qwen3_5", "qwen3_5_moe", "qwen3_vl"):
        try:
            importlib.import_module(f"transformers.models.{name}")
            support[name] = True
        except ImportError:
            support[name] = False
    return support


@dataclass
class ServerStatus:
    name: str
    url: str
    up: bool
    models: list[str] = field(default_factory=list)


def baseline_servers() -> list[ServerStatus]:
    from ..inference.openai_client import lmstudio_client, omlx_client

    out = []
    for client in (lmstudio_client(), omlx_client()):
        up = client.is_up()
        out.append(
            ServerStatus(
                name=client.name,
                url=client.base_url,
                up=up,
                models=client.list_models()[:10] if up else [],
            )
        )
    return out


def volumes() -> list[str]:
    root = Path("/Volumes")
    if not root.exists():
        return []
    try:
        return sorted(p.name for p in root.iterdir() if p.is_dir())
    except OSError:
        return []


def full_report(model_filter: str = "qwen") -> dict:
    from ..models.registry import SEARCH_ROOTS, scan_local_models

    models = scan_local_models(name_filter=model_filter)
    return {
        "hardware": asdict(hardware_info()),
        "packages": package_versions(),
        "transformers_qwen_support": transformers_qwen_support(),
        "volumes": volumes(),
        "search_roots": [str(p) for p in SEARCH_ROOTS if p.exists()],
        "local_models": [
            {
                "repo_id": m.repo_id,
                "path": str(m.path),
                "format": m.fmt,
                "size_gb": round(m.size_gb, 2),
                "model_type": m.model_type,
                "layers": m.num_hidden_layers,
                "hidden": m.hidden_size,
                "quantization": m.quantization,
                "vision": m.has_vision,
                "mtp": m.has_mtp,
                "trainable_backbone": m.is_trainable_hf,
            }
            for m in models
        ],
        "baseline_servers": [asdict(s) for s in baseline_servers()],
    }

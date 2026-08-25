"""Adapter checkpointing.

Saves only what training actually changed -- LoRA factors and the auxiliary
diffusion modules -- alongside the config and enough provenance to reload. Base
weights are never written, so a checkpoint directory is a few MB and can be
reloaded onto any copy of the same base model.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..config import ExperimentConfig, save_config
from .lora import load_lora_state_dict, lora_state_dict

ADAPTER_FILE = "adapters.safetensors"
META_FILE = "adapter_meta.json"
CONFIG_FILE = "config.yaml"


def auxiliary_state_dict(model) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for prefix, module in model.auxiliary_modules().items():
        for name, tensor in module.state_dict().items():
            state[f"{prefix}.{name}"] = tensor.detach().cpu()
    return state


def save_checkpoint(
    model,
    cfg: ExperimentConfig,
    path: str | Path,
    step: int = 0,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    state = {f"lora.{k}": v for k, v in lora_state_dict(model.base).items()}
    state.update({f"aux.{k}": v for k, v in auxiliary_state_dict(model).items()})
    if not state:
        raise RuntimeError("nothing to save: no LoRA adapters and no auxiliary modules")

    # safetensors requires contiguous, non-shared storage.
    save_file({k: v.contiguous().clone() for k, v in state.items()}, str(path / ADAPTER_FILE))

    meta = {
        "step": step,
        "num_tensors": len(state),
        "model_id": cfg.model.id,
        "model_local_path": cfg.model.local_path,
        "canvas_length": cfg.diffusion.canvas_length,
        "lora_rank": cfg.lora.rank,
        "lora_alpha": cfg.lora.alpha,
        "lora_target_preset": cfg.lora.target_preset,
        "lora_layer_indices": cfg.lora.layer_indices,
        "self_conditioning": cfg.diffusion.self_conditioning,
        "timestep_conditioning": cfg.diffusion.timestep_conditioning,
        **(extra or {}),
    }
    (path / META_FILE).write_text(json.dumps(meta, indent=2))
    save_config(cfg, path / CONFIG_FILE)
    return path


def load_checkpoint(model, path: str | Path) -> dict:
    """Load adapters + auxiliary modules into an already-constructed DiffusionQwen."""
    path = Path(path)
    file = path / ADAPTER_FILE
    if not file.exists():
        raise FileNotFoundError(f"no adapter file at {file}")

    state = load_file(str(file))
    lora = {k[len("lora.") :]: v for k, v in state.items() if k.startswith("lora.")}
    aux = {k[len("aux.") :]: v for k, v in state.items() if k.startswith("aux.")}

    if lora:
        load_lora_state_dict(model.base, lora)

    modules = model.auxiliary_modules()
    for prefix, module in modules.items():
        sub = {
            k[len(prefix) + 1 :]: v for k, v in aux.items() if k.startswith(prefix + ".")
        }
        if sub:
            module.load_state_dict(
                {k: v.to(next(module.parameters()).dtype) for k, v in sub.items()}
            )

    meta_path = path / META_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["loaded_lora_tensors"] = len(lora)
    meta["loaded_aux_tensors"] = len(aux)
    return meta

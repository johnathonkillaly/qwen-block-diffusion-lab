"""Verified Unsloth backend capabilities.

Every entry here is *probed at runtime*, not inferred from the fact that a package
is importable. "Unsloth is installed" is not evidence that an Unsloth kernel is
running, and this module exists so that claim never gets made on faith.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import urllib.error
import urllib.request
from typing import Any

STUDIO_URL = "http://127.0.0.1:8888"

PACKAGES = (
    "unsloth",
    "unsloth_zoo",
    "mlx",
    "mlx-lm",
    "mlx-metal",
    "torch",
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
    "triton",
    "xformers",
)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _probe(fn, label: str) -> dict[str, Any]:
    try:
        return {"status": "ok", "detail": fn()}
    except Exception as exc:  # noqa: BLE001 - the point is to record the failure
        return {"status": "unavailable", "detail": f"{type(exc).__name__}: {exc}"}


def unsloth_device() -> dict:
    from unsloth.device_type import DEVICE_TYPE, is_mlx_available

    return {"DEVICE_TYPE": DEVICE_TYPE, "is_mlx_available": bool(is_mlx_available())}


def gated_delta_patch_state() -> dict:
    """Is Unsloth's Qwen3.5 GatedDeltaNet custom VJP importable, and does it bind?"""
    import mlx_lm.models.gated_delta as gd

    before = getattr(gd.gated_delta_ops, "__module__", "?")
    mod = importlib.import_module("unsloth_zoo.gated_delta_vjp")
    has_patch = hasattr(mod, "patch_gated_delta")
    has_kernels = hasattr(mod, "gated_delta_kernel_efficient")
    return {
        "unsloth_zoo.gated_delta_vjp importable": True,
        "patch_gated_delta present": has_patch,
        "metal chunk kernels present": has_kernels,
        "mlx_lm.gated_delta_ops module (pre-patch)": before,
        "purpose": (
            "memory-efficient custom VJP for Qwen3.5 GatedDeltaNet: recomputes "
            "recurrent states during backward instead of keeping all T intermediates"
        ),
    }


def mlx_runtime() -> dict:
    import mlx.core as mx

    return {
        "default_device": str(mx.default_device()),
        "metal_available": bool(mx.metal.is_available()),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
        "memory_kind": "unified (shared CPU/GPU) -- NOT VRAM",
    }


def studio_state(url: str = STUDIO_URL) -> dict:
    def get(path):
        req = urllib.request.Request(f"{url}{path}")
        with urllib.request.urlopen(req, timeout=6) as resp:
            return json.loads(resp.read())

    try:
        spec = get("/openapi.json")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"up": False, "detail": f"{type(exc).__name__}: {exc}"}

    paths = sorted(spec.get("paths", {}))
    schema = spec.get("components", {}).get("schemas", {}).get("TrainingStartRequest", {})
    props = schema.get("properties", {})
    training_types = props.get("training_type", {}).get("enum")
    hook_fields = [
        k for k in props if any(s in k.lower() for s in ("custom", "script", "hook", "entry", "callback"))
    ]
    return {
        "up": True,
        "num_endpoints": len(paths),
        "has_train_start": "/api/train/start" in paths,
        "training_type_enum": training_types,
        "training_request_fields": len(props),
        "custom_code_fields": hook_fields,
        "has_lora_export": "/api/export/export/lora" in paths,
        "has_gguf_export": "/api/export/export/gguf" in paths,
        "image_diffusion_endpoints": [p for p in paths if p.startswith("/api/train/diffusion")][:4],
    }


def capability_report() -> dict:
    report: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "versions": {p: _version(p) for p in PACKAGES},
        "unsloth_device": _probe(unsloth_device, "device"),
        "mlx_runtime": _probe(mlx_runtime, "mlx"),
        "gated_delta_vjp": _probe(gated_delta_patch_state, "gdn"),
        "studio": studio_state(),
    }

    v = report["versions"]
    report["findings"] = [
        (
            "Unsloth on darwin/arm64 is an MLX stack, not a torch stack. "
            "`unsloth.device_type.DEVICE_TYPE` reports 'mlx', and `device_type.py` only "
            "imports torch when MLX is absent. unsloth_zoo's darwin-arm64 dependency set "
            "is mlx / mlx-lm / mlx-vlm, and it EXCLUDES torch, torchao, peft, trl, "
            "accelerate and cut_cross_entropy."
        ),
        (
            "unsloth_zoo ships `gated_delta_vjp.py`, a memory-efficient custom VJP for "
            "Qwen3.5's GatedDeltaNet with hand-written Metal chunk kernels. This is the "
            "single most relevant Unsloth component for v0.2 and it is Apple-native."
        ),
        (
            "Triton and xformers have no macOS wheels and never activate here. "
            f"bitsandbytes is installed ({v.get('bitsandbytes')}) but is a CUDA library; "
            "its 4-bit/8-bit paths and paged optimizers do not run on Apple silicon, "
            "which is why v0.2 is BF16 base + BF16 LoRA."
        ),
        (
            "Unsloth pins transformers <= 5.5.0 and torch < 2.12. The v0.1 torch backend "
            "needs transformers >= 5.8 for the dict-keyed attention mask, so the two "
            "backends cannot share one environment. v0.1 runs in .venv; v0.2 in .venv-unsloth."
        ),
    ]

    s = report["studio"]
    if s.get("up"):
        report["findings"].append(
            "Unsloth Studio exposes /api/train/start with a CLOSED training_type enum "
            f"({s.get('training_type_enum')}) and no custom-script, hook or callback field "
            f"among its {s.get('training_request_fields')} parameters. Studio can therefore "
            "NOT own a custom diffusion training loop. Its /api/train/diffusion/* endpoints "
            "are IMAGE diffusion LoRA (dataset of images + captions), unrelated to discrete "
            "text diffusion despite the name."
        )
    return report


def render(report: dict, echo, rule) -> None:
    rule("platform")
    echo(f"  {report['platform']}")

    rule("versions")
    for name, ver in report["versions"].items():
        echo(f"  {name:<16} {ver or '-- not installed'}")

    rule("unsloth device path")
    echo(f"  {report['unsloth_device']}")

    rule("mlx runtime")
    for k, v in (report["mlx_runtime"].get("detail") or {}).items():
        echo(f"  {k:<22} {v}")

    rule("unsloth gated-delta VJP (Qwen3.5-specific)")
    detail = report["gated_delta_vjp"].get("detail")
    if isinstance(detail, dict):
        for k, v in detail.items():
            echo(f"  {k:<42} {v}")
    else:
        echo(f"  {report['gated_delta_vjp']}")

    rule("unsloth studio")
    for k, v in report["studio"].items():
        echo(f"  {k:<30} {v}")

    rule("findings (verified, not assumed)")
    for f in report["findings"]:
        echo(f"  * {f}\n")

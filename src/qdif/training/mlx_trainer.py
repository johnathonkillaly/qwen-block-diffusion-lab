"""MLX backend seam -- deliberately NOT implemented for milestone 1.

WHY THIS IS A STUB
------------------
The stated hardware preference for this project is MLX-native training on Apple
silicon, and MLX is installed and working on this machine. It is still the wrong
choice for the *first* milestone, for one concrete reason:

    `mlx_lm.models.qwen3_5` re-implements the Gated DeltaNet recurrence
    independently of the Hugging Face reference. Milestone 1 is about proving that
    a diffusion objective produces a valid loss and valid gradients through a
    *correct* Qwen3.5 forward pass. Introducing a second, independently written
    implementation of the hardest part of the architecture at the same time as
    introducing a new training objective means a wrong number cannot be attributed
    to either one.

So the torch backend is the reference, and MLX is the optimisation to bring up
against it once the reference produces trustworthy numbers.

WHAT THE MLX PORT NEEDS
-----------------------
Everything in `qdif.diffusion` is written against plain tensors and is
framework-agnostic in structure (it uses torch ops, but no autograd tricks, no
modules, and no device assumptions beyond `.to()`), so the port is:

1. `qdif/diffusion/corruption.py`, `schedule.py`, `objective.py` -> mx array ops.
   Mechanical. Keep the CPU RNG so seeds still reproduce across backends.
2. A `DiffusionQwen`-equivalent over `mlx_lm.models.qwen3_5`. The mask injection is
   the only real work: mlx-lm builds its causal mask internally per layer type, so
   this needs either a small upstream-shaped patch or a vendored subclass. Check
   whether mlx-lm's qwen3_5 accepts a precomputed mask before vendoring.
3. LoRA: `mlx_lm.tuner` already has LoRA for linear layers; the layer-index
   filtering in `qdif.training.lora` has to be reproduced.
4. Equivalence test: identical seeds, identical corruption, compare the torch and
   MLX loss on the same batch to within bf16 tolerance. Do not skip this.

MLX is already used today for the AR baseline -- see
`qdif.inference.ar_baseline.run_mlx_baseline`, which is implemented and working.
"""

from __future__ import annotations

from ..config import ExperimentConfig

NOT_IMPLEMENTED_MESSAGE = (
    "The MLX training backend is not implemented (milestone 1 uses the torch/MPS "
    "reference path so that Gated DeltaNet correctness and the diffusion objective "
    "are not being debugged at the same time). See docs/ARCHITECTURE.md, "
    "'Backend strategy', for the port plan. MLX *inference* baselines are available "
    "via `qdif compare --ar-backend mlx`."
)


def train(cfg: ExperimentConfig, **kwargs):
    raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


def is_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True

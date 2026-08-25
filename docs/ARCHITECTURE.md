# Code architecture

## Principle

Qwen-specific plumbing, diffusion math, training, sampling, evaluation and external
inference integrations are kept separable, so that a wrong number can be attributed
to exactly one of them.

```
src/qdif/
  config.py                 typed experiment config (YAML -> dataclasses, unknown keys rejected)
  cli.py                    the qdif command line

  env/
    inspect.py              hardware, backends, local models, baseline servers
    memory.py               memory estimates, split by consumer

  models/                   ---- Qwen-specific. The only place that knows about Qwen3.5.
    registry.py             read-only local checkpoint discovery
    qwen35_adapter.py       text-only loading + live architecture introspection
    masks.py                canvas mask construction (the bidirectionality decision)
    qwen35_diffusion.py     DiffusionQwen: sequence layout, conditioning, readout

  diffusion/                ---- Architecture-free math. No Qwen import anywhere.
    corruption.py           forward noising process
    schedule.py             timestep sampling and t -> probability mapping
    objective.py            x0 cross entropy + the anti-self-deception metrics
    sampler.py              iterative denoising, block-autoregressive generation
    self_conditioning.py    timestep + self-conditioning modules

  training/
    lora.py                 LoRA injection, target presets, freeze/enable, state dicts
    torch_trainer.py        the torch / MPS training loop
    mlx_trainer.py          documented stub + port plan
    diagnostics.py          metrics, memory, CLEAN/NOISY/PREDICTED rendering
    checkpoint.py           adapter-only save/load

  inference/
    ar_baseline.py          AR control (same weights, adapters off) + mlx-lm baseline
    diffusion_generate.py   block-diffusion generation entry point
    openai_client.py        LM Studio / OMLX clients, AR baselines only

  eval/
    reconstruction.py       the noise sweep + the bidirectionality probe

  data/
    dev_corpus.py           tiny public-domain + synthetic corpus
    datasets.py             clean (prefix, canvas) windows
    collator.py             batching + dynamic corruption
```

Everything in `diffusion/` is written against plain tensors and is unit-tested with no
checkpoint: 95 of the 123 tests never load a model.

## Backend strategy — and a deliberate deviation

The stated preference order for this project was MLX-native training first, then
PyTorch/Unsloth, then LM Studio / OMLX for baseline inference. **This milestone
implements the torch/MPS backend first instead.** The reason is specific, not general:

> `mlx_lm.models.qwen3_5` re-implements the Gated DeltaNet recurrence independently of
> the Hugging Face reference. Milestone 1 is about proving that a diffusion objective
> yields a valid loss and valid gradients through a *correct* Qwen3.5 forward pass.
> Bringing up a second, independently written implementation of the hardest part of
> the architecture at the same time as a new training objective means a wrong number
> cannot be attributed to either one.

So torch is the **reference**, and MLX is the **optimisation** to bring up against it
once the reference produces trustworthy numbers. MLX is installed and working, and the
MLX *inference* baseline is implemented today (`qdif compare --mlx`).

`training.backend: mlx` raises `NotImplementedError` with a pointer to the port plan
rather than silently falling back to torch.

### MLX port plan

1. Port `diffusion/{corruption,schedule,objective}.py` to `mx` arrays. Mechanical;
   keep the CPU RNG so seeds still reproduce across backends.
2. Build a `DiffusionQwen` equivalent over `mlx_lm.models.qwen3_5`. The mask injection
   is the only real work — mlx-lm builds its masks internally per layer type, so this
   needs either an upstream-shaped patch or a vendored subclass. Check whether it
   accepts a precomputed mask before vendoring.
3. LoRA: `mlx_lm.tuner` already has LoRA for linear layers; the layer-index filtering
   in `training/lora.py` has to be reproduced.
4. **Equivalence test before any training claim.** Identical seeds, identical
   corruption, compare torch and MLX loss on the same batch to bf16 tolerance.

### Why not LM Studio / OMLX for diffusion

Both expose OpenAI-compatible `/v1/completions`. Diffusion decoding needs to feed an
arbitrary noisy token canvas through the model and read logits at *every* canvas
position. No OpenAI-compatible endpoint exposes that. `inference/openai_client.py` is
therefore restricted to autoregressive baselines by design, and says so in its
docstring. Round-tripping text through a completions endpoint to imitate diffusion
would produce numbers that look like results and are not.

Both are useful as independent AR references, and `qdif inspect` reports whether they
are up.

## The one integration point into transformers

`DiffusionQwen` does **not** monkeypatch, subclass or vendor any part of
transformers. It calls `Qwen3_5TextModel.forward` with `attention_mask` as a dict:

```python
{"full_attention": <4-D additive mask>, "linear_attention": None}
```

which is a documented code path in the upstream implementation. See
[QWEN35_NOTES.md](QWEN35_NOTES.md). The consequence is that a transformers upgrade
can break us only by changing that signature, which `qdif smoke-forward` catches
immediately.

## Seams that exist for later work

| Seam | Purpose |
|---|---|
| `training.backend` | torch / mlx dispatch |
| `training.quantization` | int8 / int4; raises today with a reason, see [MEMORY.md](MEMORY.md) |
| `training.mode: full` | full-weight conversion; implemented, not default |
| `lora.target_preset` / `layer_indices` | which layer families get adapted (research question 2) |
| `diffusion.bidirectional_canvas` | the causal control condition |
| `diffusion.loss_on` | the copy-shortcut lever |
| `DiffusionQwen.forward(committed_spans=...)` | accepted and unused; reserved for block-level mask refinement |
| `sampler.strategy` | `confidence` vs the `all` control |

## Things deliberately not done yet

- **No KV / recurrent state caching in the sampler.** A committed prefix is causal and
  could be cached, but rewriting a canvas requires rolling the DeltaNet recurrent state
  back to the block boundary. Getting that wrong corrupts results silently. Cost today
  is O(blocks x prefix), irrelevant at correctness-scale canvases, and it is the first
  thing to optimise afterwards.
- **No NVFP4 backend.** Documented as a future Blackwell-class optimisation in
  [MEMORY.md](MEMORY.md). The diffusion algorithm is precision-agnostic and must stay
  that way; NVFP4 must not reach into `diffusion/`.
- **No gradient checkpointing.** Not needed at 0.8B on 128 GB; the activation estimator
  already accounts for what it would save.

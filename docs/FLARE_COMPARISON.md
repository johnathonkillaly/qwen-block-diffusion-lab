# FLARE and our dual-recurrence method

**FLARE: Diffusion for Hybrid Language Model** — Zhu, Shi, Ge, Tan, Xu, Zhu, Kuen,
Goswami, Jain, Chen, Tao, Gu. [arXiv:2606.01774v2](https://arxiv.org/abs/2606.01774)
· [project page](https://tokflare.github.io/) ·
[code](https://github.com/yuchen-zhu-zyc/HybridDiffusion) (PolyForm Noncommercial 1.0.0).

FLARE converts **Qwen3.5 hybrid Gated-DeltaNet checkpoints** into diffusion LLMs.
That is the same starting point and the same architecture family as this repository,
so it is the closest prior art for what v0.2 is doing and the right control to compare
against. **No novelty is claimed here.** FLARE is a full published conversion
framework at 2B/4B/9B with kernels and a serving stack; this repository is a
single-machine research harness. The purpose of this document is to state the
mechanisms precisely enough that a comparison means something.

---

## What FLARE does

**Objective (Eq. 4).** `L_FLARE = L_AR + L_diff`. Each block is shown to the model
twice in one forward pass: a **clean** left-to-right view carrying the standard
next-token loss, and a **noisy** masked view carrying the diffusion loss. The noisy
term uses complementary masks — `M_b` and `M_b^c` **partition** the block — so every
token contributes exactly one AR signal and one diffusion signal at unit weight,
independent of the masking fraction.

**Corruption.** Absorbing-state **masking**: `x^ℓ → [MASK]` for `ℓ ∈ M_b`, with
random mask fractions.

**Gated DeltaNet state scheduling** (the part that matters most here). For a noisy
block `b`:

- **Initialise from the clean boundary state:** `S̃_{e(b-1)} := S_{e(b-1)}`.
- **Update with noisy tokens only** (Eq. 6):
  `S̃_ℓ = α̃_ℓ (I − β̃_ℓ k̃_ℓ k̃_ℓ^T) S̃_{ℓ-1} + β̃_ℓ k̃_ℓ ṽ_ℓ^T`.
- **Read every position from the completed block-end state:**
  `õ_ℓ = S̃_{e(b)}^T q̃_ℓ / √d_k`.

That last line is the whole trick. The recurrence still runs strictly forward — it is
never reversed — but because every noisy position reads the state *after the entire
block has been absorbed*, each position sees the whole block. Block-wide visibility
without ever running the recurrence backwards.

**Softmax layers.** An additive `{0, −∞}` mask: causal for the clean stream,
bidirectional within noisy blocks, noisy-to-clean visibility allowed, document
boundaries closed.

**Kernels.** Route I ("chunk-then-refine") materialises every block-boundary state in
HBM and runs noisy blocks as a seeded local recurrence — a correctness reference with
`L/B`-scaling state hand-offs. Route II ("fused two-stream") stores strided
checkpoints, reconstructs the boundary state in registers and immediately consumes the
noisy block. At block size 1 Route II cuts reported GDN latency from 135.10 ms to
37.69 ms; Route I overtakes at B=16 where dense chunk matmuls saturate tensor cores.

**Their headline finding, which is inconvenient for us.** FLARE reports that
"the attainable transfer quality is governed primarily by the transfer data mix rather
than by the algorithmic recipe," and that once loss and mask design are aligned,
"transfer-data quality and distribution match dominate the residual performance gap."
In other words: at their scale, **attention-mask and loss design mattered less than
data.** This repository's entire v0.1→v0.2 arc is an investigation of mask and
recurrence design on a tiny dev corpus. FLARE's result predicts our axis is the one
with less headroom. It should temper any positive v0.2 finding, and it is a good
reason not to over-invest in architecture before data.

---

## How our method differs

The distinction is about **what representation position *i* gets**.

| | mechanism at canvas position *i* |
|---|---|
| **Ours (arm B, "aligned")** | Two *separate* recurrent representations: a forward one summarising the prefix and canvas positions ≤ *i*, and a reverse one summarising canvas positions ≥ *i*, obtained by running the **same pretrained weights over the reversed canvas** from a zero state. The two are then **fused** (`mean`, scalar gate, token gate, …) at the raw recurrent output. |
| **FLARE (arm D)** | One *shared* representation: the forward recurrence runs once from the clean boundary state through the whole noisy block, and **every** position reads the same completed **block-end** state `S̃_{e(b)}`. No reversal, no fusion, no second pass. |

Concretely:

```
ours    :  H_i = Fuse( S_i^fwd q_i ,  S_i^rev q_i )     two states, per-position, fused
FLARE   :  H_i = S_end q_i                              one state, shared by all positions
v0.1 (A):  H_i = S_i^fwd q_i                            causal, no block visibility
```

Consequences worth stating:

- **Cost.** Ours runs the recurrence a second time over the canvas (measured 1.23×
  forward, 1.19× forward+backward at 4B). FLARE's readout is a single extra matmul
  against a state we already compute — essentially free in our harness.
- **Position sensitivity.** FLARE's `S̃_{e(b)}` is *identical* for every position in
  the block, so position *i*'s output varies only through `q̃_i`. Ours gives position
  *i* a genuinely position-dependent suffix summary. That is the substantive
  difference, and it is exactly what our aligned-vs-shuffled control tests: if
  position-specific suffix information is not being used, our extra pass buys nothing
  that FLARE's cheaper readout does not.
- **Boundary handling.** FLARE seeds the noisy state from the *clean* boundary state.
  Our forward pass does the same thing implicitly (it runs straight through
  `[clean prefix | noisy canvas]`). But our **reverse** pass starts from a *zero*
  state at the canvas end — a regime the pretrained recurrence never saw during
  training. That is a real weakness of our method and a plausible explanation if it
  underperforms.

---

## What we implement, and what we do not

**Arm D in this repository is FLARE's block-end state readout only.** It is a
mechanism transplant into our harness, not a reproduction of FLARE. Specifically we
do **not** implement:

- FLARE's `L_AR + L_diff` dual objective (we train a pure diffusion objective),
- masked/absorbing-state corruption (we use uniform random-token replacement, per the
  v0.1 design — see `docs/DIFFUSION_OBJECTIVE.md`),
- complementary mask partitioning,
- their transfer data mix (the thing their paper says dominates),
- Route I/II fused kernels (unnecessary here: `mlx_lm`'s `gated_delta_update` already
  returns the final recurrent state, so the block-end readout costs one extra matmul
  and **no kernel work at all**),
- AR-Trust / Diffusion-Trust serving.

So **arm D is not a FLARE baseline and must not be reported as one.** A result where
D beats B says "the block-end readout mechanism beats our dual-recurrence mechanism
*under our objective, our corruption, and our data*" — nothing about FLARE's reported
numbers.

## Why the comparison is still worth running

1. It is the natural mechanistic control for our hypothesis. Both give the canvas
   block-wide visibility through the recurrence; they differ in whether that
   visibility is position-specific. If ours does not beat block-end readout, the
   second recurrence pass is not earning its 1.2× cost.
2. It is nearly free to implement and to run.
3. FLARE's data-over-architecture finding is itself a prediction about our
   experiment: it says the gaps between A, B and D should be small relative to what
   better data would buy. Recording whether that holds at our scale is worth
   something, even as a negative.

## Arms in the Phase 3 comparison

| arm | mechanism |
|---|---|
| **A** | causal DeltaNet + bidirectional periodic full attention (v0.1) |
| **B** | aligned forward/reverse shared-weight DeltaNet fusion (v0.2 hypothesis) |
| **C** | shuffled-reverse negative control (reverse output, canvas positions permuted) |
| **D** | FLARE-style block-end recurrent-state readout |

All four share the checkpoint, the corruption seeds, the data, the LoRA
configuration and the step count. The pre-registered kill/continue criteria in
`docs/EXPERIMENTS.md` are **not** modified by the addition of D; D is an extra
reference arm, not a new bar for B to clear.

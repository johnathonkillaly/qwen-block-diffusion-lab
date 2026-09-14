# Act IV-U design — Uno-style diffusion distillation on frozen Qwen3.5-4B-Base

Companion to [`act4u_uno_source_notes.md`](act4u_uno_source_notes.md) (what IFM
published) and [`act4u_preregistered_criteria.md`](act4u_preregistered_criteria.md)
(what would count as success). This file is the *how*.

## 1. The question

> Can a small adapter learn enough of a frozen AR model's future-token trajectory to
> propose blocks of K tokens that the frozen model itself will accept — preserving
> output exactly, and reducing sequential decoding steps?

Three achievements, deliberately separated, because conflating them is how this kind
of work overclaims:

| | claim | measured by |
|---|---|---|
| **A** | prediction quality | teacher-forced slot agreement |
| **B** | verified lossless decoding | token equality vs AR greedy |
| **C** | actual speedup | tokens/sec, and sequential forwards |

**B is already achieved and is independent of training** — the verifier guarantees it
for any adapter. **A is what training buys. C is A converted into throughput, and it
is the only one that can fail for engineering rather than scientific reasons.**

## 2. What is frozen and what is trained

```
Qwen3.5-4B-Base            4,205,751,296 params   FROZEN, byte-hashed before and after
  8 full-attention layers  [3,7,11,15,19,23,27,31]
 24 Gated DeltaNet layers  causal by construction, never masked or reversed

gated LoRA r=16 a=256         21,233,664 params   0.5023% of the backbone   TRAINABLE
  q,k,v,o on the 8 attention layers
  gate,up,down on all 32 MLPs
```

Trainability is structural: `configure_trainable` freezes the whole tree and unfreezes
an allowlist of `lora_a`/`lora_b`, then *asserts* that nothing else is trainable. A
backbone weight has no gradient tensor to receive an update.

## 3. The adapter is token-conditional, and that is the whole design

The adapter contributes **only on rows holding diffusion noise**. Prefill, the seed
row, verification and any ordinary AR call run the bare backbone. This is IFM's
"conditional gated LoRA" and it is what makes the verifier trustworthy: the model
doing the verifying is bit-identical to the model we are claiming to reproduce.

Implemented as a routing context (`gated_lora.py`) rather than IFM's PEFT forward
hooks, because MLX has no hook API. The seed-row invariant — student and teacher
logits must be *bit-identical* at the seed position — is asserted in stage 0 and in
`tests/test_uno_teacher.py`, and is a continuous check that the gate is not leaking.

## 4. Training layout

Uno noises every block at once and separates clean from noisy rows with a custom
`block_diff_mask` over a doubled sequence — one forward for both. **That mask cannot
exist on a Gated DeltaNet layer**, which is causal by construction rather than by
masking. So we use one noisy block per window, at the suffix, and two plain causal
forwards. For a window `x[0 … W-1]` with block size `L`, `s = W - L - 1`:

```
teacher :  x[0] … x[W-2]                      adapter OFF   -> frozen AR distributions
student :  x[0] … x[s]  ν₁ … ν_{L-1}          adapter ON on ν rows only
supervised: positions s … s+L-1, predicting x[s+1] … x[s+L]
```

Cost: two forwards, and `L-1` supervised draft positions per window. Benefit: it is
*exactly* the computation the draft pass performs at decode time, with no mask
approximation anywhere. Full rationale in `teacher.py`.

**Objective:** IFM's default, total variation between student and teacher over the
full vocabulary, `α=β=0, γ=1`. CE and reverse KL are implemented and selectable.

**Corruption:** `uniform` (t ~ U(0,1) per sequence, Uno/SDAR-faithful) or `full`
(every draft row noised, which is the only state that occurs at inference). Both are
run; neither is assumed.

## 5. Decoding

Per cycle, block size `L` (`decode.py`):

```
draft   [ seed , ν₁ … ν_{L-1} ]      adapter ON except the seed row
        -> c (exact AR token, seed row is unadapted) and proposals p₁ … p_{L-1}
verify  [ seed , c , p₁ … p_{L-1} ]  adapter OFF
        -> accept the longest prefix where pᵢ == argmax(verifier)
        -> commit c, the accepted prefix, and either a correction or a free lookahead
```

Fully accepted → `L+1` tokens from **2** forwards. Partially accepted → `j+2` tokens
from **3** forwards (the extra one replays the accepted tokens; see below). A cycle
never commits fewer than 2 tokens and never stalls.

### The architecture tax

IFM rewind the KV cache to an arbitrary length between passes. We cannot: a recurrence
has no "state 3 steps ago". So a partially-accepted cycle costs a third forward to
replay the accepted tokens from the last committed snapshot. The tax falls exactly on
the cycles where the adapter did badly — which is the right place for it, but it is
real, and it raises the acceptance rate needed to break even.

Consequences, both measured rather than assumed:

- **max TPF = (L+1)/2**, so `K=1` can never exceed 1.0 and is structurally a control.
- **`K=1` trains nothing.** At `L=1` there are no draft rows, the LoRA mask is
  all-zero, and the path is pure AR. The brief asks for K=1 as a learning sanity
  check; in Uno's actual design that is vacuous, so the sanity gate moves to `K=2`
  (one draft row). This is a documented deviation from the brief, forced by the
  algorithm rather than chosen.

## 6. Controls

| # | control | expectation |
|---|---|---|
| 1 | untrained adapter (zero-init ⇒ exact no-op) | the floor; **measured**, see criteria |
| 2 | shuffled teacher blocks (`shuffle_targets`) | must fail to improve |
| 5 | `K=1` | AR output at AR cost; TPF ≤ 1.0 by construction |
| 6 | native MTP head | inspected, **not run** — see below |

**Control 4 (ordinary next-token LoRA)** is the right control for "is this just extra
task adaptation?" and is implemented via `ce_weight>0, ce_target=data` at the same
parameter count.

**Control 6 (MTP):** Qwen3.5-4B ships an Eagle-style MTP head, but `mlx_lm`'s loader
drops it (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`, see
`docs/QWEN35_NOTES.md`). Using it as a speculative baseline means materialising and
wiring weights the MLX path never loads. Out of scope for this milestone; recorded as
not done rather than quietly skipped.

## 7. Known implementation realities

Both discovered on the real backbone and both reported rather than worked around:

1. **The backbone is not chunking-invariant.** Greedy decoding with a KV cache and
   greedy decoding with full recomputation produce *different token sequences* on
   Qwen3.5-4B — with no adapter involved at all. So exact-equality claims are made in
   the cacheless reference regime, and the cached regime is reported against the
   AR-vs-AR disagreement as its noise floor.
2. **Byte-hashing MLX arrays needs care.** `memoryview` on a lazy array, and
   `.view(mx.uint8)` on `layers.18.linear_attn.conv1d.weight`, both produced unstable
   digests for numerically identical tensors, i.e. a false "the backbone changed"
   alarm. Fixed by evaluating first and reinterpreting at the dtype's own width
   (`model.py:_canonical_bytes`).

## 8. Layout

```
src/qdif/uno/
  gated_lora.py   token-conditional LoRA + routing context + freezing
  model.py        UnoModel; ar_logits / draft_logits; backbone fingerprint
  noise.py        uniform / deterministic / mask draft noise
  teacher.py      training-batch layout, teacher and student logits
  losses.py       TV, reverse KL, CE, combine
  verifier.py     greedy acceptance + slow sequential reference
  decode.py       AR baseline and the Uno cycle, cached and uncached
  cache_utils.py  hybrid-cache snapshot/restore
  metrics.py      slot metrics, entropy buckets
  trainer.py      the training loop + adapter save/load
  data.py         wikitext windows + the held-out prompt suite
  tiny.py         a small real hybrid model for tests

scripts/uno.py    stage0 | baseline | overfit | train | bench
tests/test_uno_*.py
```

Deliberately independent of `qdif.mlx_backend`, which carries Acts I–IV-N's canvas,
bidirectional-DeltaNet and structured-noise machinery. None of that applies here, and
sharing it would risk changing runs those Acts still own.

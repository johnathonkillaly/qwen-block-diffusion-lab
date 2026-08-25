# Bidirectional Gated DeltaNet (v0.2)

## The hypothesis

v0.1 established that Qwen3.5's Gated DeltaNet is causal **by construction**, not by
masking: a causal depthwise `conv1d` feeds a sequential recurrence whose chunked form
hard-codes `tril`/`triu`. So v0.1 left all DeltaNet layers causal and obtained
backward information flow only from the periodic full-attention layers — 8 of 32 at
4B.

v0.2 asks whether a *pretrained causal recurrence* can supply useful bidirectional
features when the **same weights** are evaluated a second time on the reversed canvas:

```
H_f     = D(x_1 ... x_C ; W)      forward, exactly as pretrained
H_r_rev = D(x_C ... x_1 ; W)      reverse, same W
H_r     = reverse(H_r_rev)        realign to original positions
H_bi    = Fuse(H_f, H_r)
```

`W` is shared, never copied. Verified by object identity, not value equality
(`test_weights_are_shared_not_duplicated`).

## Where the fusion happens, and why

mlx-lm's `GatedDeltaNet.__call__` in order:

| step | operation | positional? |
|---|---|---|
| 1 | `in_proj_qkv` / `in_proj_z` / `in_proj_b` / `in_proj_a` | position-local |
| 2 | causal depthwise `conv1d` over qkv | **directional** |
| 3 | reshape to heads, `rms_norm` + scaling on q, k | position-local |
| 4 | `gated_delta_update` → `y [B, S, Hv, Dv]` | **the recurrence** |
| 5 | `norm(y, z)` — `RMSNormGated`, `z` from step 1 | position-local, nonlinear |
| 6 | `out_proj` → hidden_size | position-local, linear, bias-free |
| 7 | (`DecoderLayer`) `h = x + r` | residual |

**We fuse at the output of step 4** — the raw recurrent output, before the gated norm
and before the output projection.

Reasons, in order of importance:

1. **Step 4 is the only stage that carries directional information.** Steps 1, 3, 5
   and 6 are position-local; step 2 is directional but is part of what we want to
   *run* in reverse, not a place to combine.
2. **`z` stays correctly aligned.** The output gate is a position-local projection of
   the un-reversed input, so the forward pass's `z` is already the right gate for the
   fused representation, and gating happens exactly **once** with pretrained
   semantics intact.
3. **`out_proj` is applied once**, so exactly one contribution enters the residual
   stream. Fusing after `out_proj` would be *mathematically identical* for `mean`
   (out_proj is linear and bias-free) but would double-count for any gated fusion.
4. **The residual stays a single addition.** Running the whole layer twice and
   averaging post-residual outputs would silently halve the residual branch and
   change the layer's effective depth semantics. That is the failure mode this design
   exists to avoid, and it is why "just run it twice and average" was rejected.

### Uncertainty, stated plainly

The reverse pass starts from a **zero recurrent state** at the canvas end. The
pretrained recurrence has never been evaluated from a zero state mid-document —
during pretraining, position *i*'s state always accumulated real left context. So
`H_r` is not "what the model would have computed if text ran backwards"; it is the
model's operator applied to a reversed sequence from a cold start. Whether that is a
*useful* representation is precisely the empirical question, and the first
measurement (below) does not support it.

## The leakage boundary

Sequence layout is `[ fixed causal history | editable canvas ]`.

* Forward pass: full sequence, unchanged. Canvas position *i* sees the prefix and
  canvas positions ≤ *i*.
* Reverse pass: **canvas only**, `[P, T)`, from a zero state. Canvas position *i*
  additionally sees canvas positions ≥ *i*, and nothing outside the canvas.
* Prefix positions are **never fused** — they keep the pure forward representation.

Both required consequences hold:

* No canvas information reaches a prefix position, so a committed prefix is stable
  and block-autoregressive commitment stays sound.
* The reverse pass cannot see the prompt, so committed history is never re-entered
  from the future direction.

Running the reverse pass over the full sequence would give prefix positions
information about the canvas. That is the leak this design prevents, tested by:

* `test_prefix_is_never_touched_by_the_reverse_pass` — prefix output bit-identical
  with bidirectional on vs off;
* `test_canvas_edit_cannot_change_prefix_representation` — editing a canvas token
  leaves every prefix position bit-identical;
* `test_reverse_recurrence_carries_information_backwards` — and yet a later canvas
  token *does* move an earlier canvas one.

## State and cache semantics

`BidirectionalGatedDeltaNet` **raises** on any incremental cache. v0.1 noted that a
recurrent state goes stale the moment an earlier canvas token is rewritten; v0.2
refuses to risk silently-stale state and recomputes the canvas recurrence on every
denoising step. This is deliberately expensive. Prefix-state reuse, partial
recomputation, parallel scans and cache invalidation are later research.

## Fusion strategies

| strategy | formula | params/layer @4B | init behaviour |
|---|---|---|---|
| `mean` | `(H_f + H_r)/2` | 0 | **not** a no-op — halves the forward path immediately. The control. |
| `scalar_gate` | `g·H_f + (1−g)·H_r`, `g = σ(α)` | 1 | `g = 0.95`, starts near pretrained |
| `token_gate` | `g_i = σ(W_g[feat(H_f);feat(H_r)] + b_g)`, per token+head | 2,080 | `W_g = 0`, `b_g = logit(0.95)` → uniform 0.95 |
| `concat_proj` | `W_p[H_f;H_r]`, shared across heads | 32,768 | `[I;0]` → exact identity on `H_f` |

Two reductions are documented rather than hidden:

* `token_gate` gates per **(token, head)**, not per value channel. A full-channel
  gate would need `2·Hv·Dv → Hv·Dv` = 33 M parameters per layer at 4B. Gate features
  are per-head means over the value dimension.
* `concat_proj` shares one `2·Dv → Dv` projection across heads for the same reason.

### Controls (not training targets)

| control | formula | isolates |
|---|---|---|
| `forward_scaled_control` | `g·H_f` | attenuation of the forward path, no reverse info at all |
| `shuffled_reverse_control` | `g·H_f + (1−g)·shuffle(H_r)` | reverse magnitude/statistics with position alignment destroyed |

These exist because the gate sweep on its own is confounded: lowering `g` both admits
the reverse direction **and** attenuates the pretrained forward path.

## First measurement — and it does not support the hypothesis

Qwen3.5-4B-Base, **zero training** (LoRA and timestep conditioner both
zero-initialised, so the fusion is the *only* difference), 2 batches × 2, canvas 32,
identical seeded corruption in every condition, t = 0.5:

```
condition                      g=1.0    g=0.75     g=0.5    g=0.25     g=0.0
scalar_gate                   8.1838    7.8789    7.5729    7.3256    7.4303
forward_scaled_control        8.1838    8.1415    8.1928    8.4278   13.6508
shuffled_reverse_control      8.1807    7.8588    7.4774    6.8908    7.0739
```

Reading it:

1. **Attenuation is ruled out.** `forward_scaled_control` stays flat (8.14–8.43) and
   collapses at g = 0 (13.65). Simply scaling the forward path down does *not*
   reproduce the improvement. So the gain is genuinely coming from the second pass.
2. **But it is not position-aligned information.** `shuffled_reverse_control` — the
   same reverse output with its canvas positions randomly permuted — is **as good or
   better** at every gate value (6.89 vs 7.33 at g = 0.25).

If destroying the positional alignment of `H_r` does not hurt, then at initialisation
the benefit is not "position *i* learning about positions > *i*". It is an
unstructured effect: adding a correctly-scaled, roughly-decorrelated signal to the
DeltaNet output, which partially disrupts the model's autoregressive readout — and
disrupting AR-ness happens to help under the unshifted x0 objective (v0.1 established
that the unadapted model's whole problem is that it is still predicting token *i+1*).

**This is a negative result for the v0.2 hypothesis as stated, at initialisation.**

### What it does not show

It does **not** refute the hypothesis. The measurement is on an *untrained* model.
The entire premise of v0.2 is that adapters *learn* to use the reverse direction, and
a pretrained network has never seen a fused representation. It is unsurprising that
aligned and shuffled are indistinguishable before any learning.

What it does do is lower the prior, and make the held-out trained comparison the only
thing that can decide. It also means: **do not read a lower untrained loss as
evidence that bidirectional DeltaNet works.**

## Cost

Measured on 4B, canvas 32 + prefix 16, MLX/Metal, median of 3:

| | forward | forward+backward |
|---|---|---|
| causal (A) | 0.0586 s | 0.1440 s |
| bidirectional | 0.0721 s | 0.1707 s |
| **ratio** | **1.23×** | **1.19×** |

Lower than the naive 2× because only 24 of 32 layers are DeltaNet and the reverse
pass covers only the canvas (32 of 48 positions), not the full sequence. Training
throughput: 260 tok/s causal vs 212 tok/s bidirectional. Peak unified memory 10.16 GB
vs 10.41 GB.

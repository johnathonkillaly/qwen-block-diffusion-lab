# Uno source notes — what IFM actually published, and where we deviate

**Investigated 2026-09-03**, the day of the K2 Horizon release. Unlike Act III's
FLARE work — where we only had a paper and had to guess at the objective — the Uno
training code, the inference runtime and a released adapter are all public. So this
document is mostly *transcription*, not reconstruction.

Every claim below is tagged. Read the tags.

- **[C]onfirmed** — read directly out of IFM's code, config or model card.
- **[I]nference** — strongly implied by the code but not stated in words.
- **[A]pproximation** — our invention, because our model or hardware forces it.

**Nothing tagged [A] may be described as "Uno" in any result we write.**

---

## 0. Sources, with provenance

| what | where | verified |
|---|---|---|
| Runtime + training code | `github.com/ifm-ai/uno` | cloned at `pushed_at 2026-09-03T13:43:11Z`, Apache-2.0 |
| Repo tagline | *"Unlocking Lossless Speedups in LLMs via Discrete Diffusion"* | GitHub API description |
| Released adapter + frozen base | `huggingface.co/s-sahoo/uno-qwen3-8B` | model card, `adapter/adapter_config.json` |
| K2 adapters | `IFM/K2-Horizon-7B-Uno`, `IFM/K2-Horizon-0.9B-Uno` | HF org listing |
| Backbone modeling code | `modeling_sdar.py` on the adapter repo | 1302 lines, read |
| Press claims | PRNewswire release, 2026-09-03 | ~3× speedup, six models 0.9B–375B |

The citation block in the README is **empty** (`@article{uno2026, title={}, ...}`) —
there is no paper yet. The model card says the full evaluation suite is still to come.
So the code is currently the only authoritative specification, which is why this
document leans on it so heavily.

`s-sahoo` is Subham Sahoo (MDLM / discrete-diffusion author); the objective below is
recognisably from that lineage.

---

## 1. What Uno is — [C]

> "Uno applies the adapter selectively during draft-noise forwards. Seed, prefill,
> verification, and autoregressive rows use the frozen base weights. Loading the
> adapter as an ordinary always-on PEFT adapter does not reproduce Uno decoding."
> — `s-sahoo/uno-qwen3-8B` model card

That paragraph is the whole method in three sentences. Uno is:

- one set of **frozen** AR weights, `θ`;
- one **rank-128 LoRA** adapter, `φ`, which is applied **per token row**, only to
  rows that hold diffusion noise;
- a **two-forward decode cycle** (draft, then verify) where the verify pass runs the
  frozen model with the adapter off.

The press release's "diffusion distillation" is `φ` learning to imitate `θ`'s
future-token distributions. The "3× speedup" is speculative decoding where the draft
model *is* the target model wearing an adapter.

### Confirmed configuration [C]

| | value | source |
|---|---|---|
| LoRA rank / alpha | **128 / 2048** (alpha = 16 × rank) | `adapter_config.json`, `constants.py` |
| LoRA targets | `q,k,v,o,gate,up,down` — all 7 projections | `adapter_config.json` |
| Trainable | LoRA A/B **only**; asserted at runtime | `training/lora.py:validate_only_lora_trainable` |
| Objective default | `CE_ALPHA=0`, `KL_BETA=0`, **`TV_GAMMA=1`** | `constants.py` |
| Noise | `noise="uniform"` — **random token ids from `[1, mask_token_id)`** | `modeling.py:configure_uniform_noise`, `engine/noise.py` |
| Block curriculum | B = 2, 4, 6, 8, 12, 16; half an epoch each | `uno_3epoch_curriculum.yaml` |
| Training scale | 14.75B tokens, 28,125 steps, 16 GPUs, seq len 4096 | curriculum yaml |
| LR / warmup | 1e-5, 562 warmup, **no decay** | README, `constants.py` |
| Corpus | OpenThoughts3-1.2M, revision-pinned | `constants.py` |
| Eval sampler | linear, B=16, T=1.0, top-p 0.95, top-k 50 | model card |
| Reported | ~2.71 average TPF (tokens per forward) | `K2-Horizon-7B-Uno` card |

Note `noise="uniform"` means **random-token replacement**, not `[MASK]` absorbing
state. This matters to us — see §6.

---

## 2. The training objective — [C]

`nano_vllm_uno/training/losses.py` + `trainer.py`. The default is **total variation
distance between the student's and teacher's full-vocabulary distributions**:

```
L = γ · (1/N) · Σ_positions  ‖ softmax(z_student) − softmax(z_teacher) ‖₁
```

with `γ = 1`, `α = β = 0`. Implemented as a custom autograd Function
(`_ChunkedTotalVariation`) that chunks the vocabulary axis so it never materialises a
dense FP32 probability matrix. CE (with `1/p_mask` importance weighting, MDLM-style)
and reverse KL exist and are wired up, but are **off by default**.

TV rather than KL is a deliberate choice and it fits the decode rule: greedy
speculative acceptance depends on the *argmax agreeing*, and TV upper-bounds the
probability that two distributions disagree on any event. KL punishes tail mismatch
that acceptance never sees.

### Where the teacher comes from — [C], and this is the elegant part

There is **no separate teacher model and no separate teacher forward.** For each
training sequence, `prepare_for_bd_training` builds a length-`2n` input that
interleaves, per block, a **noisy copy** `x_t` and a **clean copy** `x_0` of the same
tokens. One transformer pass. Then:

- **clean rows** run with the LoRA masked to zero → they are exactly the frozen AR
  model → their logits are the **teacher**;
- **noisy rows** run with the LoRA active → they are the **student**;
- TV is taken between the two, position by position.

The gating is `TokenwiseLoraRouter` (`training/lora.py`): a forward hook on every
PEFT `lora_A` module that multiplies its output by a 0/1 token mask. Clean rows get
zero low-rank update, so `θ` is reproduced exactly.

`train/transformer_forwards: 1` is logged every step, which is the point.

### The attention mask — [C]

`block_diff_mask` in `modeling_sdar.py`, with `use_regular_causal=True` (the default).
Over the concatenated `[x_t ; x_0]` sequence, a query attends to a key iff:

1. **within-block, noisy→noisy, causal**: same block, both in `x_t`, `q_idx ≥ kv_idx`;
2. **offset block-causal, noisy→clean**: `block_q > block_kv` and key in `x_0`;
3. **token-causal, clean→clean**: `q_idx ≥ kv_idx`, both in `x_0`.

So a noisy token at block `b`, offset `j` sees **the true tokens of every earlier
block, plus the noise placeholders of its own block up to `j`** — and never the true
tokens of its own block. That is precisely the information state the draft pass has at
inference. The clean region is a plain causal AR pass, untouched.

---

## 3. The decode algorithm — [C]

`nano_vllm_uno/engine/two_pass_decoding.py`. Per cycle, for block length `L`:

```
draft   input  [ seed , noise₁ , … , noise_{L-1} ]     LoRA ON except the seed row
        output logits[0]      → clean token c   (pure AR: seed row has LoRA off)
               logits[1:]     → proposals p₁ … p_{L-1}

verify  input  [ c , p₁ , … , p_{L-1} ]                LoRA OFF everywhere
        output logits[i] = θ's next-token distribution after prefix …c,p₁…p_i
```

Then accept the longest prefix where `pᵢ == argmax(verify_logits[i-1])` (greedy) or
by the standard speculative accept/reject ratio (stochastic). Details that are easy to
get wrong and are worth copying exactly:

- **The seed row has LoRA disabled** (`lora_mask_batch[:, 0] = 0.0`). So the first
  token of every block is the exact frozen-AR token and is *always* correct — it costs
  nothing to verify and is committed unconditionally.
- **A lookahead token is free.** `logits[:, -1]` of the verify pass is `θ`'s
  next-token distribution after the *whole* accepted block. If the entire block was
  accepted, that token is committed too. So a fully-accepted cycle yields **`L+1`
  tokens from 2 forwards** — this is where TPF > L/2 comes from.
- **On rejection**, the first mismatched proposal is *replaced* by the verifier's own
  token (`_build_committed_payload`), so a rejected block still commits
  `first_reject + 2` tokens and never wastes a cycle.
- **Accounting**: `stats["forwards"] += 2` per cycle, unconditionally. TPF =
  committed tokens ÷ forwards. IFM's 2.71 TPF at `B=16` means ≈ 5.4 committed tokens
  per cycle.
- Noise for the draft rows is drawn from `[1, mask_token_id)` and there is a
  `deterministic_uniform` mode that hashes the prompt, for reproducible decoding.

A tree sampler also exists (`draft_tree.py`, FA3-only, `V=60`, candidate top-k 32).
**We are not reproducing the tree sampler** — it needs FlashAttention-3.

---

## 4. What we can reproduce faithfully

| Uno component | our status |
|---|---|
| Frozen backbone, LoRA-only trainable | **reproduce exactly** |
| Token-conditional (gated) LoRA | **reproduce exactly** — new `token_mask` on `LoRALinear` |
| Teacher = same weights, adapter off | **reproduce exactly** |
| TV objective over full vocabulary | **reproduce exactly** (chunked, MLX) |
| CE / reverse-KL alternatives | **reproduce** (implemented, off by default) |
| Two-pass draft→verify decode | **reproduce exactly**, including seed-row gating, lookahead token and rejection replacement |
| Greedy acceptance rule | **reproduce exactly** |
| Uniform random-token noise | **reproduce exactly** |
| Block-size curriculum | **reproduce** (2→16 shape, far fewer tokens) |
| Linear sampler with KV cache | **reproduce** in MLX |
| Tree sampler | **not attempted** — requires FA3 |
| 14.75B training tokens | **not attempted** — we are ~4 orders of magnitude smaller |

---

## 5. Where we must deviate, and why — [A]

### 5.1 The single-forward training layout does not survive Gated DeltaNet

This is the one substantive architectural deviation and it needs to be stated loudly.

Uno's one-pass trick relies on `block_diff_mask`, an arbitrary boolean attention mask
over a length-`2n` sequence. Qwen3-8B is a standard transformer, so every layer can
consume that mask.

**Qwen3.5-4B-Base is a hybrid: 24 of its 32 layers are Gated DeltaNet.** As
`FINDINGS.md` records, those layers are *causal by construction, not by masking* — a
causal depthwise conv feeding a sequential recurrence with `tril`/`triu` baked in. The
`attention_mask` they accept is a padding mask. **There is no way to give a DeltaNet
layer `block_diff_mask`.** Interleaving clean and noisy rows in one sequence would let
the recurrence carry clean-token information into noisy rows in a way we cannot gate,
which would leak the answer into the student and invalidate the entire experiment.

**Our approximation:** one noisy block per sequence, placed at the **suffix**, and two
plain causal forwards:

```
teacher :  [ x₁ … x_n , x_{n+1} … x_{n+L-1} ]              LoRA off, plain causal
student :  [ x₁ … x_n , seed=x_n? , ν₁ … ν_{L-1} ]         LoRA on noise rows only
```

This costs 2 forwards per step instead of 1, and supervises only `L-1` draft positions
per sequence instead of every block. In exchange it is **exactly the computation the
draft pass performs at inference** — arguably a *closer* train/test match than Uno's
own layout, since there is no mask approximation at all. We pay for it in throughput,
which is acceptable at our scale and is the honest trade.

Because the first `n` tokens are identical in both forwards and the adapter is off
there, the shared prefix can be computed once and reused via the KV cache. That
optimisation is deferred until the naive reference is proven correct.

### 5.2 Other deviations

- **Base, not instruct.** We use Qwen3.5-4B-Base with plain text; Uno trains on
  OpenThoughts reasoning traces with a prompt mask so only completions are supervised.
  We supervise all non-prefix positions.
- **`tie_word_embeddings=True`** for us, `False` for Uno-Qwen3-8B. The readout is the
  embedding table. This does not affect the method (we train no embedding rows) but it
  does mean our vocabulary is 248,320 wide against their 151,936, so the TV chunking
  matters more.
- **LoRA targets.** Uno adapts `q,k,v,o,gate,up,down` on all 36 layers. Only 8 of our
  32 layers have `q/k/v/o` at all; the other 24 expose DeltaNet projections
  (`in_proj_qkv`, `in_proj_z/b/a`, `out_proj`) that are *not* the same modules. Which
  of these to adapt is an empirical question we test as Variants A/B/C rather than
  assuming Uno's answer transfers.
- **Rank.** Uno uses r=128 at 8B. We start lower and sweep, because our budget is a
  few thousand steps rather than 28,125 — a rank we cannot train is worse than a
  smaller one we can.
- **No tree sampler, no CUDA graphs, no FA3, no torch.compile.** Every wall-clock
  number we produce is against our own unoptimised MLX AR baseline measured in the
  same harness, never against IFM's.

---

## 6. One thing worth flagging to our own prior work

Act II found that **uniform random-token corruption plateaued at ~6–7% masked
accuracy**, and Act III found absorbing-state `[MASK]` corruption was "what made
denoising work here". Uno uses **uniform random-token noise** and it evidently works.

These are not in conflict, and the reason is instructive. In Acts II/III the model had
to *reconstruct* corrupted positions from a mostly-corrupted canvas — the noise token
was where the answer had to come from. In Uno the noise token is a **placeholder that
carries no information and is not supposed to**: the answer comes from the clean
causal prefix, and the adapter's job is to learn to *ignore* the placeholder and
extrapolate. Uniform noise is arguably better for that than `[MASK]`, because it gives
the adapter no stable shortcut feature to latch onto.

We keep `mask` as a selectable noise mode so this is testable rather than asserted.

---

## 7. Honest summary of what remains unknown

- No paper, no training curves, no ablations. We cannot check whether TV beats CE at
  scale, only whether it does at ours.
- The 3× press figure has no published protocol behind it (batch size, hardware,
  baseline implementation). **We will not compare our numbers to it.**
- The 2.71 TPF figure is on the model card without a benchmark breakdown.
- We have not verified the released adapter runs, because it needs CUDA + FA2.

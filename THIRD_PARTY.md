# Third-party attribution and licence audit

*I am not a lawyer and this is not legal advice.* This file records what was actually
checked, on 2026-08-25, so that anyone redistributing or building on this repository
can make their own assessment.

## What is in this repository

All source under `src/`, `tests/`, `configs/`, `scripts/` and `docs/` was written for
this project. **No source was copied from any third-party repository.** In particular:

- The FLARE reference implementation
  ([HybridDiffusion](https://github.com/yuchen-zhu-zyc/HybridDiffusion), PolyForm
  Noncommercial 1.0.0) was **not** consulted for code and nothing was copied from it.
  Act III implements the *methodological ideas* from the paper's published
  mathematical specification. PolyForm Noncommercial restricts use of *their code*;
  it does not restrict independent implementation of published methods.
- `src/qdif/mlx_backend/deltanet.py::deltanet_front_end` deliberately re-expresses the
  *call sequence* of `mlx_lm.models.qwen3_5.GatedDeltaNet.__call__` so that it can be
  split around the recurrence. It calls the model's own submodules; it does not copy
  mlx-lm's source text. mlx-lm is MIT, which is compatible with a permissive licence
  here in any case. A test (`test_forward_path_matches_upstream`) asserts bit-exact
  agreement, which is also how we detect upstream drift.

No model weights, datasets, checkpoints, caches or virtual environments are committed.
`.gitignore` covers `*.safetensors`, `*.gguf`, `*.bin`, `runs/`, `.venv*/`.

## Dependency licences (verified from installed package metadata)

| package | licence | how we use it |
|---|---|---|
| `mlx`, `mlx-metal` | MIT | array/autograd runtime |
| `mlx-lm` | MIT | Qwen3.5 model definition and loader |
| `transformers` | Apache-2.0 | Act I/II torch backend, tokenizer/config utilities |
| `torch` | BSD-3-Clause | Act I/II backend only |
| `datasets`, `huggingface-hub`, `tokenizers`, `safetensors` | Apache-2.0 | data and I/O |
| `numpy` | BSD-3-Clause | arrays, RNG |
| `pyyaml` | MIT | config |
| `pytest` | MIT | tests |
| **`unsloth`** | **Apache-2.0** (wheel `License-Expression`, matching file headers) | device detection |
| **`unsloth_zoo`** | **see the discrepancy below** | optional Gated DeltaNet custom VJP |

### The `unsloth_zoo` licence discrepancy — read this before redistributing

The `unsloth_zoo` wheel metadata declares:

```
License-Expression: LGPL-3.0-or-later
```

but the per-file headers, **including the file we actually touch**
(`unsloth_zoo/gated_delta_vjp.py`), declare:

```
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
```

LGPL-3.0 and AGPL-3.0 have materially different obligations — LGPL explicitly
accommodates use as a library by differently-licensed works, AGPL does not and adds a
network-use provision. We have not resolved which governs; that is a question for the
Unsloth maintainers.

**How this repository responds to that uncertainty:**

1. We **do not vendor, copy or modify** any `unsloth_zoo` source. We call one
   documented public entry point at runtime, `patch_gated_delta()`, and only if the
   user has installed the package themselves.
2. The dependency is **optional**. `qdif` runs without Unsloth installed:
   `patch_unsloth_gated_delta()` catches `ImportError`, reports
   `unavailable (...)` in the run header, and mlx-lm's own `gated_delta_update` is used
   instead. Every run header states which path is active, so no result silently
   depends on it. `pyproject.toml` keeps Unsloth in an optional extra, never a required
   dependency.
3. Anyone distributing a **combined** work that bundles `unsloth_zoo` should take
   their own advice on which licence applies.

## Model weights

**`unsloth/Qwen3.5-4B-Base`** — model card declares `license: apache-2.0`, pointing at
`Qwen/Qwen3.5-4B-Base/LICENSE`. Weights are downloaded by the user; nothing is
redistributed here. Qwen3.5 is a product of the Qwen team / Alibaba. This project has
no affiliation with them.

Also referenced in earlier acts and not redistributed: `Qwen/Qwen3.5-0.8B`, and various
local GGUF/MLX quantised exports used only as autoregressive inference references.

## Datasets

| dataset | revision | licence | use |
|---|---|---|---|
| `Salesforce/wikitext` `wikitext-2-raw-v1` | `b08601e04326c79dfdd32d625aee71d232d685c3` | CC BY-SA 3.0 | Act II Phase 3 held-out comparison |
| `Salesforce/wikitext` `wikitext-103-raw-v1` | `b08601e04326c79dfdd32d625aee71d232d685c3` | CC BY-SA 3.0 | Act III transfer corpus |

WikiText is derived from Wikipedia and carries CC BY-SA 3.0. **We do not redistribute
it** — the user downloads it via `datasets` at run time, and no dataset content is
committed. Note that CC BY-SA is share-alike *for the data*; it does not attach to
independently written code that merely reads it, but anyone redistributing derived
*text* must honour it.

The tiny development corpus in `src/qdif/data/dev_corpus.py` is public-domain text
(Austen 1813, Carroll 1865, Melville 1851) plus synthetic sentences written for this
repository.

## Prior work and credit

- **FLARE: Diffusion for Hybrid Language Model** — Yuchen Zhu, Jing Shi, Chongjian Ge,
  Hao Tan, Yiran Xu, Wanrong Zhu, Jason Kuen, Koustava Goswami, Rajiv Jain, Yongxin
  Chen, Molei Tao, Jiuxiang Gu. [arXiv:2606.01774](https://arxiv.org/abs/2606.01774).
  Act III reproduces methodological ideas from this paper. Any errors in the
  reproduction are ours.
- **Qwen3.5** — the Qwen team, for the hybrid Gated-DeltaNet backbone.
- **Gated DeltaNet** and the DeltaNet line of work, on which Qwen3.5's linear-attention
  layers are based.
- **Unsloth** (Daniel Han-Chen, Michael Han-Chen and the Unsloth team) — for the
  Apple/MLX path and the Qwen3.5-specific Gated DeltaNet custom VJP.
- **MLX and mlx-lm** (Apple) — array framework and Qwen3.5 model implementation.
- **DiffusionGemma** — the original inspiration for attempting AR→diffusion conversion.

No affiliation with or endorsement by any of the above is claimed.

## Recommended licence for this repository

**Apache-2.0**, on the following reasoning:

- Every line of our implementation is independently written, so we are free to choose.
- Apache-2.0 matches the licence of the model weights we target (Qwen3.5) and of much
  of the dependency stack (transformers, datasets, huggingface-hub, unsloth itself).
- It is compatible with the MIT-licensed MLX components we build directly on.
- Unlike MIT, it carries an explicit patent grant, which is worth having for anything
  touching model architectures.
- It does **not** conflict with the optional Unsloth dependency, because we neither
  copy nor distribute Unsloth code. Users who bundle `unsloth_zoo` into a
  redistributed combined work inherit that package's obligations independently of our
  licence, which is why the discrepancy above is flagged rather than buried.

Apache-2.0 was chosen after this audit, not assumed by default.

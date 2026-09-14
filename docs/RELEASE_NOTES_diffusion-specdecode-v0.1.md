# Diffusion Speculative Decoding Research Preview — Qwen3.5-4B / MLX

**Tag:** `diffusion-specdecode-v0.1`. A research preview of a closed research project. This
is not a production release.

## What this is

The final state of a research project that asked whether a small diffusion-trained adapter
can draft several future tokens in parallel for an unmodified autoregressive model, which
verifies them and stays authoritative.

On the tested setup it can, modestly.

| | |
|---|---|
| Target model | `unsloth/Qwen3.5-4B-Base`, snapshot `61541fe4aed37b6a1e615edd5cbf5a3f660312b1`, bfloat16, frozen |
| Runtime | this repository's research Python harness, MLX 0.32.1, mlx-lm 0.31.3, Python 3.13.12 |
| Hardware | Apple M4 Max, 128 GB unified memory, macOS 27.0 |
| Decoding | greedy, fixed K=4 |
| Result | **1.245×** end-to-end tokens/s over native AR decoding **in the same harness** (60.32 vs 48.46 tok/s; 27 prompts × 3 repeats, 128 tokens); **1.230×** on loop-free text; reproduced at 1.248×, 1.238× and 1.256× in later sessions |

**Scope of the claim.** Only the configuration above. No claim is made about optimised
inference engines (llama.cpp, vLLM, LM Studio, `mlx_lm`'s generator), CUDA, other Qwen
models, sampling, other hardware, or speculative decoding in general.

**Correctness caveat (bf16 ties).** Output equals native greedy output except where the
target's top two logits tie in bfloat16. There a width-1 AR forward and a width-5 verify
forward can break the tie differently. That happens at about 1.24% of decoded positions.
Every audited divergence (759 in Act IV-S, 210 in Act IV-U6) was within one bf16 ULP of a
tie, and none was a decoder defect.

## Adapter asset

| | |
|---|---|
| File | `qwen3.5-4b-base-diffusion-drafter-u4b1-step16000.safetensors` |
| Size | 84,966,094 bytes |
| **SHA-256** | `8200c339d3d47363a3920fc4aca58f3535fc8bf75be431e35204f374700a4f42` |
| What it is | r=16 token-conditional LoRA (21,233,664 parameters) over full-attention and MLP modules. Its checkpoint name in the repository is `runs/u4b1/step-16000` (Act IV-U4 arm B1). It was trained to step 16,000, at block size 8 for its final 3,200 steps, with seed 20260903 and config hash `bd2dbffae35cc6db…` |
| Used by | every speed number above; frozen for Act IV-S and Act IV-U6 |

Verify it:

```bash
shasum -a 256 qwen3.5-4b-base-diffusion-drafter-u4b1-step16000.safetensors
```

Use it with the repository's scripts via `--adapter /path/to/qwen3.5-4b-base-diffusion-drafter-u4b1-step16000.safetensors`,
for example `scripts/spec_decode.py --adapter … ksweep --block-sizes 4`.

**Not included:** optimizer state, the other experimental checkpoints (their digests are in
`results/checkpoint_manifest.json`), the base model, and any dataset.

**Licensing.**
* The repository's code is Apache-2.0.
* The base model's card declares Apache-2.0.
* The adapter was trained on windows of WikiText-103 (CC BY-SA 3.0), with targets from the
  frozen base model's own distribution.
* No separate licence is declared for the adapter file. See `THIRD_PARTY.md` before
  redistributing it.

## What did not work

Each of these was tested and did not improve wall-clock decoding in this setup:
* **Refinement.** More diffusion refinement at inference raised accepted prefix
  1.07 → 2.78 and brought throughput down to 0.776× AR.
* **Wider decode blocks.** K=8 gave 1.128× and K=16 0.528× AR.
* **Adaptive K and confidence scheduling.** The best policy reached 1.230×, against 1.238×
  for fixed K=4.
* **Longer-horizon training.** K=6, K=8 and a curriculum, meant to improve K=4 decoding:
  `U4 OBSERVATION WAS NOISE`, and `NO CROSS-HORIZON TRANSFER` also holds at the tested
  budget.
* **Decoupled draft/verify width and staged verification.** Every staged arm was
  significantly slower than K=4 (−1.58% to −9.01%): `VERIFY COST DOMINATES`.

## What was learned

* **Algorithmic metrics are not wall-clock metrics.** Acceptance, tokens per forward and
  committed tokens per cycle all improved in arms that were slower.
* **A training launch is one draw.** Identical relaunches on MLX/Metal gave materially
  different adapters. The cause is not established.
* **Paired, interleaved timing is required.** Two identical decoder arms had medians 2.3%
  apart.

## Documentation

All links are to this tag.
* [README](https://github.com/johnathonkillaly/qwen-block-diffusion-lab/blob/diffusion-specdecode-v0.1/README.md)
* [Project summary](https://github.com/johnathonkillaly/qwen-block-diffusion-lab/blob/diffusion-specdecode-v0.1/docs/PROJECT_SUMMARY_THROUGH_U6.md)
* [Results index and raw evidence](https://github.com/johnathonkillaly/qwen-block-diffusion-lab/blob/diffusion-specdecode-v0.1/docs/RESULTS_INDEX.md)
* [Speculative decoding results](https://github.com/johnathonkillaly/qwen-block-diffusion-lab/blob/diffusion-specdecode-v0.1/docs/RESULTS_SPECULATIVE.md)
* [Act IV-U5 results](https://github.com/johnathonkillaly/qwen-block-diffusion-lab/blob/diffusion-specdecode-v0.1/docs/act4u5_results.md)
* [Act IV-U6 results](https://github.com/johnathonkillaly/qwen-block-diffusion-lab/blob/diffusion-specdecode-v0.1/docs/act4u6_results.md)
* [Findings](https://github.com/johnathonkillaly/qwen-block-diffusion-lab/blob/diffusion-specdecode-v0.1/FINDINGS.md)
* [Checkpoint manifest](https://github.com/johnathonkillaly/qwen-block-diffusion-lab/blob/diffusion-specdecode-v0.1/results/checkpoint_manifest.json)

**Status:** the research project is closed. No further stage is planned.

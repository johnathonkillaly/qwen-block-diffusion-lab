# Act III success criteria — pre-registered

**Written and committed before the main transfer run**, calibrated against the
measured initialisation baseline from `qdif act3-check` (commit of this file precedes
the run commit; see `git log`). Not to be moved afterwards. Acts I and II both
produced results that looked like success under weaker criteria, which is why these
exist.

## Measured baseline at initialisation (adapters zero-initialised)

| quantity | value |
|---|---|
| `L_AR` | 2.549 (reference perplexity ≈ 12.8) |
| `L_diff` | 8.539 |
| masked accuracy @ t=0.5 | 2.34% |
| visible preservation @ t=0.5 | 0.00% |
| canvas-conditioning L1 | 1.5046 (of a maximum 2.0) |
| trainable | 63,050,752 (1.477%) |

`visible_preservation = 0.00%` at init is correct, not broken: the unadapted model
reads out `logits[i] -> x[i+1]`, so at a *visible* canvas position it emits the next
token rather than the one in front of it.

## The failure signature we are testing against

Act II Phase 3 ended with held-out masked-position accuracy of **6–7%, essentially
independent of `t`**, copy rate ~3.5%, and lift over copy ≈ −83% at low noise. That is
a model that stopped reading the canvas. Every criterion below is chosen so that this
outcome cannot pass.

## HEALTH GATE — all seven must hold on held-out text

Evaluated on the WikiText-103 **validation** split, never trained on, at
`t ∈ {0.10, 0.25, 0.50, 0.75, 0.90, 1.00}`, at the final checkpoint.

1. **Noise dependence (the anti-flatness test).**
   `masked_accuracy(0.10) > masked_accuracy(0.50) > masked_accuracy(0.90)`,
   with each gap **≥ 3 percentage points**.
   *Phase 3 failed this: its curve was flat.*

2. **Low-noise reconstruction materially exceeds Phase 3.**
   `masked_accuracy(0.10) ≥ 25%`.
   *Phase 3's best low-noise number was 9.92%.*

3. **Visible-token preservation.**
   `visible_preservation(0.10) ≥ 80%` **and** `visible_preservation(0.50) ≥ 80%`.
   *Phase 3 scored ~7% here — it was destroying known-correct tokens.*

4. **Improvement over training, on the hard subset.**
   `masked_accuracy(0.50)` at the final step ≥ **2×** its value at step 0.

5. **Canvas conditioning does not collapse.**
   `canvas_l1 ≥ 0.5` at the final step (init baseline 1.5046; 0.0 would mean the model
   ignores `x_t` entirely).

6. **AR health is preserved.**
   `ar_reference_perplexity` at the final step ≤ **3×** its value at step 0
   (≈ 12.8 → must stay ≤ ~38.4).
   *Acts I/II degraded AR by 14.4× with no AR term in the objective.*

7. **All of the above on held-out text**, not the training split.

## Interpretation rules, fixed in advance

- **`t = 1.00` is not a reconstruction test.** With the whole canvas masked, the
  particular held-out continuation is not identifiable. Cross entropy there measures
  the prior. Exact reconstruction at `t = 1.00` is not evidence of anything and is not
  part of any criterion.
- **Falling `L_total` is not success.** Act I taught this. The gate is the seven items
  above.
- **`masked_accuracy` is the primary quantity**, not identity accuracy: under mask
  corruption, identity accuracy is inflated by the visible positions the model can
  simply copy.

## Outcome definitions

- **SUCCESS** — all seven hold. Proceed to iterative block generation and the
  AR-vs-diffusion demonstration.
- **PARTIAL** — 1, 3 and 5 hold (the model genuinely uses the canvas and preserves
  visible tokens) but 2 or 4 fall short. Report as a working-but-undertrained
  reproduction, with the throughput/step budget stated as the limiting factor.
- **FAILURE** — 1, 3 or 5 fails. The canvas-ignoring pathology has recurred despite
  the FLARE-style recipe. Diagnose before adding steps; do not simply train longer.

## Stop conditions

Abort and diagnose rather than spending more compute if any of:

- `canvas_l1` falls below 0.1 at any evaluation (canvas ignoring);
- `visible_preservation(0.10)` is below 20% after 25% of the run (catastrophic
  destruction of known-correct tokens);
- `ar_reference_perplexity` exceeds 10× baseline (AR collapse — the AR term is not
  doing its job);
- gradient norm is non-finite, or `L_diff` fails to fall at all over the first 25%.

## Known implementation gap

FLARE mentions "logit shift to the noisy-stream diffusion terms to align them with AR
semantics" without giving the formula in the sections available to us. We did **not**
guess at it. If Act III fails, this gap must be stated as a candidate cause before
attributing the failure to the method. See
[ACT3_OBJECTIVE.md](ACT3_OBJECTIVE.md) deviation 5.

## Supplementary training-launch robustness

*Not part of the preregistered scorecard. C_k8_r2 is not substituted for C_k8, the C launches are not averaged, and no frozen gate is rescored.* Criteria §11 A3–A4.

**Headline (principal metric: mean accepted prefix).**

> C_k8 is highly launch-sensitive under this training setup; no stable positive horizon-training effect is established.
> The C_k8 effect is launch-sensitive; the original frozen U5 result may be real but is not robust to an independent C training launch.
> The assumption that A_k4 launch spread is representative of C_k8 training variability is not supported by this check.

**Replication outcome.** Neither performance nor the mechanism profile replicates (first_slot_only vs none).

| metric | A_k4 launches | A mean | A band | C1 effect | C2 effect | \|C1 − C2\| | case |
|---|---|---|---|---|---|---|---|
| mean_accepted_prefix | 1.0283, 1.0288, 0.9759 | 1.0110 | 0.0529 | -0.0230 | +0.0716 | 0.0946 | 5 |
| tpf | 1.4145, 1.4371, 1.4035 | 1.4184 | 0.0336 | -0.0203 | +0.0245 | 0.0448 | 5 |
| tok_s | 57.6836, 57.5021, 56.3361 | 57.1739 | 1.3475 | -0.4747 | +1.2352 | 1.7099 | 5 |

| arm | Δ prefix vs A mean | > band | Δ TPF | > band | Δ tok/s | > band |
|---|---|---|---|---|---|---|
| B_k6@16000 | +0.0259 | no | +0.0130 | no | +0.5528 | no |
| C_k8@16000 | -0.0230 | no | -0.0203 | no | -0.4747 | no |
| C_k8_r2@16000 | +0.0716 | yes | +0.0245 | no | +1.2352 | no |
| D_curr@16000 | +0.0173 | no | -0.0039 | no | +0.3224 | no |

| launch | Δ +2 | Δ +3 | Δ +4 | profile |
|---|---|---|---|---|
| C1 | +0.0260 | -0.0026 | +0.0104 | first_slot_only |
| C2 | +0.0104 | -0.0026 | +0.0104 | none |

**Limits.** Two C_k8 launches do not establish C's standard deviation, a confidence interval across training launches, heteroskedasticity, or a distribution of K=8 outcomes. This check answers only whether the C effect survives one independent rerun, and whether the difference between the two launches looks roughly compatible with the A_k4 launch spread.

*u5-eval records each prompt's tok/s as the best of three keyed repeats. Repeats decode identical tokens, so the selection is on timing jitter only, and it applies identically to every arm in this session.*

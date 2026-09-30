# Failure-case audit

For each H6 dataset, this table selects the seed with the largest full-minus-numerical MSE difference among all five registered seeds. This is explicitly a post-hoc adverse-case selection, not a representative random sample or a model-selection rule. Positive differences mean degradation. No selected run is removed from the main table.

| dataset | seed | delta_mse | delta_mae | dominant_expert | dominant_mean_weight |
|---|---|---|---|---|---|
| Climate | 2024 | 0.001029 | 0.000231 | Low-rank | 0.999955 |
| Economy | 2026 | 0.01111 | 0.00499 | TextFiLM | 0.999637 |
| Energy | 2026 | 0.081231 | 0.106222 | TextFiLM | 0.999959 |
| Security | 2026 | 1.126514 | 0.11606 | Low-rank | 0.48337 |
| SocialGood | 2026 | 0.01692 | 0.025755 | TextFiLM | 0.999823 |

The dominant-expert column describes concurrent average routing concentration. It does not establish why a run failed: the learned backbone/adapter, optimization randomness, feedback and gate can interact. A route-disabled or matched-seed ablation changes the learning trajectory and is not a controlled intervention on an otherwise identical final model. Report the paired mean and SD alongside these cases, and do not explain every small difference as either noise or negative transfer.

The primary all-seed contrasts and improvement counts are in main_pairwise_analysis.md. The near-single-expert mechanism figures use fixed seed 2023 and were not chosen from this adverse-case table. Upstream text provenance remains insufficient to diagnose source-level semantic causes.

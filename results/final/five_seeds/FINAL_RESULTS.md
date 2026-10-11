# Five-seed final results (seeds 2025, 2026, 2027, 5, 6)

Plan: `reports/final5_20261010/PLAN.json` (sha256 `0e9e943f2a17`), ablation design frozen in `reports/seed5_final_20261010/ABLATION_PLAN.json` before any seed-5/6 result. Execution-only amendment: `PLAN_AMENDMENT_parallel.json`. Seeds 2025–2027 reuse the V1 main table and the On-Adapter confirmation; seeds 5 and 6 are new.

S = geometric mean over 36 tasks of the 5-seed mean CVaR₉₀ ratio to iTransformer (lower is better). Rank over 180 task-seed cells among 7 methods. All values recomputed from saved per-origin forecasts.

## 1. Main table

| Method | S ↓ | Mean CVaR₉₀ rank ↓ | Mean MAE | Tasks better than backbone | Tasks >2% worse | Worst task ratio | Per-seed S (2025/2026/2027/5/6) |
|---|---:|---:|---:|---:|---:|---:|---|
| iTransformer | 1.0000 | 3.844 | 1.0615 | 0/36 | 0 | 1.000 | 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 |
| On-Adapter (ours) | 1.0005 | 3.356 | 1.0699 | 23/36 | 3 | 1.072 | 1.0008 / 1.0012 / 0.9994 / 0.9972 / 1.0037 |
| MM-TSFlib | 1.0134 | 4.006 | 1.0665 | 14/36 | 9 | 1.148 | 1.0117 / 1.0177 / 1.0137 / 1.0074 / 1.0146 |
| GPT4MTS | 1.0362 | 3.989 | 1.0842 | 18/36 | 8 | 1.669 | 1.0312 / 1.0352 / 1.0393 / 1.0283 / 1.0415 |
| ADAPT-Z | 1.0501 | 3.856 | 1.1421 | 17/36 | 12 | 1.826 | 1.0430 / 1.0539 / 1.0505 / 1.0331 / 1.0663 |
| DSOF | 1.0547 | 3.956 | 1.2898 | 15/36 | 20 | 1.620 | 1.0398 / 1.0639 / 1.0470 / 1.0593 / 1.0585 |
| Original On-Adapter | 1.0799 | 4.133 | 1.0930 | 15/36 | 10 | 2.272 | 1.0676 / 1.0831 / 1.1208 / 1.0465 / 1.0592 |
| Proceed | 1.1118 | 4.994 | 1.1827 | 11/36 | 18 | 1.701 | 1.1038 / 1.1165 / 1.1107 / 1.0997 / 1.1238 |

Strictly first on S: False. Mean share of origins using the adapter: 0.44.

## 2. Ablations (all designs of the frozen plan)

| Method | S ↓ | Mean CVaR₉₀ rank ↓ | Mean MAE | Tasks better than backbone | Tasks >2% worse | Worst task ratio | Per-seed S (2025/2026/2027/5/6) |
|---|---:|---:|---:|---:|---:|---:|---|
| L0 Original On-Adapter | 1.0799 | 4.133 | 1.0930 | 15/36 | 10 | 2.272 | 1.0676 / 1.0831 / 1.1208 / 1.0465 / 1.0592 |
| L1 Original + online updates off | 1.0033 | 3.856 | 1.0644 | 15/36 | 2 | 1.027 | 1.0031 / 1.0029 / 1.0018 / 1.0031 / 1.0053 |
| L2 + VICReg off, 15 epochs | 1.0046 | 3.589 | 1.0648 | 20/36 | 5 | 1.056 | 1.0048 / 1.0044 / 1.0009 / 1.0015 / 1.0108 |
| L3 + backbone-first gate | 1.0009 | 3.561 | 1.0646 | 19/36 | 3 | 1.053 | 1.0019 / 0.9997 / 0.9986 / 1.0013 / 1.0029 |
| On-Adapter (ours) | 1.0005 | 3.356 | 1.0699 | 23/36 | 3 | 1.072 | 1.0008 / 1.0012 / 0.9994 / 0.9972 / 1.0037 |
| ours - gate | 1.0061 | 3.361 | 1.0719 | 24/36 | 8 | 1.079 | 1.0010 / 1.0098 / 1.0026 / 0.9993 / 1.0169 |
| ours - online updates off (lr 3e-4) | 1.0584 | 3.950 | 1.1110 | 17/36 | 10 | 1.912 | 1.0438 / 1.0580 / 1.0742 / 1.0367 / 1.0644 |
| ours - VICReg off (VICReg on) | 1.0007 | 3.433 | 1.0626 | 19/36 | 4 | 1.048 | 1.0015 / 0.9982 / 0.9967 / 0.9991 / 1.0073 |
| ours - 15 epochs (3 epochs) | 1.0007 | 3.489 | 1.0633 | 22/36 | 3 | 1.029 | 1.0018 / 0.9997 / 1.0000 / 1.0005 / 1.0018 |
| ours, gate gs 0.8 | 1.0007 | 3.433 | 1.0700 | 22/36 | 2 | 1.074 | 1.0003 / 1.0029 / 0.9971 / 0.9973 / 1.0057 |
| ours, gate gs 1.0 | 1.0001 | 3.433 | 1.0702 | 24/36 | 3 | 1.065 | 0.9999 / 1.0003 / 1.0007 / 0.9968 / 1.0024 |
| Original On-Adapter + gate | 1.0611 | 4.089 | 1.0821 | 15/36 | 9 | 2.051 | 1.0569 / 1.0631 / 1.0886 / 1.0413 / 1.0451 |

Ranks of ablation variants are computed with the variant in place of Original On-Adapter among the 7 methods.

## 3. Selected highlights (chosen after seeing the results; the full tables above are the complete record)

1. **Best tail-risk rank of all methods.** Mean CVaR₉₀ rank 3.356 over 180 task-seed cells; iTransformer 3.844, best other baseline 3.856.
2. **Lowest tail risk among all non-backbone methods, with the fewest harmful tasks.** S 1.0005 vs 1.0134 for the next best (MM-TSFlib); only 3 tasks more than 2% worse than the backbone (others 8–20); worst task 1.072 (others ≥ 1.148).
3. **Most tasks with a lower tail than the static backbone:** 23/36 (other methods 11–18).
4. **The online-update fix is the decisive mechanism.** Turning off parameter updates moves Original On-Adapter from S 1.0799 to 1.0033 and its worst task from 2.27 to 1.027; putting the original update rule back into our method raises S to 1.0584 (worst task 1.91).
5. **The backbone-first gate removes harm and is insensitive to its setting.** Without the gate S is 1.0061 with 8 harmful tasks; with it 1.0005 and 3; gate discount 0.8 / 0.9 / 1.0 gives S 1.0007 / 1.0005 / 1.0001.

## 4. What the results do not show

- On-Adapter is not strictly better than its static backbone on S (1.0005 vs 1); per seed it is below 1 on seeds 2027 and 5 and above 1 on 2025, 2026 and 6.
- Mean MAE is not an advantage: 1.0699 for On-Adapter vs 1.0615 for iTransformer and 1.0665 for MM-TSFlib.
- Removing VICReg-off or the 15-epoch budget changes S only slightly (1.0007, 1.0007); their contribution is mainly to rank and to the number of tasks better than the backbone. The CVaR loss improves rank (3.561 → 3.356) and S slightly (1.0009 → 1.0005).
- The gate alone does not rescue the original update rule (Original On-Adapter + gate: S 1.0611).
- Seeds 2025–2027 share the test period with development; results are exploratory.

## 5. Online-component ablations (plan `ONLINE_ABLATION_PLAN.json`, sha256 `389a309e5306`, frozen before these trajectories)

| Configuration | S ↓ | Mean rank ↓ | Tasks better than backbone | Tasks >2% worse | Worst task | Per-seed S (2025/2026/2027/5/6) |
|---|---:|---:|---:|---:|---:|---|
| On-Adapter (ours) | 1.0005 | 3.356 | 23/36 | 3 | 1.072 | 1.0008 / 1.0012 / 0.9994 / 0.9972 / 1.0037 |
| ours with static gate (validation choice) | 1.0064 | 3.394 | 25/36 | 8 | 1.079 | 1.0007 / 1.0100 / 1.0032 / 1.0001 / 1.0173 |
| ours - gate | 1.0061 | 3.361 | 24/36 | 8 | 1.079 | 1.0010 / 1.0098 / 1.0026 / 0.9993 / 1.0169 |
| ours - feedback (hg = ge = 0) | 1.0005 | 3.344 | 22/36 | 2 | 1.073 | 1.0010 / 1.0007 / 0.9995 / 0.9974 / 1.0036 |
| ours - feedback, no gate | 1.0068 | 3.339 | 24/36 | 8 | 1.089 | 1.0015 / 1.0103 / 1.0036 / 1.0001 / 1.0176 |

The static gate uses the adapter in 77% of task-seed cells.

6. **(selected highlight) The online gate is the working dynamic component.** Replacing it with a static choice made on validation raises S from 1.0005 to 1.0064 and the harmful tasks from 3 to 8 (SocialGood 1.058 vs 1.005, Health_US 1.047 vs 1.033).
- The matured-label gradient feedback (hg, ge) has no measurable effect: with it zeroed, S is 1.000482 vs 1.000495.

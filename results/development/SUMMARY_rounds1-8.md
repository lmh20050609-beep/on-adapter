# On-Adapter seed=2 nine-domain optimization: results of rounds 1–7 and the round-8 validation direction

Date: 2026-10-09. All results are **exploratory iterations**: the seed2 test period was used 7 times during rounds 1–7.

## 1. Fixed protocol and verdict criteria

- **Nine domains**, with fixed sampling frequencies and windows:
  - Environment: daily, L96, H = 48 / 96 / 192 / 336;
  - Energy, Health_US: weekly, L24, H = 12 / 24 / 36 / 48;
  - Traffic, SocialGood, Security, Economy, Climate, Agriculture: monthly, L8, H = 6 / 8 / 10 / 12.
- A single training seed = 2. The text is the V1 minimal repair with independent encoding. Numeric data, splits, scalers, targets, masks, forecast origins and scoring are all unchanged.
- **Main statistic:**

  S = exp( mean over the 36 domain/H tasks of log( CVaR₉₀(method) / CVaR₉₀(iTransformer) ) )

  The 36 tasks have equal weight; iTransformer's S is always 1; lower is better.
- **Success criterion:** S(On) strictly below all 6 baselines, including iTransformer.
- **Baselines:** the seed2 main experiment finished 252/252 cells with 0 failures (`method_comparison_seed2_v1_20261009`). Once built, the baseline results were reused as fixed and never rerun.

## 2. Conclusion

**Goal not met.** The best is round 5's On-Adapter-TA with S = **1.003279**: 2nd place, 0.33% behind the backbone iTransformer. It beats the other 5 baselines and is first among all online (dynamic) methods; the original On-Adapter is 1.0757.

## 3. Full comparison table for the best round (round 5)

| Method | Main statistic S | Mean CVaR₉₀ rank | Mean MAE | Mean MSE | Mean CVaR₉₀ | Tasks better than backbone |
|---|---:|---:|---:|---:|---:|---:|
| iTransformer (static) | **1.000000** | **3.472** | 1.0621 | 12.076 | 2.6345 | — |
| **On-Adapter-TA (round 5)** | **1.003279** | 3.556 | 1.0688 | 12.130 | 2.6504 | 17/36 |
| GPT4MTS (static, text) | 1.019648 | 3.778 | 1.0815 | 12.178 | 2.6334 | 17/36 |
| MM-TSFlib (static, text) | 1.021796 | 4.194 | 1.0760 | 12.237 | 2.7067 | 13/36 |
| DSOF (online) | 1.057524 | 3.889 | 1.2977 | 13.983 | 2.7857 | 17/36 |
| ADAPT-Z (online) | 1.061936 | 3.861 | 1.1677 | 14.556 | 2.9157 | 17/36 |
| Original On-Adapter (online, reference) | 1.075659 | — | 1.0936 | 12.354 | 2.7277 | 13/36 |
| Proceed (online) | 1.116890 | 5.250 | 1.1882 | 13.737 | 2.8824 | 8/36 |

Mean ranks are computed among 7 methods, with the round-5 version taking On-Adapter's place.

### Round-5 per-domain/H results (On / iTransformer CVaR₉₀ ratio; below 1 is better)

| Domain | 1st H | 2nd H | 3rd H | 4th H | Within-domain geometric mean |
|---|---:|---:|---:|---:|---:|
| Environment (48/96/192/336) | 1.0019 | 1.0010 | 0.9983 | 1.0063 | 1.0019 |
| Energy (12/24/36/48) | 1.0042 | 0.9970 | 0.9994 | 1.0149 | 1.0038 |
| **Health_US (12/24/36/48)** | 0.9905 | 1.0256 | **1.0725** | **1.0502** | **1.0342** |
| Traffic (6/8/10/12) | 0.9776 | 0.9724 | 0.9886 | 0.9957 | **0.9836** |
| SocialGood | 1.0383 | 0.9855 | 1.0003 | 1.0000 | 1.0059 |
| Security | 0.9940 | 0.9972 | 1.0192 | 1.0000 | 1.0026 |
| Economy | 0.9988 | 0.9963 | 1.0006 | 1.0012 | 0.9992 |
| Climate | 1.0004 | 0.9999 | 1.0010 | 1.0001 | 1.0004 |
| Agriculture | 0.9942 | 0.9975 | 1.0023 | 1.0010 | 0.9987 |

The remaining gap comes mainly from Health_US H24 / H36 / H48, with a combined log contribution of +0.144, followed by SocialGood H6 (+0.038).

## 4. Summary of all rounds

| Round | Hypothesis and change | Candidate selected on validation (validation score) | Test S | Place | Outcome |
|---|---|---|---:|---|---|
| 1 | Output-space online residual calibration instead of parameter-level online updates | lr0, a = 0.75, γ = 1 (0.9609) | 1.014567 | 2 | Not met. Climate, Energy, Health_US hurt on test |
| 2 | Residuals grouped by seasonal phase (P = 12 / 52 / 7) | phase, a = 0.75, γ = 1 (0.9425) | 1.012861 | 2 | Not met. Climate 0.92–0.94 on validation, 1.10–1.13 on test: reversed |
| 3 | Training side: turn off the dominant VICReg alignment term, zero-start residual, epoch cap 3 → 15 | a0e15 (0.9877) | 1.006174 | 2 | Not met. Validation improves; test no better than the original training with lr0 (1.0053) |
| 4 | a0e15 + in-stream expert gate (experts include phase/per-step calibration) | {T, phase, per-step}, gs = 0.7 (robust score 0.9809) | 1.008974 | 2 | Not met. Calibration components fail on test again |
| **5** | **a0e15 + backbone-first abstaining gate** | **gs = 0.9, μ = 0 (0.9895)** | **1.003279** | **2** | **Best so far.** Feedback delay at weekly long horizons makes the gate's evidence stale |
| 6 | a0e15 + small-step online updates (± gate) | lr = 3e-5 + gate (0.9883) | 1.003412 | 2 | Not met; tied with round 5 |
| 7 | Offline-learned seasonal-phase output bias (trained jointly with the adapter) | Round-5 configuration selected again (0.9895; new variant 0.9903) | 1.003279 | 2 | New direction rejected by validation |

Notes:
- In rounds 1–3 the selection score is the whole-validation geometric mean. From round 4 on, a robust score was declared in advance: the larger of the whole-stream geometric mean and the geometric mean of the worst of three time segments.
- Before each round ran, its plan was frozen (`roundN/PLAN.json` and its sha256); the test was computed only once, after selection.

## 5. Main lessons from the evidence

1. **The original On is under-trained.** The training objective is 4–17 while validation MSE is only 0.03–3: the VICReg alignment auxiliary term is one to two orders of magnitude larger than the task loss and dominates training. Turning it off and training longer improves validation (0.9934 → 0.9877).
2. **The validation and test segments behave differently.** Online residual calibration, learning rates and the direction of adapter corrections repeatedly work on validation and reverse on test. When the validation stream is cut into three segments, every online residual calibration loses its advantage in the last third.
3. **"Do no harm to the backbone" brought the biggest improvement.** The backbone-first abstaining gate lowers S from 1.0062 to 1.0033. Its limitation is that for weekly H36/H48, fully matured feedback arrives only after 36–48 origins, by which time the evidence is stale.
4. **Upper-bound estimate** (exploratory, extrapolated from test): even if round 5 fell back exactly to the backbone on Health_US H24–H48, S would be only about 0.9993, a lead of 0.07%. That is the same order as the numeric difference between On's internally computed CPU backbone and the main-experiment GPU backbone (at most 0.0073) and as single-seed fluctuation.
5. **Approaches considered but not adopted:**
   - residual shrinkage λ < 1: not supported on validation; forcing it from test experience would be tuning on test directly;
   - turning the adapter off per domain: would be manually setting domain-specific parameters from test results;
   - using the original online update (lr3e-4) as a gate expert: it diverges abruptly late in validation, faster than the gate can switch back.

## 6. Round 8: validation direction (done, test not used)

- **Hypothesis:** the adapter's correction changes sign across time segments, possibly partly because of the initialization and dropout randomness of a single adapter. Averaging several members can reduce that variance. Since MAE satisfies the triangle inequality and CVaR is a convex risk measure, the risk of the average is at most the average of the members' risks.
- **Procedure:** train 3 members with the a0e15 recipe, m = 0, 1, 2. m = 0 reuses the existing a0e15; for m = 1, 2 the random seed is set to 2 + 1000 × m before building the model. Forecasts are averaged arithmetically, with no member selection.
- **Candidates** (selected by the robust validation score):
  1. the round-5 configuration;
  2. the 3-member ensemble;
  3. the 3-member ensemble + backbone-first gate.
- **Test-use rule:** a test is computed once only if the selected candidate is not the round-5 configuration and its robust validation score is ≤ 0.985; otherwise the test segment is not touched.
- **Expectation:** low. If the sign changes on Health_US are a systematic shift between periods, all members err in the same direction and averaging does not help.
- **Plan:** `round8/PLAN.json`, sha256 `7012a1df…`.
- **Result (2026-10-09 15:35):** validation selected the round-5 configuration again (robust score 0.9895).
  - 3-member ensemble + gate: 0.9928 (whole stream 0.9870, early segment 0.9928);
  - 3-member ensemble: 1.0002 (whole stream 0.9857, but early segment 1.0002).
  - The ensemble improves the whole stream but worsens the early segment, so its gain is unstable across periods. It did not reach the 0.985 threshold, **the test was not computed**, and the final result remains round 5's S = 1.003279.

## 7. Interpretation and limitations

- Supportable statement: under the unified nine-domain protocol with seed 2, the improved On-Adapter (S = 1.0033) has the lowest tail risk of all online methods and also beats both static multimodal methods; it is essentially tied with, but does not beat, its own frozen static backbone.
- **Unequal tuning effort:** On went through 7 rounds; all other baselines use their original settings.
- There is only a single seed, and the seed2 test segment has been used many times, so the conclusions can only be considered exploratory.
- Credible evidence of "better than the backbone" would require freezing the method first and then running a test only once on new seeds.

## 8. Artifact paths

- Goal status and failure log: `C:\path\to\iadapter-baselines\reports\seed2_goal_20261009\GOAL_STATUS.md`
- Per-round plans, selections and results: `C:\path\to\iadapter-baselines\reports\seed2_goal_20261009\round1\` … `round8\` (`PLAN.json`, `PLAN.sha256`, `selection.json`, `results.json`)
- Results of the best round: `C:\path\to\iadapter-baselines\reports\seed2_goal_20261009\round5\results.json`
- Round-5 per-origin forecasts: `E:\FedNoise\iadapter-baselines\runs\seed2_round5_ta_20261009\<domain>\H<h>\ta_r0_predictions.npz`
- seed2 baseline main experiment: `E:\FedNoise\iadapter-baselines\runs\method_comparison_seed2_v1_20261009\`
- Training-variant checkpoints: `E:\FedNoise\iadapter-baselines\runs\method_comparison_seed2_onv3_a0e15_20261009\`
- Method packages (each with `FORK.json`): `method_comparison_seed2_v1_20261009`, `_onv3_`, `_onv4_`, `_onv5_`
- Scripts: `C:\path\to\iadapter-baselines\scripts\seed2_round{1..8}_*.py`, `onadapter_harness_20261008.py`

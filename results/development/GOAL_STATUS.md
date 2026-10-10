# GOAL_STATUS: On-Adapter first on the main statistic under the seed=2 nine-domain protocol

Goal: under the fixed nine-domain periodic protocol (V1 text, independent encoding, L/H as in PROTOCOL) and a single training seed = 2, the main statistic of On-Adapter,
S = exp(mean over 36 tasks of log CVaR90(method)/CVaR90(iTransformer)), must be strictly below all six baselines (including iTransformer = 1).
The verdict is recomputed from forecasts saved at full precision. Single-seed conclusions are exploratory only; this test period has been observed repeatedly.

## Status (2026-10-09 14:10)

| Item | Status |
|---|---|
| seed2 baseline compatibility | No pre-existing seed2 results; seeds 2025–2027 are not used as paired controls |
| seed2 main experiment | **252/252 done, 0 failures** (`method_comparison_seed2_v1_20261009`); baseline results are fixed |
| seed2 baseline main statistics | iTransformer 1.0000, GPT4MTS 1.0196, MM-TSFlib 1.0218, DSOF 1.0575, ADAPT-Z 1.0619, Proceed 1.1169; original On 1.0757 |
| Best On so far | Round 5 On-Adapter-TA, S = 1.003279 (2nd, 0.33% behind the backbone) |
| In progress | Nothing. Round 12 has finished its test (8th use of the seed2 test segment) |

## Round register

| Round | Hypothesis | Candidate family | Selection | Status |
|---|---|---|---|---|
| 1 | Output-space online residual calibration instead of parameter-level online updates | base∈{lr0,lr3e-4} × a × γ | GM of seed2 validation CVaR90 ratios, one global candidate | Selected lr0/a.75/γ1 (validation 0.961); test S = 1.014567, goal not met |
| 2 | Residual calibration grouped by seasonal phase (P = 12/52/7) | none/step/phase × a × γ | Same | Selected phase/a.75/γ1 (validation 0.9425); test S = 1.012861, goal not met |
| 3 | Training side: remove the dominant VICReg auxiliary term, zero-start residual, number of epochs | orig + a0e3/a0e15/a0z0e15/a1z0e15, online lr0 | Same (r0) | Selected a0e15 (validation 0.9877); test S = 1.006174, goal not met |
| 4 | a0e15 + in-stream expert gate (including phase/per-step calibration) | 3 expert sets × gs | Robust score max(whole, worst third) | Selected {T, phase, per-step} gs.7 (0.9809); test S = 1.008974, goal not met |
| 5 | a0e15 + backbone-first abstaining gate | gs × μ | Robust score | Selected gs.9 μ0 (0.9895); test S = 1.003279, goal not met (closest) |
| 6 | a0e15 + small-step online updates (± abstaining gate) | lr∈{0,3e-6,1e-5,3e-5} | Robust score | Selected lr3e-5 + gate (0.9883); test S = 1.003412, goal not met |
| 7 | Offline-learned seasonal-phase output bias (onv4) | a0e15/pb × gate | Robust score | Validation selected the round-5 configuration again (0.9895; pb + gate 0.9903); test S = 1.003279, goal not met |
| 8 | 3-member adapter ensemble (onv5, fixed initializations, averaged) | round-5 configuration / ens3 ± gate | Robust score; test threshold ≤ 0.985 | Validation selected the round-5 configuration again (ens3 + gate 0.9928, ens3 1.0002); test not triggered |
| 9 | Training loss aligned with the tail metric (onv6: mae / msetail) | See round9/PLAN.json | Same as round 8 | **Not started, at the user's request** (2026-10-09). Only the plan is frozen, plus 1 smoke cell; the derivation in scripts/seed2_round9_loss.py is unfinished and must be fixed before use |
| 10 | Online updates with task loss only (gate-entropy auxiliary term removed) | r5/r6/taskonly ± gate | Robust score; test threshold ≤ 0.985 | 0.9883, tied with round 6; test not triggered |
| 11 | Online updates with L2-SP anchoring to the offline weights (λ 0.01/0.1/1) | r5/r6/anchored ± gate | Same | Drift goes down but scores are unchanged (best 0.9883); test not triggered |
| 12 | Offline: CVaR variational tail loss (λ 0.1/0.5) / per-step output residual head | 8 candidates | Same | Selected CVaR λ0.5 + gate (0.9846, meets the threshold); tested once after user confirmation: S = 1.003775 (2nd); mean CVaR₉₀ rank 3.444, the best of the 7 methods; goal not met |

Development evidence (seed2025–27 validation streams only): `dev/h1_bias_screen.py`, `dev/h1b_js_screen.py`, `dev/h1c_screen.py`.
With a = 0.5 and γ = 0.97 fixed, the score is 0.9675 (lr0: 0.9969); James–Stein adaptive shrinkage and in-stream selected shrinkage are both worse; Agriculture is hurt.

## Failure log

- Round 1: validation 0.961 → test 1.0146. Climate (+6–7%), Energy and Health_US are hurt on test; the per-step residual mean is unstable across time segments (`round1/analyze_gap.py`).
- Round 2: phase calibration improves Climate/Energy/Health_US on validation and reverses on test (Climate 1.10–1.13). Splitting validation into three segments shows that every online residual calibration loses its advantage in the last third (`round3/dev/h3_robust_screen.py`): the gain from online residual calibration is transient, so it is no longer the main direction.
- Round 3: a0e15 improves validation 0.9934 → 0.9877, but its test 1.0062 is no better than the original training with lr0 (1.0053); Health_US H36/48 and SocialGood H6 reverse between validation and test.
- Round 4: gate including calibration experts, test 1.0090; the calibration components fail on test again.
- Round 5: abstaining gate 1.0033; the remaining gap is Health_US H24–48 (+0.144 in log) and SocialGood H6 (+0.038); for weekly long horizons, the full-maturity delay (36–48 origins) makes the gate's evidence stale.
- Residual shrinkage λ < 1 is not selected on validation; forcing a λ based on test experience would be tuning on test directly, so it was not adopted.
- Mechanism evidence: On's training objective is 4–17 while validation MSE is 0.03–3; the VICReg alignment auxiliary term dominates training (basis for round 3).

## Reproduction entry points (in WSL, first cd into the baseline directory)

- Main experiment: `./.venv/Scripts/python.exe -m benchmark_seed2_v1.runner plan|run` (resumable)
- Round 1: `bash scripts/seed2_round1_dispatch.sh` (detached from the session: `setsid nohup ... &`); single steps: `scripts/seed2_round1_rc.py --val-cell/--select/--test-cell/--score`
- Method package: `method_comparison_seed2_v1_20261009/` (FORK.json records the differences from the V1 package); config `configs/method_comparison_seed2_v1_20261009.json`

## Test-use policy (from round 8 on)

The seed2 test segment had been used 7 times. From then on, candidates are developed on validation only; a test is spent only when a candidate's robust validation score is ≤ 0.985 (the round-5 configuration scores 0.9895). Otherwise the test is not touched, to avoid selecting schemes on the same test segment.

Upper-bound estimate (exploratory, extrapolated from test): if round 5 fell back exactly to the backbone on Health_US H24–48, S ≈ 0.9993 — a lead of only 0.07%, the same order as the CPU/GPU numeric difference and single-seed fluctuation.

- Round 12: CVaR tail loss, validation 0.9744 (whole stream) / robust 0.9846, test S = 1.003775; before the gate 1.0116 (22/36 tasks better than the backbone, but the tail losses are concentrated). Health_US (validation 0.913 → test 1.036) and SocialGood (0.934 → 1.026) again reverse between validation and test; SocialGood H6 is still 1.082 even though the adapter is used at only 3% of origins, which shows that a few switches land on tail origins.

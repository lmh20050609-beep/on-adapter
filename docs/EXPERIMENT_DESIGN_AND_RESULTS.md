# On-Adapter: experimental design, algorithmic improvements and confirmation results

Compiled 2026-10-09. Every number in this document was recomputed from the saved per-origin forecasts; source paths are given in each section and in section 11.
Paths are relative to the repository root. Training outputs (checkpoints and per-origin forecasts) are not in the repository; on the original machine they were under `E:\FedNoise\iadapter-baselines\runs\`, and `…\` below stands for the original working directory `C:\Users\18452\Documents\ChatGPT\FedNoise\iadapter-baselines\`.

---

## 1. Summary

In the confirmation experiment — **frozen method, 3 independent training seeds (2025/2026/2027), baselines not tuned at all** — the improved On-Adapter (hereafter **On-Adapter-CV**):

- has **main statistic S = 1.000495**: essentially tied with its frozen static backbone iTransformer (S = 1.000000), 0.05% behind, **not strictly better**; 2nd of 7 methods;
- has **mean CVaR₉₀ rank 3.380, the best of the 7 methods** (iTransformer 3.843, the others 3.73–5.07);
- has lower tail risk than the backbone on **22/36** domain/H tasks;
- **beats all 5 other baselines**: about 1.4% ahead of the third-placed MM-TSFlib (1.0147) and more than 5% ahead of every online method;
- improves the main statistic by about 8.5% over the original On-Adapter (S = 1.0936), and lowers mean MAE from 1.0974 to 1.0716.

Almost all of the remaining gap is in Health_US (within-domain ratio 1.029).

---

## 2. Benchmark configuration

### 2.1 Data

- **Source:** Time-MMD multimodal time series (numeric series plus the text `fact` field), taken from the original paper's code and data supplement
  `C:\Users\18452\Downloads\31817_iAdapter_Adaptive_lightw_Code And Data Supplement (1).zip` and the CSV files of its 9 domains.
- **Text version: V1 minimal repair.** It masks only the individual entries verified against their sources as postdating the time they describe: 413 of 30,944 entries, 1.34%. Numeric columns, targets and masks are byte-identical.
  - data: `data/`
  - rules: `data_repair/source_evidence/decision_rules_all9.json` (348 rules, sha256 `85152a0e…`)
  - report: `results/data_repair/REPORT.md`
- **Text encoding:** independent encoding path. Each non-empty text gets its own forward pass with batch size always 1, so other texts in a batch cannot affect the result. Encodings are cached by identity, isolated from historical caches.

### 2.2 Domains, sampling frequencies and windows (fixed, no resampling)

| Domain | Frequency | Lookback L | Horizons H |
|---|---|---:|---|
| Environment | daily | 96 | 48 / 96 / 192 / 336 |
| Energy, Health_US | weekly | 24 | 12 / 24 / 36 / 48 |
| Traffic, SocialGood, Security, Economy, Climate, Agriculture | monthly | 8 | 6 / 8 / 10 / 12 |

9 domains × 4 horizons = **36 domain/H tasks**.

### 2.3 Splits and preprocessing (identical for all methods)

- Chronological 60% / 20% / 20% train / validation / test split;
- mean and standard deviation are fitted on the training segment only;
- missing inputs: forward-fill from past values, then the training mean; missing targets: masked, never imputed;
- the forecast-origin grid and the masks are the same for all methods;
- feedback rule for online methods: an old origin can be used for an update only when `old_origin + H ≤ current_origin` (all of its labels have matured);
- checkpoints are selected by validation MSE; nothing is tuned on the test segment.
- config: `configs/benchmark.json`

---

## 3. Baselines and sources

| Method | Type | Reads text | Code source (pinned commit) | Source of results in this study |
|---|---|---|---|---|
| iTransformer | static | no | thuml/Time-Series-Library `4e938a17` | Reused historical cells under the same protocol, verified by hash and numeric equivalence |
| ADAPT-Z | online | no | xiannanhuang/ADAPT-Z `f5b70e11` | Same as above |
| DSOF | online | no | yyalau/iclr2025_dsof `11cebd98` | Same as above |
| Proceed | online | no | SJTU-DMTai/OnlineTSF `0a864619` | Same as above |
| GPT4MTS | static, multimodal | yes: BERT-base, 512 tokens, CLS pooling | Flora-jia-jfr/GPT4MTS `c85e74b2` | Newly trained in the V1 main experiment |
| MM-TSFlib | static, multimodal | yes: first 6 layers of GPT-2, 256 tokens | AdityaLab/MM-TSFlib `e789ce78` | Newly trained in the V1 main experiment |
| On-Adapter (original) | online, multimodal | yes: first 6 layers of BERT-base, 64 tokens, mean pooling | Implementation from the original paper's supplement | Newly trained in the V1 main experiment |

- Pretrained weights: `google-bert/bert-base-uncased@86b5e093`, `openai-community/gpt2@607a30d7`.
- Source manifests: `third_party/code/*/SOURCE.json`, `third_party/weights/*/SOURCE.json`.
- **Baseline main table** (3 seeds × 36 tasks × 7 methods = 756 cells):
  - the 432 cells of the 4 methods that do not read text are reused from historical cells under the same protocol; see `benchmark/legacy_numeric_reuse_manifest.json`;
  - the 324 cells of the 3 text-reading methods were retrained on V1 with 0 failures; the audit passed, with 0 changes to 10,846 historical artifacts;
  - results: `results/benchmark/results.json`, `REPORT.md`.
- **No baseline was tuned** in this study.

---

## 4. Statistical method (frozen definitions)

Implementation: `cell_metrics` in `benchmark/score.py`. The confirmation experiment and all development rounds call this function directly.

1. **Scoring space:** all observed target elements of all selected numeric variables, on the training-standardized scale.
2. **Per-origin loss:** for each forecast origin, the MAE over all of its observed target elements.
3. **CVaR₉₀:** the mean loss within the worst 10% of probability mass of the empirical distribution with equal weight per origin; the boundary origin enters with a fractional weight. Cross-checked against the variational definition with error below 1e-12.
4. **Main statistic:**

   S(method) = exp( (1/36) · Σ_{domain,H} log( mean_seed[ CVaR₉₀(method) / CVaR₉₀(iTransformer) ] ) )

   That is: in each domain/H/seed cell, divide by the iTransformer of the same cell; average over the 3 seeds; then take the equally weighted geometric mean over the 36 tasks. iTransformer is always 1; lower is better.
5. **Also reported:**
   - mean CVaR₉₀ rank: the 7 methods are ranked within each of the 108 domain/H/seed cells and the ranks are averaged;
   - mean MAE, mean MSE, mean CVaR₉₀;
   - number of tasks whose tail is below the backbone's;
   - per-domain/H results.
6. **Success criterion:** S(On) strictly below all 6 other methods, judged at full precision without rounding.

---

## 5. Structure and diagnosis of the original On-Adapter

**Structure:**
- the frozen iTransformer encoder outputs a per-variable latent feature z;
- the IG-SMoE adapter fuses 5 experts (Identity / Trend / Seasonality / Fluctuation / TextFiLM); its inputs are z, the historical gradient feedback hg, the text vector and text gradient ge, and window-shift statistics;
- the residual is added to z and passed through the frozen output head to give the forecast;
- training loss: MSE + gate entropy + VICReg text–numeric alignment;
- at test time, once an origin's labels mature, the adapter takes one online Adam step (lr 3e-4).

**Problems found by diagnosis** (evidence on the original machine, not in this repository: `…\reports\hg_scale_diagnostic_20261008\`, `online_replay_consistency_20261008\`, `domain_gap_analysis_20261008\`):

| Problem | Evidence |
|---|---|
| Online parameter updates diverge on non-monthly domains | With updates off, the Energy CVaR ratio goes from about 1.96 back to about 1.00; on weekly domains the correction grows 12–13× by the end of the stream |
| The VICReg alignment term dominates training | The training objective is 4–17 while validation MSE is only 0.03–3; the adapter learns almost no forecast correction |
| Training budget too small | The original rule trains On for only 3 epochs; with a cap of 15, some cells were still improving at epoch 15 |
| Training loss does not match the evaluation metric | Training uses element-wise MSE; evaluation looks at the tail of per-origin MAE |
| The scale of the gradient feedback hg differs by about 111× between training and test | Real, but neither turning the feedback off nor rescaling it per batch removes the damage — **not the main cause** |

Engineering check: after restoring the entry random state on the GPU, the online trajectories of two historical cells were reproduced bit for bit; the replay harness is bitwise identical to `evaluate_online` in the original package.

---

## 6. Algorithmic improvements: the final method On-Adapter-CV

Compared with the original On-Adapter, only the following 5 things change; everything else (data, backbone, expert structure, text encoding, feedback computation) is unchanged.

| Change | Content | Basis (development round) |
|---|---|---|
| ① Turn off the VICReg alignment term | Alignment weight set to 0 | Round 3: removes the domination of training by the auxiliary loss |
| ② Training budget | On's epoch cap goes from 3 to 15; early-stopping patience stays at 3; checkpoints are still chosen by validation MSE | Round 3: validation 0.9934 → 0.9877 |
| ③ CVaR₉₀ tail loss | L = MSE + λ·[τ + mean(max(e_i − τ, 0)) / 0.1], λ = 0.5; e_i is the observed per-origin MAE, with the same variables, mask and scale as the scorer; τ is an exponential moving average, across batches, of each batch's 90% quantile | Round 12: whole-stream validation 0.9744, the best of all development |
| ④ Backbone-first abstaining gate | Output by default the frozen-backbone forecast computed inside On; switch to the adapter when its discounted mean per-origin MAE (γ = 0.9) on matured origins is below the backbone's; require at least 4 matured origins, with hysteresis | Round 5: test from 1.0062 to 1.0033 |
| ⑤ Turn off online parameter updates (lr 0) | The online hg/ge feedback and the gate decisions still run at every origin; only the parameters are not updated | Rounds 6, 10, 11: small steps, removing the auxiliary term and anchoring brought no validation gain |

Information boundary:
- the gate and the feedback use only matured labels with `o + H ≤ t`;
- a forecast never reads current or future targets or masks;
- verified: changing the values of not-yet-matured targets does not change any earlier forecast.

Code:
- method package: `on_adapter/` (`FORK.json` records every difference)
- gate: `abstain` in `confirmation/round5_gate.py`
- replay harness: `confirmation/harness.py`

---

## 7. Development process (seed = 2, exploratory)

- A full 7-method baseline was built for seed 2 (252/252 cells, 0 failures); see `…\reports\benchmark_seed2_v1_20261009\` on the original machine (not in this repository).
- Each round froze its plan first (`PLAN.json` and its sha256), selected on the validation set only, and computed the test set only once.
- From round 4 on, selection used a robust score: the worse of the whole-stream geometric mean and the geometric mean of the worst of three time segments.
- From round 8 on, the test set was used only when the robust validation score was ≤ 0.985.

| Round | Change | Validation (robust score) | Test S | Outcome |
|---|---|---:|---:|---|
| 1 | Per-step online residual calibration | 0.961 | 1.0146 | Validation–test reversal |
| 2 | Seasonal-phase residual calibration | 0.9425 | 1.0129 | Same |
| 3 | ①② training fixes | 0.9877 | 1.0062 | The adapter starts learning corrections |
| 4 | Gate + calibration experts | 0.9809 | 1.0090 | Calibration components fail |
| 5 | ④ backbone-first gate | 0.9895 | 1.0033 | Best main statistic on seed 2 |
| 6 | Add small-step online updates | 0.9883 | 1.0034 | Tied with round 5 |
| 7 | Offline seasonal bias | Selected round 5 again | — | Rejected |
| 8 | 3-member ensemble | 0.9928 | not tested | Unstable across time segments |
| 10 | Online updates without the auxiliary term | 0.9883 | not tested | No difference |
| 11 | Online anchoring (L2-SP) | 0.9883 | not tested | Drift drops but scores are unchanged |
| 12 | ③ CVaR tail loss / output residual head | **0.9846** | 1.0038 | Best mean rank of the 7 methods |

The seed-2 test segment was used 8 times in total; all of the above results are exploratory.

Summaries: `results/development/SUMMARY_rounds1-8.md`, `GOAL_STATUS.md`; per-round details: `results/development/round*/`.

---

## 8. Confirmation experiment and results (seeds 2025 / 2026 / 2027)

**Design:**
- plan: `results/confirmation/PLAN.json` (sha256 `ead8c55f…`), frozen before any confirmation trajectory was produced;
- the primary method was pre-specified as On-Adapter-CV with no selection; the round-5 configuration On-Adapter-TA is reported alongside as a control;
- only On's 108 cells were newly trained; the backbone is the same set of iTransformer checkpoints as in the main table;
- the online entry random seed is 100 × seed;
- the baselines are taken directly from the frozen V1 main table.

**Main results** (`results/confirmation/results.json`):

| Method | Main statistic S | Mean CVaR₉₀ rank | Mean MAE | Mean MSE | Mean CVaR₉₀ | Tasks with lower tail than backbone |
|---|---:|---:|---:|---:|---:|---:|
| iTransformer | **1.000000** | 3.843 | 1.0628 | 11.978 | 2.6293 | — |
| **On-Adapter-CV** | **1.000495** | **3.380** | 1.0716 | 12.100 | 2.6380 | **22/36** |
| &emsp;before the gate (the adapter alone) | 1.004584 | — | 1.0733 | 12.079 | 2.6484 | 23/36 |
| MM-TSFlib | 1.014709 | 4.009 | 1.0690 | 12.089 | 2.6775 | 16/36 |
| GPT4MTS | 1.035984 | 4.000 | 1.0879 | 12.205 | 2.6616 | 16/36 |
| ADAPT-Z | 1.049515 | 3.731 | 1.1421 | 13.390 | 2.7875 | 18/36 |
| DSOF | 1.050956 | 3.972 | 1.2860 | 13.904 | 2.7706 | 15/36 |
| On-Adapter (original) | 1.093644 | — | 1.0974 | 12.210 | 2.7404 | 17/36 |
| Proceed | 1.110804 | 5.065 | 1.1752 | 13.486 | 2.8612 | 11/36 |

**Per-seed main statistic:** 2025 → 1.00082, 2026 → 1.00118, 2027 → **0.99938**.

**Per-domain CVaR₉₀ relative to the backbone** (averaged over 3 seeds, then geometric mean over the 4 horizons):

| Domain | On-Adapter-CV | Before the gate | Share of origins using the adapter |
|---|---:|---:|---:|
| Environment | **0.990** | 0.986 | 0.60 |
| Energy | 1.002 | 1.002 | 0.43 |
| Health_US | 1.029 | 1.044 | 0.46 |
| Traffic | **0.983** | 0.983 | 0.81 |
| SocialGood | 1.006 | 1.049 | 0.12 |
| Security | **0.999** | 0.995 | 0.23 |
| Economy | **0.998** | 0.995 | 0.54 |
| Climate | 1.003 | 0.996 | 0.25 |
| Agriculture | **0.995** | 0.995 | 0.58 |

Worst tasks: Health_US H48 1.058, H36 1.027, H24 1.022; SocialGood H8 1.016.

**Control On-Adapter-TA (round-5 configuration, no CVaR loss):** completed under the same frozen plan as the primary method.

| Method | Main statistic S | Mean CVaR₉₀ rank | Mean MAE | Mean MSE | Tasks with lower tail than backbone | Per-seed S |
|---|---:|---:|---:|---:|---:|---|
| On-Adapter-CV (primary) | 1.000495 | **3.380** | 1.0716 | 12.100 | **22/36** | 1.00082 / 1.00118 / 0.99938 |
| On-Adapter-TA (ablation) | **1.000108** | 3.481 | **1.0655** | **12.008** | 21/36 | 1.00192 / 0.99972 / 0.99857 |

Per-domain comparison (CV / TA):
- CV better: Environment 0.990 / 1.002, Agriculture 0.995 / 0.998, Traffic 0.983 / 0.985;
- TA better: Health_US 1.029 / 1.018, SocialGood 1.006 / 1.000, Climate 1.003 / 0.999.

The CVaR loss improves the mean rank but does not lower the main statistic, because it worsens the tail on Health_US. Both tie the backbone; neither strictly wins.

Run log: the control's training once left a stale GPU lock after a process interruption; after manually confirming that the original process no longer existed, it was resumed from checkpoint. See `results/confirmation/recovery/NOTE.txt`.

---

## 9. Strengths of our method (based on the confirmation data in section 8)

1. **Best tail-risk rank.** Mean CVaR₉₀ rank 3.380 is the best of the 7 methods, about 0.46 ranks ahead of the static backbone (3.843). On has lower tail risk than the backbone on 22/36 tasks.
2. **Beats all 5 other baselines, including every online and every multimodal method.** Main statistic 1.4% ahead of MM-TSFlib, 3.5% ahead of GPT4MTS, and 5%–11% ahead of ADAPT-Z, DSOF and Proceed.
3. **Large improvement over the original On-Adapter, from interpretable mechanism fixes.** Main statistic from 1.0936 to 1.0005; mean MAE from 1.0974 to 1.0716; tasks with a lower tail than the backbone from 17 to 22.
4. **Robustness from "do no harm to the backbone".** The gate lowers the adapter's own tail risk (1.0046) further to 1.0005; on SocialGood, where the adapter is error-prone, it uses the adapter at only 12% of origins.
5. **Stable across training randomness.** The main statistic of all 3 seeds lies in 0.9994–1.0012; the conclusion does not change once the method is frozen.
6. **Clear ablation.** Without the CVaR loss, the main statistic is 1.0001 (below 1 on 2 of 3 seeds) and the mean rank is 3.48. The backbone-first gate plus the training fixes alone already reach a tie with the backbone; the CVaR loss buys a better mean rank at the cost of the Health_US tail.

**What was not achieved (stated plainly):** the main statistic is not strictly below the backbone iTransformer; it is 0.05% behind. The remaining gap comes mainly from Health_US, where validation–test reversals recur across several methods.

---

## 10. Limitations and disclosures

- **Same test period:** seeds 2025–2027 share the test period with development seed 2. The confirmation tests robustness to training randomness, not generalization to new data. The V1 main table had already been observed.
- **Unequal tuning effort:** On went through 12 development rounds; the baselines all use their original settings.
- **Backbone numeric difference:** On's internal backbone forecast is computed on the CPU; its maximum absolute difference from the main-table iTransformer (GPU) is 0.0101.
- **Online parameter updates are off:** the online component keeps only the feedback and the gate. The small-step online-update version (round 6) scores close, but there is no evidence that parameter updates add anything.
- **Text causality:** the numeric data still use the original `end_date` availability proxy; V1 only repairs confirmed timing mismatches, so we cannot claim the text is entirely leak-free.
- **No claim of significance, universal superiority or SOTA.**

---

## 11. Reproduction entry points (from the repository root)

- V1 main table: `python -m benchmark.score`
- Confirmation: `bash confirmation/dispatch.sh`; scoring only: `python confirmation/confirm.py --score`
- Per-origin confirmation forecasts (after and before the gate, backbone, targets, masks, origins, gate choices): `E:\FedNoise\iadapter-baselines\runs\v1conf_traj_20261009\final\cv\<domain>\H<h>\seed<s>\predictions.npz`
- Confirmation checkpoints: `E:\FedNoise\iadapter-baselines\runs\method_comparison_v1conf_cv_20261009\`

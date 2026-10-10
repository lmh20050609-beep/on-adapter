# Handoff: full nine-domain V1 text-repair evaluation of On-Adapter

2026-10-08. This document hands the current work over to the Claude Code CLI (or another session) to continue.

## 0. Status in one sentence

**Everything is done.** Task 1 (9-domain V1 data and source grading), task 2 (protocol freeze) and task 3 (51 path checks)
are complete; the full 756-cell evaluation (324 newly trained + 432 reused) finished **324/324 training cells with 0 failures**;
the wrap-up pipeline (audit / tables / compare / score / manifest / build_report) has been run in full.
The final report is [REPORT.md](REPORT.md) in the same folder.

### 0.1 Summary of the final conclusions

Main statistic (equal-weight geometric mean across domain/H tasks of CVaR90 relative to the iTransformer backbone; lower is better):

| Method | V1 | History (V0 + old encoding) | Relative difference |
|---|---|---|---|
| iTransformer | 1.0000 | 1.0000 | +0.00% |
| MM-TSFlib | 1.0147 | 1.0155 | -0.08% |
| GPT4MTS | 1.0360 | 1.0293 | +0.65% |
| ADAPT-Z | 1.0495 | 1.0495 | +0.00% |
| DSOF | 1.0510 | 1.0510 | +0.00% |
| On-Adapter | 1.0936 | 1.0930 | +0.06% |
| Proceed | 1.1108 | 1.1108 | +0.00% |

- **Maximum relative difference 0.65%**: neither the text repair nor the encoding fix materially changes the comparative structure among methods; the ranking is unchanged.
- **On-Adapter is not first** (mean rank 4.17); it is clearly worse on Energy (H24 ratio 2.53)
  and clearly better on SocialGood (0.86–0.94) and Traffic (0.96–0.99).
- Exploratory geometric mean over the six monthly domains: On-Adapter **0.9786** (lowest of the seven methods), consistent with the historical record of 0.9787.
- Disclosed remedial check: the timing mismatches in the Time-MMD text are **systematic** (e.g. an entry dated 2001 says "as of March 2024"),
  not 13 isolated problems. But **the results barely change after the repair**, which shows these mismatches are not the main source of the old comparative structure,
  and conversely cannot prove that the rest of the text is clean.

### 0.2 Wrap-up status

| Wrap-up step | Result |
|---|---|
| `finalize audit` | 324/324 complete, 0 failures, execution lock consistent |
| `finalize tables` | 324 text cells → 27 frozen encoding tables |
| `finalize compare` | 10,846 historical artifacts, **all unchanged, none added** |
| `score` | All 756 cells assembled; independent Float64 recomputation passed |
| `finalize manifest` | All 3,280 artifacts passed hash verification |
| `build_report` | [REPORT.md](REPORT.md) |

**The execution lock had to be re-frozen once**: `data_cache.py` was fixed once after the first freeze (adding the `text_version` attribute
to fix an `AttributeError` in the cache-loading path). It was verified that **the `source_fingerprint` of all 324 cells is identical and equal to the current code**,
after which the lock was re-frozen to match the code actually run; the reuse manifest and matrix hashes did not change. The fix touches no scientific parameter.

## 1. Environment and paths

| Item | Value |
|---|---|
| Project root | `C:\Users\18452\Documents\ChatGPT\FedNoise` |
| Baseline directory | `C:\Users\18452\Documents\ChatGPT\FedNoise\iadapter-baselines` |
| Interpreter | `iadapter-baselines\.venv\Scripts\python.exe` (Windows Python, not WSL python3) |
| How to run | `.\venv\Scripts\python.exe -m <module>` from the `iadapter-baselines` directory |
| GPU | RTX 4060 Laptop 8GB, torch 2.8.0+cu129, serial execution |
| Training-output drive | `E:\FedNoise\iadapter-baselines\runs\` |
| Cache drive | `E:\FedNoise\` |

**Note**: paths in config files must be in Windows form (`E:/...` or `C:\...`). Writing WSL paths such as `/mnt/c/...` into a config makes native Windows Python unable to find the files.

### 1.1 Usage from WSL (when the Claude Code CLI runs inside WSL)

The CLI runs in WSL, but the project uses **native Windows Python**. Calling across the two systems has one hard requirement:

**Always `cd` into the Windows project directory first, then call the venv's python.exe.**

```bash
# ✅ Correct
cd /mnt/c/Users/18452/Documents/ChatGPT/FedNoise/iadapter-baselines
./.venv/Scripts/python.exe -m benchmark_textrepair_v1.runner plan

# ❌ Wrong: calling by absolute path from a WSL cwd (e.g. /home/wuwa)
/mnt/c/Users/18452/Documents/ChatGPT/FedNoise/iadapter-baselines/.venv/Scripts/python.exe -c "import benchmark_textrepair_v1"
# → ModuleNotFoundError
```

Reason (verified): Windows Python takes `sys.path[0]` from the current directory; when the cwd is a WSL directory it sees
`\\wsl.localhost\Ubuntu\home\wuwa`, `sys.path[0]` is empty, and every package import fails.
After `cd` into `/mnt/c/...`, the Windows-side cwd shows as
`C:\Users\18452\Documents\ChatGPT\FedNoise\iadapter-baselines` and imports work.

Other WSL-side details:

- Do not run project code with WSL's `python3` (it has no torch). Use it only for plain-text/JSON processing.
- Query the GPU with the Windows binary in the repo: `./nvidia-smi.exe --query-gpu=memory.used,memory.total --format=csv,noheader`
- Run background processes with `nohup ... > log 2>&1 < /dev/null &`; they keep running as long as WSL is alive.
  Process query: `pgrep -f "benchmark_textrepair_v1.runner run"`
- Training output is on drive `E:`, which is `/mnt/e/FedNoise/...` under WSL
- Use the CLI's background/monitoring features for long tasks; long foreground `sleep` is restricted

**First instruction for the CLI (can be pasted as is)**

> Read `reports/benchmark_textrepair_v1_20261008/HANDOFF.md` (note the WSL usage section: first `cd /mnt/c/Users/18452/Documents/ChatGPT/FedNoise/iadapter-baselines`, then call `./.venv/Scripts/python.exe`).
> First check the progress of `benchmark_textrepair_v1.runner` and whether anything failed; if the controller has exited before finishing, resume with `runner run` (completed cells are skipped automatically).
> Once all 324 cells are complete, run audit / tables / compare / score / manifest / build_report in the order of HANDOFF section 5, and write the results into REPORT.md.

## 2. Directories added in this round (historical files untouched)

| Purpose | Path |
|---|---|
| 9-domain V1 data and source report | [reports/benchmark_text_repair_all9_v1_20261008/](reports/benchmark_text_repair_all9_v1_20261008/) |
| Data-building code | [benchmark_text_repair_all9_v1/](benchmark_text_repair_all9_v1/) |
| Execution compute package | [method_comparison_textrepair_v1_20261008/](method_comparison_textrepair_v1_20261008/) |
| Driver/checks/scoring for this round | [benchmark_textrepair_v1/](benchmark_textrepair_v1/) |
| Reports and frozen artifacts for this round | [reports/benchmark_textrepair_v1_20261008/](reports/benchmark_textrepair_v1_20261008/) |
| New training outputs | `E:\FedNoise\iadapter-baselines\runs\method_comparison_textrepair_v1_20261008\` |
| New cache | `E:\FedNoise\textrepair_v1_20261008_cache\` |

## 3. Completed work

### Task 1: extend the V1 rules to 9 domains
- Froze **348** source rules: [decision_rules_all9.json](benchmark_text_repair_all9_v1/source_evidence/decision_rules_all9.json)
  SHA256 `85152a0e584b9bb5717895596c26b903f497e3a1285b0c02c78f51021f948689`
  - 56 frozen rules inherited from the 4-domain pilot (SHA256 `3d0109e3dd46e5f2a312dc59e8823d9f6d556437e7b06e6246531e5a11813eda`, byte-identical)
  - 292 new rules from independent review of 5 domains (Environment / Health_US / Traffic / SocialGood / Climate)
- 9-domain V1 data: [reports/benchmark_text_repair_all9_v1_20261008/data/V1/](reports/benchmark_text_repair_all9_v1_20261008/data/V1/) (**only the `fact` column changes**)
- Masked 413 entry occurrences (1.34% of 30,944 entries); all numeric rows, forecast origins, masks and scoring targets are kept
- V1 of Agriculture/Economy/Security is **byte-identical** to the frozen three-domain pilot files
- Report: [reports/benchmark_text_repair_all9_v1_20261008/REPORT.md](reports/benchmark_text_repair_all9_v1_20261008/REPORT.md)

### Task 2: protocol freeze
- [PROTOCOL.md](reports/benchmark_textrepair_v1_20261008/PROTOCOL.md)
- [frozen_matrix_textrepair_v1.json](reports/benchmark_textrepair_v1_20261008/frozen_matrix_textrepair_v1.json) SHA256 `5ab90e412009efb6d8e3e6e7e08f97772f5fe6348d0a57aa3f38e4eaebab0e4c` (756 cells)
- [execution_lock.json](reports/benchmark_textrepair_v1_20261008/execution_lock.json) SHA256 `cf7b935f3c9e56706f2fd0a9c3cb51b1d36f6c2c95cf526ecb794dfbb39a9703` (32 entries)
- Config: [configs/method_comparison_textrepair_v1_20261008.json](configs/method_comparison_textrepair_v1_20261008.json) SHA256 `013b27a845ec582b51cc046f30f5d5bb678c03fe200ecd60699af75c6fc8cf87`
- Reuse manifest: [legacy_numeric_reuse_manifest.json](benchmark_textrepair_v1/legacy_numeric_reuse_manifest.json) SHA256 `c3256eb039a50e0be120b79474df53dcb1ea1b22966a387b16a763b9c4c64fc2`

### Task 3: checks of previously uncovered paths
- [path_checks.json](reports/benchmark_textrepair_v1_20261008/path_checks.json): **all 51 passed, 0 failed**
- Covers daily (Environment L96), weekly (Energy/Health_US L24), non-H12 windows, and GPT4MTS and MM-TSFlib, which had not been trained before
- Items: origin/target/mask/standardization alignment 13/13; forecasts do not consume future y/mask 13/13; visibility of text gather rows 13/13;
  perturbing future text does not affect past outputs 3/3 (tolerance 0); preprocessing/cache version isolation 9/9

### Proof of reuse compatibility
- [reuse_verification.json](reports/benchmark_textrepair_v1_20261008/reuse_verification.json): 432/432 cells hash-verified; across the 9 domains only the `fact` column differs
- [numeric_reuse_equivalence.json](reports/benchmark_textrepair_v1_20261008/numeric_reuse_equivalence.json): 5/5 sampled cells retrained on V1 data have predictions/targets/mask/origins **bitwise equal (tolerance 0)**

### Protection snapshot
- [protected_before.json](reports/benchmark_textrepair_v1_20261008/protected_before.json): hashes of 10,846 historical artifacts (before starting)

## 4. Run status and resuming

**The experiment is over (324/324, 0 failures), and no process is running.** To rerun or resume:

- Progress: see [controller_state.json](reports/benchmark_textrepair_v1_20261008/controller_state.json); per-cell log in [campaign.log](reports/benchmark_textrepair_v1_20261008/campaign.log)
- Order: the 324 cells run in domain → H → seed → method order, each in its own OS process
- Measured total time about 7–8 hours (Environment is slowest; its 36 cells take about half the time)

**Resuming** (if the process is interrupted, just run it again; completed cells are skipped automatically after their artifact hashes are verified):

```powershell
cd C:\Users\18452\Documents\ChatGPT\FedNoise\iadapter-baselines
.venv\Scripts\python.exe -m benchmark_textrepair_v1.runner plan     # show what remains
.venv\Scripts\python.exe -m benchmark_textrepair_v1.runner run      # continue/start
```

## 5. Wrap-up sequence after the run

```powershell
cd C:\Users\18452\Documents\ChatGPT\FedNoise\iadapter-baselines

# 1) Completion audit (check result.json and artifact hashes per cell; execution lock unchanged)
.venv\Scripts\python.exe -m benchmark_textrepair_v1.finalize audit

# 2) Record the identity of the actual encoded input tables
.venv\Scripts\python.exe -m benchmark_textrepair_v1.finalize tables

# 3) Protection comparison (were historical artifacts changed?)
.venv\Scripts\python.exe -m benchmark_textrepair_v1.finalize compare

# 4) Main statistic: assemble 756 cells + independent Float64 recomputation (including the variational CVaR cross-check)
.venv\Scripts\python.exe -m benchmark_textrepair_v1.score
.venv\Scripts\python.exe -m benchmark_textrepair_v1.score --historical   # historical reference (optional)

# 5) Artifact manifest and verification
.venv\Scripts\python.exe -m benchmark_textrepair_v1.finalize manifest

# 6) Generate the final report
.venv\Scripts\python.exe -m benchmark_textrepair_v1.build_report
```

(Optional) rebuild the V1 data/rules (idempotent, does not overwrite history):

```powershell
.venv\Scripts\python.exe -m benchmark_text_repair_all9_v1.inventory
.venv\Scripts\python.exe -m benchmark_text_repair_all9_v1.decisions
.venv\Scripts\python.exe -m benchmark_text_repair_all9_v1.build_data
.venv\Scripts\python.exe -m benchmark_text_repair_all9_v1.token_effects
.venv\Scripts\python.exe -m benchmark_text_repair_all9_v1.build_report
```

## 6. Main-statistic definition (frozen)

- Scoring space: **observed** target elements of all selected numeric variables, in the space standardized by the training-set mean/std
- MAE / MSE are scored over valid target elements
- `CVaR_q`: over the distribution of per-origin MAE of observed elements, the worst q probability mass under a uniform distribution over origins, with a fractional weight for the boundary origin;
  independently cross-checked against the variational definition (`cell_metrics` in `benchmark_textrepair_v1/score.py`)
- **Main statistic**: within each domain/H, first average the risk over seeds, then divide by the CVaR90 of the shared iTransformer backbone for the same domain/H/seed,
  and take the **equal-weight geometric mean** over the 36 domain/H tasks; also report mean/median rank, MAE, MSE, CVaR80/95 and per-domain/H results
- Pre-declared exploratory groupings: the six monthly domains, the daily/weekly/monthly frequency combinations, frequency-equal-weight sensitivity

## 7. Key scientific findings (for writing up)

1. **The timing mismatches in the Time-MMD text are systematic, not 13 isolated problems.** For example, a SocialGood entry dated 2001 reads
   "unemployment rate as of March 2024"; a Traffic entry dated 1983 reads "maximum traffic volume in 2001".
   This is exactly why V1 exists.
2. **The V1 intervention is sparse in On-Adapter's actual input**: in most domains On-Adapter's 64-token window does not contain the masked entries themselves;
   its row token changes come mainly from the trailing 1200-character window of `clean_fact` shifting after the deletion. The 256-token window of MM-TSFlib keeps some of them.
   See [repair_token_effects.csv](reports/benchmark_text_repair_all9_v1_20261008/repair_token_effects.csv).
3. **In the historical 756 cells, On-Adapter is not first on CVaR90** (historical reference in
   [historical_reference.json](reports/benchmark_textrepair_v1_20261008/historical_reference.json));
   that reference uses the old batched encoding + the original V0 text, **cannot be put into the new main table**, and is only a directional comparison.

## 8. Pitfalls already hit (to avoid repeating them)

| Symptom | Root cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'utils.rev_in'` / `'models.MLP'` | Building models from different authors' repositories one after another in the same process; the top-level module names `utils`/`models`/`layers` shadow each other | The real campaign uses **one OS process per cell**; check scripts use `checks.purge_author_modules()` |
| `FileNotFoundError: ...manifest.json.tmp...` | The cache directory path exceeds the Windows 260-character limit | Checks use a short cache root (e.g. `E:/FedNoise/tpc_tra`) |
| On-Adapter perturbation check "failed" | **Bug in the check script**: the rebuilt model was not put in `.eval()`, so dropout was in training mode | After adding `.eval()` it passes at tolerance 0; **no tolerance was loosened** |
| Control file reports "paused" | control_file in the config was written as a WSL path | Use Windows paths throughout |
| `Health.csv` vs `Health_US.csv` | `DOMAIN_FILES` in the package uses `Health.csv`; V1 data file names must match | File names in the V1 directory = `Agriculture.csv … Health.csv … Traffic.csv` |
| `benchmark_textrepair_v1` import fails | The package name is `method_comparison_textrepair_v1_20261008`; or the cwd is not the project directory (`sys.path[0]` empty) | Mind the full package name; **first `cd` into `/mnt/c/.../iadapter-baselines`, then call the venv python** |

**Rule**: do not modify any `.py` under `method_comparison_textrepair_v1_20261008/` — they are pinned by
`execution_lock.json`, and changing them makes the provenance check of completed cells fail. Changing the driver/scoring scripts
(`benchmark_textrepair_v1/*.py`) is not affected.

## 9. Limitations that must be kept

- The protocol was frozen after the old test set had been observed repeatedly; **it is not an independent confirmation or a preregistration**
- Source decisions cover only candidate entries hit by screening; all C/D entries are kept and disclosed; V1 **does not mean** "every entry is certified leak-free"
- The numeric data still use the original `end_date` availability proxy and the retrospective target-mask origin grid; this does not prove causality in the sense of real publication times and historical revisions
- The 432 cells are reused historical results under the same protocol; their text independence is shown by the code path + the numeric-equivalence rerun, but this is not the same as retraining them in this round
- iTransformer is the shared backbone and ratios are relative quantities; no claim of significance, universal superiority or SOTA may be made
- This must not be used to claim that "all historical numeric data and text have been proven strictly leak-free"

## 10. External constraints (carried over)

No paid resources, no external messages, no scheduled tasks, no stopping unrelated processes;
do not overwrite the original ZIP, historical data, historical code packages, historical results or checkpoints;
do not expand into recollecting the whole benchmark or redesigning the On method itself.

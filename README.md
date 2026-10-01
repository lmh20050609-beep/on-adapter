# iAdapter revision evidence package

Current 2026-10-01 execution: [ordered results and real progress](refine-logs/EXPERIMENT_RESULTS.md), [progress routes](refine-logs/replan_execution/PROGRESS_AND_NEXT_ROUTES.md).

Current delivery: [review decisions](refine-logs/review_delivery/REVIEW_DECISIONS.md),
[consolidated responses](refine-logs/review_delivery/REVIEW_RESPONSE.md),
[replacement manuscript text](refine-logs/review_delivery/MANUSCRIPT_REPLACEMENTS.md),
and [current coverage](docs/REVIEW_COVERAGE_CURRENT.md).
Earlier stage reports are historical; protocols and counts must remain separate.

Independent revision implementation based on the outer IG-SMoE source. Original
files and historical results are not modified. This package does not reproduce
the original ten-domain/multiple-backbone results.

Python: `/home/wuwa/miniconda3/envs/llm/bin/python`.
Working directory: `/home/wuwa/projects/research/iadapter`.

The canonical project directory is `iadapter`. Historical absolute paths in
experiment records are resolved through the checksum-verified mappings in
`docs/PATH_MIGRATION.json`; obsolete project directories are no longer required.


## Repository contents and setup

本仓库同步投稿代码、当前修订实现、实验改进、测试与review证据。最新结论见上方交付入口；TiMi启发已形成一个固定0.5B趋势表示探索；当前不称完整TiMi复现，预检误判与预测检验分开报告。

- Current executable revision: `iadapter/`; preserved submission code: `submission/code/`.
- Published aggregate results and figures: `reports/` and `refine-logs/`.
- Environment and fresh-clone instructions: [GitHub setup and sync scope](docs/GITHUB_SYNC.md), [requirements](requirements.txt).
- Large datasets, model weights, caches and complete raw runs remain local; restore inputs before running experiments or artifact-dependent tests.

## Imported submission materials

- `submission/code/`: all 215 files from the supplied code/data ZIP, including
  GPT4MTS_z, VoT_z, the data directory and original scripts.
- `submission/archives/`: unchanged original ZIP.
- `submission/documents/`: technical supplement and reproducibility checklist PDFs.
- `submission/audit/REPORT.md`: findings about baseline identities and protocol differences.
- `docs/review_routes/EXPERIMENT_PLAN.md`: five proposed revision routes.
- `submission/IMPORT_MANIFEST.json`: file-copy verification and migration record.

The verified revision implementation remains in `iadapter/`; its raw results
remain in `runs/`. Submission source is preserved separately for inspection and
selective integration. See `docs/DIRECTORY_LAYOUT.md` for paths and commands.

```bash
/home/wuwa/miniconda3/envs/llm/bin/python -m unittest discover -s tests -v
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.prepare
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.run --dataset Economy --horizon 6 --seed 2023 --variant full --run_dir runs/example
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.campaign
```

The campaign implements 190 adaptation evaluations and shares 35 backbone
checkpoints. Every prediction is emitted before mature-label updates at that
origin. Text record end dates are availability proxies, not verified publication
times. Missing targets are masked. Validation never trains adapter parameters.

See `docs/PROTOCOL.md`, `docs/DECISIONS.md`, and `refine-logs/` for details.

## Inspect the completed evidence

```bash
python -m iadapter.campaign --status
python -m iadapter.campaign --audit-completed
python -m iadapter.audit --output reports/original_csv_audit.json
python -m iadapter.analyze
python -m iadapter.write_evidence
```

Use the Python interpreter specified above. Status and campaign audit are read-only.
Running the default campaign on a completed queue audits it and exits without
retraining or changing completion timestamps, even after reporting-code commits.
Incomplete queues still require their original clean experiment revision.
Analysis must pass the original-CSV audit before replacing derived tables/figures.
The audit validates source targets, masks, splits, scalers, origin/text IDs,
maturity/feedback traces, checkpoints, configurations and metrics. It cannot
establish true text publication times or recover original-paper baseline runs.

See `docs/BACKUP.md` for the verified full backup and restore command, and
`docs/REVIEW_COVERAGE.md` for the requirement-by-requirement delivery status.

The complete method/pseudocode appendix is `reports/method_appendix.md`.
`main_pairwise_analysis.md` includes all three main-method contrasts;
`state_paired_analysis.md` gives within-group matched-seed differences, and
`failure_cases.md` preserves explicitly selected adverse cases. State-change
scores and validation thresholds are checked against the original CSVs.

## New review development sprint

`refine-logs/review_sprint/EXPERIMENT_RESULTS.md` reports the completed 42-cell
development experiment and 48 same-state text interventions. These results use
the 70%-80% development interval, not the original test table. Exact commands,
plan, code review and tracker are in the same directory. Raw artifacts are under
`experiments/review_sprint_20260929/`; the original 190-run campaign is unchanged.

## Completed continuation

[Continuation code, audits and manuscript material](refine-logs/continuation_20261001/README.md) records completed recent-record-first and locked Agriculture/Health experiments plus passed native VoT runtime integration. Current outcomes, adverse cases and remaining baseline gaps are summarized below.

## 2026-10-01 completed continuation and first-principles follow-up

N1 completed 12 new evaluations plus 12 reused controls. N3 completed all 50 evaluation cells across Agriculture/Health, with ten shared backbones and thirty warm-ups; all source-target checks pass. Native VoT passes its three-stage real-target integration smoke, while trustworthy prior generation and formal matched comparison remain unfinished.

- [Current execution tracker](refine-logs/continuation_20261001/CONTINUATION_TRACKER.md)
- [Updated manuscript replacement and separate N1/N3 tables](refine-logs/continuation_20261001/MANUSCRIPT_INTEGRATED.md)
- [Current runtime/source audit](refine-logs/continuation_20261001/RUNTIME_STATUS.json)
- [Original forecast archive inventory](refine-logs/continuation_20261001/RAW_FORECAST_ARCHIVE.json)
- [First-principles method proposal](refine-logs/innovation_20261001/FINAL_PROPOSAL.md)
- [Exploratory forecast complementarity diagnostic](refine-logs/innovation_20261001/diagnostics/RESULTS.md)

Small and variable improvements, stronger static controls and adverse outcomes are retained. The first-principles adaptive calibration route is planned; its adaptive coefficients have not yet been evaluated. Hindsight oracle values in the diagnostic are explicitly non-deployable. The full iAdapter revision goal remains in progress.

# iAdapter revision evidence package

Independent revision implementation based on the outer IG-SMoE source. Original
files and historical results are not modified. This package does not reproduce
the original ten-domain/multiple-backbone results.

Python: `/home/wuwa/miniconda3/envs/llm/bin/python`.
Working directory: `/home/wuwa/projects/research/iadapter`.

The canonical project directory is `iadapter`. Historical absolute paths in
experiment records are resolved through the checksum-verified mappings in
`docs/PATH_MIGRATION.json`; obsolete project directories are no longer required.

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

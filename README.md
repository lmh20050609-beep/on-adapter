# iAdapter revision evidence package

Independent revision implementation based on the outer IG-SMoE source. Original
files and historical results are not modified. This package does not reproduce
the original ten-domain/multiple-backbone results.

Python: `/home/wuwa/miniconda3/envs/llm/bin/python`.
Working directory: `/home/wuwa/projects/research/iadapter_revision`.

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

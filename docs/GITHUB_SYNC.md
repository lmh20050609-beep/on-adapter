# GitHub synchronization — 2026-09-30

Target: https://github.com/lmh20050609-beep/iadapter (private at synchronization).
The local revision history is retained; the canonical working repository remains
`/home/wuwa/projects/research/iadapter`.

## Included

- `submission/code/`: submitted Python implementations, model/configuration files and original scripts, preserved for provenance. Some original scripts have top-level training calls: inspect them rather than importing them into the revised executor.
- `iadapter/`: revised model definitions, causal preprocessing, frozen contextual BERT cache, origin-level delayed replay, explicit ablations, experiment queues, native GPT4MTS port and shared online executor.
- `tests/`: protocol, future-input invariance, masked targets, ablation boundaries and artifact-audit regression tests.
- `reports/`: original 190-run aggregate tables, paired results, measured costs, method appendix and mechanism figures. These are now versioned so repository links resolve after cloning.
- `refine-logs/`: development plans, source provenance, component/content/compression studies, common-OT static and online results, consolidated review responses and replacement manuscript sections.
- `requirements.txt`: pinned versions from the environment actually used; no new dependency was installed for this sync.

## Inputs and large local artifacts

Datasets under `submission/code/data/`, archives/PDF submissions, BERT/GPT2 weights,
full checkpoints, encoding caches and raw experiment directories remain local and
are excluded from Git. `data` is a relative symlink to `submission/code/data`.
Restore the five exact paired CSVs there before running dataset-dependent commands:
Economy.csv, Energy.csv, Security.csv, SocialGood.csv and Climate.csv.
Their schema/provenance is documented in `submission/audit/data-inventory.json`.
Do not substitute differently preprocessed files and claim to reproduce saved results.

Place the complete local BERT model/tokenizer at `weights/bert-base-uncased/`.
For external GPT4MTS experiments also restore `weights/gpt2/` and the pinned
`external/gpt4mts/` author source. Source commits and weight provenance are recorded
under `refine-logs/external_baselines/`. External source snapshots are not redistributed
as newly licensed project code. Requirements cover the revision package, not every
historical baseline in the submitted supplement.

Saved reports contain historical local absolute paths and source SHAs. They remain
unchanged as provenance; they are not portable references to uploaded raw artifacts.
`docs/PATH_MIGRATION.json` handles the known local directory migration only.
A fresh clone contains report-level evidence, not the full offline experiment archive.

## Setup and checks

Use Linux or WSL with Python 3.10. Install the appropriate PyTorch 2.6.0 build for
your device, then `python -m pip install -r requirements.txt`.
For the measured CUDA 12.4 environment, the PyTorch wheel index is
`https://download.pytorch.org/whl/cu124`.

CPU checks that do not require the datasets or pretrained weights:

```bash
python -m unittest discover -s tests -p test_protocol.py -v
python -m unittest discover -s tests -p test_future_context.py -v
python -m unittest discover -s tests -p test_shared_online.py -v
```

Full `python -m unittest discover -s tests -v` also includes artifact checks and
requires the matching local data/results. Do not interpret missing local artifacts
on a fresh clone as a reproduced experiment failure.

After restoring data and BERT weights, one new revision run can be launched with:

```bash
python -m iadapter.prepare --datasets Economy
python -m iadapter.run --dataset Economy --horizon 6 --seed 2023 --variant full --run_dir runs/new_economy_h6_s2023_full
```

Do not reuse an existing run directory. Larger matrices and historical audits have
additional source-commit and artifact checks; consult their recorded plans first.

## Scientific status

The completed bounded review evidence package supports mixed/negative findings,
not universal superiority. Native VoT provenance, original Table1–3 recovery,
verified text publication time and unseen-data confirmation remain unresolved.
Full-paper LaTeX integration has not been completed. TiMi was read and discussed;
its suggested text representation or separated routing changes have NOT been
implemented or tested in this repository.

Current entry point: `refine-logs/review_delivery/REVIEW_DECISIONS.md`.

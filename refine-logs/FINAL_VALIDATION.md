# Final validation — 2026-09-29

- 190/190 predefined adaptation evaluations completed; no failed cells.
- 35 unique shared backbone configurations.
- All 190 MSE/MAE values recomputed from saved predictions and original target masks.
- Matched methods and seeds share scoring streams; matched methods share their seed-specific backbone checkpoint.
- Every eligible origin consumed once, only after target maturity; pending final origins retained without flush.
- Final unittest: 11 tests passed, 1.364 seconds; full output in logs/protocol_tests.log.
- Complete 12-layer BERT preprocessing: five datasets; direct/cache checks passed.
- Six isolated efficiency cases completed: two datasets x three primary methods, 50 warm-up and 200 measured iterations.
- Main and mechanism PNG/PDF files rendered and inspected by executor.

## Same-family figure/evidence review

Reviewer: /root/experiment_code_review. Acceptance status: provisional; not a
cross-family scientific certification. Reviewer recomputed the 75 primary MSEs
and verified matched targets/masks/checkpoints. Main figure faithfully retains
Energy's adverse seed and distinguishes paired percentages from raw mean MSE.

Final review found no blocking visualization/claim issues. It verified:
Energy and SocialGood seed-2023 Fluctuation average weights 0.999969 and 0.999643;
22/25 H6 full runs with a single expert mean weight above .99; lowest mean MSE
2/5 and MAE 1/5; no-routing-text-gradient lower mean MSE in 3/3 domains; trainable
adapter/projection parameter count 6.1–6.2 times this minimal numerical backbone.
The report explicitly preserves these negative/limiting results.

## Unresolved original-paper evidence

Original GPT4MTS/VoT matched comparison, ten-domain/multiple-backbone tables,
actual publication timestamps and original LaTeX sources remain unavailable.
The delivered package does not certify those claims or include a compiled full paper.

## Follow-up audit and queue repair — 2026-09-29

- Full original/revision backup created and tar-compared before these changes (docs/BACKUP.md).
- 22 unittest cases passed, including independently changed target/mask/origin evidence and report-gate rejection.
- Independent original-CSV audit passed all 190 runs and 35 checkpoints.
- Completed campaign can be inspected after reporting commits without retraining or rewriting finished_at.
- Analysis regenerates tables only after the source audit passes; writing regenerated from those tables.
- Queue bytes and all 1,755 run/checkpoint/cache file sizes and modification times remained unchanged.
- Current coverage and unresolved-source requirements are in docs/REVIEW_COVERAGE.md.
- Full command outputs and checks are in reports/tests-review-completion.txt,
  original_csv_audit.json and review_completion_validation.json.

## Review deliverable completion pass — 2026-09-29

- 28 tests passed; new checks reject modified state scores/validation thresholds,
  missing paired seeds, duplicated method cells and mismatched valid-target counts.
- Original-CSV audit still passes all 190 runs, now including state grouping.
- Added all-main-method and within-state-group paired differences, explicit adverse cases,
  and an implementation-aligned method/algorithm appendix.
- Re-rendered and visually inspected Energy/SocialGood mechanism figures with distinct
  proxy text-arrival and mature-target markers; no clipping or legend overlap found.
- Regenerated manuscript and response drafts; raw data/runs/checkpoints/cache unchanged.
- No new experiments, model selection, external messages or training were performed.

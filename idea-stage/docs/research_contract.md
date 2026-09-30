# Research contract

Question: under fixed chronological availability and delayed supervision, does
text improve a frozen numerical forecaster, and do IG-SMoE components add value
over additive text fusion? Outcomes may be positive, negative, or inconclusive.

Evidence: 75 main H6 runs, 85 component-ablation runs, 30 H12 runs; original
dataset targets with missing-target masks; five matched seeds; measured costs.
No claim of original ADAPT-Z reproduction, true publication-time verification,
ten-domain SOTA, causal expert semantics, or universal delay robustness.
Do not select best seeds or mix unresolved historical CSVs into new results.

## 2026-09-29 development sprint

Next evidence: two-domain three-seed 2x2 routing-gradient/entropy ablation,
single-MLP control, and same-state text-content interventions. Backbone selection
uses rows 60%-70%; online candidate assessment uses 70%-80%; only the first 60%
is used for fitting the scaler and warm-up. This is exploratory development,
not an independent unseen-test confirmation. Existing 190-run evidence remains
unchanged. Details: refine-logs/review_sprint/EXPERIMENT_PLAN.md.

## 2026-09-30 ordered replan execution

The user authorized executing refine-logs/EXPERIMENT_PLAN.md in order. C1 asks
whether deployment gradient-feedback refresh has incremental value beyond
parameter updates from the same feedback-warmed full checkpoint. C2 asks whether
content provides incremental prediction benefit beyond availability metadata and
strong simple controls under the identical visible-text budget. Both may fail.
B1 uses 30 new OT evaluations and 30 exact reused references; B2 uses training-only
PCA/Ridge fits with selection-only lambda choice. Subsequent B3/B4/B5 are conditional
on the declared gates, not a requirement to run all 104 possible evaluations.
No target substitution, result-based tuning, stronger claims from failed controls,
or unseen-data/causal/publication-time assertion is authorized by completion.

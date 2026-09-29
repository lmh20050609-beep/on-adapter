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

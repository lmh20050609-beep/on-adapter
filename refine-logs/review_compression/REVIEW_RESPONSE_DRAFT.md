# Reviewer-response supplement: projection/structure trade-off

This paragraph supplements the five-seed component study with a distinct, preregistered three-seed development experiment. It is not a new held-out test or an extension of every earlier ablation to five seeds.

## ECmL F4; LCo1 W2/P4 — parameter efficiency and necessary complexity

We isolated text-projection capacity and expert structure in a 2×2 comparison: projection hidden width 96 versus 8, and the five-expert adapter versus a single MLP. The five-expert reference disables routing text-gradient conditioning and the decisiveness penalty; both structures retain expert-side text feedback and numerical feedback. All settings share the same 64-dimensional projection output, training protocol, seed-specific backbone, and compatible initial parameters. The wide controls are reused exactly, giving 24 setting cells across Energy and SocialGood at H=6 with three seeds, of which 12 are newly evaluated.

Reducing projection width decreases projection parameters from 80,160 to 6,856. However, projection-only compression increases Energy mean MSE by 2.37% relative to the wide five-expert reference. Combining compression with a single MLP reduces total trainable parameters by approximately 79.5%–79.7%, but its SocialGood mean MSE increases by 1.32% relative to that reference. Consequently, neither compressed candidate satisfies the predeclared condition of at most 1% mean MSE/MAE degradation on both domains. We retain these failures rather than loosen the threshold or search additional widths. This engineering criterion is not a statistical non-inferiority test.

The interpretation also depends on the control. Compressing the single-MLP projection alone changes Energy mean MSE from 0.016636 to 0.016978, whereas SocialGood changes from 0.190786 to 0.190794. We report both this within-structure contrast and the combined comparison, instead of attributing the entire change to projection compression. Numerical and simple-text addition results from the same three seeds are included as references.

Independent timing measurements cover the four settings in the same three rounds, with 50 warm-up and 200 measured calls per round. Cached prediction, mature-record update, and fresh-text/BERT prediction are reported separately, with round ranges and P95. Parameter savings alone do not establish latency savings; the timing table is the evidence for the latter. Repeated updates act only on isolated warm-state copies and do not change the scientific predictions. The shared-backbone, original-target/mask, feedback chronology, and source-control checks passed.

In particular, projection-only compression reduces parameters by approximately 44%, but cached-prediction median reductions are only 2.85% and 3.40%, with overlapping round ranges; the update median increases by 22.75% on Energy and 0.32% on SocialGood. We do not interpret these as stable acceleration. The combined compact single-MLP model reduces cached-prediction medians by 35.02% and 35.23%, and update medians by 35.35% and 58.21%, relative to the wide five-expert reference. Fresh-text prediction reductions are smaller, at 15.52% and 10.62%. These costs do not override the failed cross-domain accuracy threshold.

These findings support a measured complexity–accuracy trade-off and a narrower efficiency claim. They do not establish that every original component is necessary, that the compressed model is generally superior, or that current-text semantics are effectively exploited. The smallest adapter still has 34,184/33,928 trainable parameters including the text projection on Energy/SocialGood, so we do not describe it as smaller than the approximately 27k-parameter numerical backbone used here.

## Evidence and writing check

- Design and thresholds: `EXPERIMENT_PLAN.md` (fixed before runs).
- Accuracy and paired effects: `accuracy_evidence/summary.csv`, `paired_runs.csv`, `paired_summary.csv`, `decisions.csv`, `simple_reference_summary.csv`.
- Runtime and memory: `cost_evidence/REPORT.md`, `rounds.csv`, `summary.csv`; timing repetitions are not optimization seeds.
- Validation: `accuracy_evidence/validation.json`; source training commit `354090c19527ba337507f7a1397cc05cdba916da`.
- All numerical claims describe the fixed development streams. No p-values, broad significance claims, native GPT4MTS/VoT superiority, or independent-data confirmation are implied.
- This is text for later manuscript integration; no complete LaTeX manuscript has been updated or submitted by this operation.

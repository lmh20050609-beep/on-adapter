# Point-by-point response draft

This draft refers to the NEW revision evidence package. Original Tables 1–3 are not presented as reproduced. Replace R1/R2/R3 with final manuscript table numbers after integration. Both reviews' unresolved concerns remain explicit.

## Reviewer ECmL

### W1 / Feedback 1 — consistency and significance

We agree that the original blanket consistency/significance statement was stronger than the available evidence. We have removed that statement. The revised controlled evaluation reports five seeds, paired differences and both improvements and regressions. At H=6, the full model has the lowest mean MSE on 2/5 datasets and MAE on 1/5 among the three controlled alternatives. We do not have a verified paired-significance analysis supporting the original statement, and the new package makes descriptive claims only. Evidence: main_table.md, paired_differences.csv, stats-appendix.md. Manuscript: Results and experimental setup replacements.

### W2 / Feedback 2 — incompatible position/run aggregates

We agree that averaging baseline positions while reporting run variability for a selected iAdapter position is not a matched comparison. The new controlled experiments fix the backbone and tail insertion position, share each pretrained seed-specific checkpoint, and report only across-seed variability. We have not recovered matched GPT4MTS/VoT runs; therefore the old Table 2 superiority claim must be withdrawn from the verified revision rather than relabelled as validated. Evidence: config.json in each run, validation.json. Remaining limitation: external multimodal baseline comparison is unresolved.

### W3 / Feedback 3 — text-gradient counterexamples and unisolated components

We separated removal of all text gradients from removal of routing-only text gradients. The latter retains gradient-conditioned expert inputs while using unmodulated text in the gate and removing the direct textual-gradient logit term. Eight variants are evaluated on Energy and SocialGood, with an additional routing-gradient ablation on Security. The routing-gradient removal has lower mean MSE on 3/3 domains. We report paired variation and do not assume these differences are noise or claim every component is necessary. We additionally isolate the entropy term, numerical feedback, distribution descriptor, homogeneous experts and uniform fusion. Evidence: ablation_table.csv, component_verdicts.csv, per-run configs.

### W4 / Feedback 4 — implementation, chronology and efficiency

We provide fixed expert/update hyperparameters, original-context replay, explicit maturity queues and saved per-origin traces. During verification we found a historical text/numerical replay misalignment and differences between stored code variants; the new implementation corrects these issues and is evaluated independently of historical CSVs. Input preprocessing is causal under a fixed training-fitted scaler and invalid original targets are masked. Complete contextual BERT is used. We report parameter counts, cached-text and fresh-text latency, update latency, GPU allocation and replay/text cache memory. Record end_date is an availability proxy, not a verified publication timestamp. Evidence: PROTOCOL.md, DECISIONS.md, validation.json, efficiency.json, text_encoding.json.

The accompanying method_appendix.md now specifies each branch's affine dimensions, activations, initialization, gate inputs, entropy sign and origin-level pseudocode. Its spectral transform is explicitly along latent coordinates, not the time axis. The source audit additionally verifies historical-window state scores and validation-only grouping thresholds.

## Reviewer LCo1

### W1 / P3 — small and inconsistent improvements

We have narrowed the claims and report all five seeds with matched differences. The result counts are 2/5 for mean MSE and 1/5 for mean MAE in the H6 controlled comparisons. Negative outcomes remain in the tables. We do not infer statistical significance or broad generalization from small average gains. Evidence: main_table.md and stats-appendix.md.

### W2 / P4 — individual components

The revised ablation matrix separates text, all text gradients, routing-only text gradients, numerical gradients, the distribution descriptor, the decisiveness regularizer, expert heterogeneity and learned fusion. Homogeneous MLP experts approximately match the heterogeneous expert parameter budget, with the exact difference recorded per run. Effective expert count is a diagnostic of soft weights, not a reduction in evaluated expert count. Evidence: ablation_table.csv, component_verdicts.csv and expert_matching fields in metrics.json.

### W3 / P1 / P2 — replay and controlled protocols

Each method uses the same forecast-origin timeline and valid target set. Forecasts precede updates; only fully matured origins are consumed, exactly once. Replays retain their original text IDs and both feedback snapshots, and newly computed feedback affects only later forecasts. The numerical backbone, insertion location and seed-specific checkpoint are shared across the three main alternatives. This controls the new internal comparisons. GPT4MTS/VoT were not re-evaluated in this local package, so their original comparison remains unresolved and is not used to claim superiority. Evidence: updates.jsonl, prediction_context.json, validation.json.

### W4 — useful textual signals and routing under shifts

We add descriptive low/medium/high state-change analyses using historical-window scores and thresholds selected from validation data, plus full-stream Energy/SocialGood routing figures with fixed seed 2023. These figures align first-step forecasts, historical state-change scores, text availability and expert weights. They do not establish semantic causality or prove that any particular type of text is universally useful. Finer text-content interventions remain outside this limited revision package. Evidence: state_groups.csv, figure-catalog.md, mechanism figures.

For each state group we additionally report all three matched-seed contrasts, including additive text versus numerical adaptation, with paired-difference standard deviations and the number of improving seeds (state_paired_analysis.md). Across all H6 origins, additive text improves mean MSE on 0/5 domains and mean MAE on 1/5 (main_pairwise_analysis.md). We therefore do not assume that access to the supplied text is beneficial by itself.

The diagnostics also reveal strong routing concentration: 22 of 25 full-model H6 runs assign more than 99% mean weight to one expert. We therefore do not use these visualizations as evidence of adaptive temporal specialization. This is a limitation of the measured configuration, not a successful mechanism claim.

### P5 — datasets, horizons and table organization

The revised main table covers Economy, Energy, Security, SocialGood and Climate at H=6, with five seeds and a shared protocol. The H=12 extension covers Economy and Energy. Core ablations cover Energy and SocialGood at H=6, plus the routing-text-gradient ablation on Security. All tables explicitly name scope and aggregation. This reduced, traceable evaluation replaces broad unsupported claims; it does not reconstruct the old ten-domain evaluation. Evidence: manifest.json, summary.csv and experiment setup replacement.

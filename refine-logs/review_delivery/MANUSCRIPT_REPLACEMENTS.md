# Current manuscript replacement sections

Integration draft, not a compiled revised manuscript. This text supersedes the empirical claims in earlier stage-specific drafts while keeping their data available. Preserve separate table blocks and full negative results.

## Abstract

We study text-assisted adaptation of a frozen time-series forecaster when multi-step targets become available with delay. The revised implementation records origin-specific numerical, textual and feedback context, emits predictions before updates, and replays each mature origin once. A soft mixture of heterogeneous residual experts is compared with numerical and additive-text controls. Our evaluation includes five datasets, five optimization seeds, component ablations, text-content interventions and measured online costs. Benefits are limited and depend on the dataset and metric: the main revised H=6 evaluation yields the lowest mean MSE on two of five datasets and the lowest mean MAE on one. Additional common-target comparisons retain both static external models and explicitly defined online extensions. They do not establish general superiority or the necessity of every component. Text availability is approximated by dataset record end dates; verified publication-time provenance remains unavailable.

## Method and reproducibility

Use ../../reports/method_appendix.md for the full expert equations, gate, descriptor and origin-level algorithm, and ../../docs/PROTOCOL.md for the fixed original revision setup. The measured backbone is a disclosed minimal ITransformerZ with two layers and dimension 32. It remains frozen during adapter updates. At each origin t, a forecast for the following H observations is saved before any update at t. Each origin s is consumed once when s+H≤t, using its saved input, text identifiers and feedback snapshots; the newly generated feedback only affects later forecasts. Final pending origins are not flushed. Training-only normalization, historical forward fill and original-target validity masks are shared by the controls. BERT-base is frozen and contextual: 12 layers, attention masks and masked mean pooling over at most 64 tokens for the internal methods. Frozen representations are cached, while their projection remains trainable. Missing-text representations are zeroed after projection. The availability indicator is binary.

## Experiments: two protocols, separate claims

The original revised online evaluation uses chronological 60/20/20 splits, five seeds (2023–2027), Economy, Energy, Security, SocialGood and Climate at H=6, with H=12 restricted to Economy and Energy. Internal methods share each seed-specific backbone and second-layer tail insertion. Backbone checkpoint selection uses the validation split, while adapter warm-up uses only the training stream. The full original matrix consists of 190 adapter evaluations and 35 shared backbone configurations.

Subsequent exploratory studies use the first 60% for training, 60%–70% for checkpoint selection and 70%–80% for development assessment. These studies are not independent confirmation and do not overwrite the earlier evaluation. The external comparison predicts OT only at H=6, with lookback 48 for Energy and 36 otherwise. It evaluates numerical adaptation, iAdapter and a pinned author-model GPT4MTS port in both static and online modes. Static evaluation keeps weights fixed; internal feedback is zero. Online evaluation starts from the identical static parameters with empty optimizer, feedback and maturity queue. A common executor predicts first and updates each mature target once. The fixed online optimizer is Adam (lr 0.001, weight decay 0.0001, clip norm 1, batch 1). GPT4MTS retains its permitted trainable parameters but has no gradient-feedback cache. This is an explicit extension, not the author's original online procedure.

GPT4MTS uses historical CLS representations with token limit 512, whereas iAdapter uses current selected mean-pooled representations with token limit 64. Architecture, text consumption and training differ, despite shared OT targets, origins and availability cutoffs. Thus the comparison estimates complete-system performance under a common task; it does not isolate MoE or text-budget effects. End dates remain availability proxies in every block.

## Results: retain strong controls and adverse cases

In the original revised H6 comparison, iAdapter has the lowest mean MSE on 2/5 domains and MAE on 1/5. Additive text improves mean MSE over numerical adaptation on 0/5 domains and MAE on 1/5. In the separate OT development study, static iAdapter has the lowest mean MSE on 0/5 domains and MAE on 1/5. Online iAdapter improves its own static mean MSE and MAE on Economy, Energy and SocialGood but worsens both on Security and Climate. Parameter updates and feedback refresh change together, so this contrast does not identify feedback-cache causality.

Against numerical online adaptation, online iAdapter has lower mean MSE only on Security (0.095% relative reduction) and Climate (0.242%); mean MAE is higher on all five domains. The fixed GPT4MTS online extension increases both errors in all 25 runs relative to its static form. We retain static GPT4MTS beside that extension: against the static model, online iAdapter lowers mean MSE only on Climate and mean MAE on Security and Climate. These findings do not support a general superiority claim or a conclusion about the intrinsic failure of GPT4MTS. All tables provide five-seed sample SD and paired differences without significance claims.

## Components, text and cost

Removing routing text gradients lowers mean MSE in the three evaluated original ablation domains. The development 2×2 routing-gradient/entropy experiment isolates the two switches over three seeds. Extending no entropy to five seeds reduces mean MSE relative to full by 5.74% on Energy and 1.03% on SocialGood, but the two added Energy seeds worsen on both metrics. A single-MLP alternative reduces parameters and measured latency compared with full but provides little additional accuracy benefit over additive text. Projection compression fails the predeclared two-domain accuracy/parameter gate. These outcomes do not support universal component necessity or selection of a new default from the observed data.

Current-content interventions at a fixed state do not show a consistent advantage for correct text over the specified substitutions. Because historical feedback remains fixed, these probes do not exclude information in prior text. Strongly concentrated routing in the original H6 full runs (22/25 with one expert above 99% mean weight) limits interpretations of dynamic semantic specialization. Historical-window state-change groups and aligned routing plots are descriptive rather than causal or independently validated drift detection.

Use the original efficiency table and the later isolated three-round cost table separately. They report trainable and frozen parameters, fresh-text encoding, cached prediction, delayed update, medians/P95, allocated GPU memory and logical CPU/cache payloads on an RTX 4060 Laptop. Full adapter-plus-projection parameters are approximately six times the small numerical backbone, so relative parameter efficiency is not established. Cache-path latency is not equivalent to end-to-end deployment cost.

## Limitations and claim changes

Remove “consistently and significantly,” universal SOTA, universal component necessity, validated expert semantics, verified real publication-time causality and arbitrary-delay robustness. Do not re-label the submitted GPT4MTS_z or VoT_z proxies as native methods. Do not restore original Tables 1–3 without matching source records. The native VoT prior-generation and origin-availability chain is unresolved; this is a provenance gap, not proof of leakage. New data or an untouched period would be needed for independent confirmation of any selected improvement. The present evidence answers the revised research questions with mixed and negative results; it does not establish submission readiness or acceptance.

## Table and figure integration map

| Block | Authoritative artifact | Scope |
|---|---|---|
| R1 main and H12 | ../../reports/main_table.tex; summary.csv | Original revised chronological evaluation |
| R1 ablation | ../../reports/ablation_table.tex; component_verdicts.csv | Two domains plus Security targeted check |
| R1 cost | ../../reports/efficiency_table.tex; efficiency.json | Economy/Energy, original methods |
| R1 mechanisms | ../../reports/figures/; figure-catalog.md | Full-stream plots and descriptive grouping |
| R2 components/content | ../review_sprint/five_seed_evidence/; REVIEW_UPDATE_FIVE_SEED.md | Development; preserve three/five-seed distinction |
| R3 compression | ../review_compression/REVIEW_FINDINGS.md | Failed engineering gate; no expanded search |
| R4 all external modes | ../external_baselines/online_evidence/ALL_CONTROLS.md | OT H6, 75 static plus 75 online cells |

# Additional reviewer-response draft: development experiments

This text describes new exploratory development evidence, not a recovered original table or an independent held-out confirmation. All statements below are conditional on Energy/SocialGood, H=6 and three seeds. The chronological train/checkpoint-selection/development intervals are 60%/10%/10%; the remaining old test interval is not evaluated here.

## ECmL W3 / Feedback 3; LCo1 W2 / P4 — isolated components

We added a matched 2×2 study of routing text-gradient conditioning and the decisiveness entropy penalty, sharing each seed-specific backbone and the initial common parameters. We also evaluated a single-MLP adapter. Removing entropy reduced mean MSE and MAE on both development domains, with MSE reductions in 6/6 matched runs and MAE reductions in 5/6. Removing routing text gradients alone increased mean MSE on both domains, whereas removing both terms reduced mean MSE relative to the full model. We report the paired differences and interaction rather than assume independent or universally beneficial component effects.

Energy: full MSE 0.018823; no-entropy 0.016988; no-routing-text-gradient 0.020711; both removed 0.016856. SocialGood: full MSE 0.190055; no-entropy 0.187889; no-routing-text-gradient 0.192794; both removed 0.188309.

These observations differ from the earlier routing-gradient ablation on the old test interval. We retain both outcomes and do not attribute that difference to a single cause without a controlled follow-up. The single-MLP model reduces trainable parameters but changes both structure and capacity, so it does not establish an isolated causal role for each expert. Evidence: main_summary.csv, paired_runs.csv, paired_summary.csv, factorial_interaction.csv.

## LCo1 W4 — usefulness of textual content

At each development forecast origin, we held model weights, numerical input and historical feedback fixed and replaced the current text representation with a training-prefix mean, a token-length-matched earlier text, or a zero representation. Diagnostic forwards do not update the reference model or alter its subsequent trajectory. The correct-text and replacement predictions use identical original target masks. All donor records precede their query origins and none has identical cleaned text in this experiment.

Replacing content with a mean or historical text produces small prediction changes and does not consistently worsen error. Zeroing the current representation has a larger, domain-dependent effect, but this also changes availability and is not sufficient evidence of semantic utilization. We therefore do not present branch activity or a routing plot as proof that the model extracts useful current-text semantics. Historical text feedback is held fixed, so this intervention is not a removal of all past textual information. Evidence: text_probe_runs.csv, text_probe_summary.csv, donor_content_audit.csv and saved same-origin intervention predictions.

## ECmL W1 / Feedback 1; LCo1 W1 / P3 — strength of gains

We report all three seeds and sample standard deviations of matched differences. These repeats measure optimization variability conditional on the fixed development streams; overlapping windows are not independent repeats. We make no statistical-significance or broad SOTA claim. Engineering candidates selected from these results still require independent confirmation. Original GPT4MTS/VoT comparisons and true publication-time provenance remain unresolved.

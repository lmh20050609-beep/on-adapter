# Consolidated reviewer-response draft

Status: local response draft for manuscript integration; no response has been submitted and no complete manuscript has been edited. Evidence scope is a revised implementation plus explicitly separated development studies, not recovery of the original submitted tables. A same-family result-to-claim assessment is recorded separately and is provisional.

## ECmL — Feedback 1: inconsistent gains and significance

We agree that the phrase “consistently and significantly” is not justified. We withdraw that claim from the proposed revision. The original Table 1–3 results have not been fully traced to matching configurations and predictions, so we do not present the new experiments as a reconstruction of those tables. In the revised five-domain H=6 controlled online evaluation, iAdapter has the lowest mean MSE on 2/5 domains and mean MAE on 1/5 among numerical additive adaptation, additive text fusion and the full adapter. Additive text fusion itself reduces mean MSE on 0/5 domains and mean MAE on 1/5 relative to numerical adaptation. All five seeds and negative outcomes are retained.

A separate common-OT development study now includes five seeds of three static systems and their explicitly defined online counterparts. In that online comparison, iAdapter has lower mean MSE than the numerical online adapter only on Security and Climate, by approximately 0.095% and 0.242%, and higher mean MAE on all five domains. Its mean MSE is higher on Economy, Energy and SocialGood by approximately 14.10%, 42.85% and 1.65%. These differences cannot be described as broadly favorable. We report sample SD over seeds and matched-seed difference SD. We have not conducted paired significance tests and make no statistical significance claim. Seeds quantify optimization variability on fixed time streams; overlapping forecast targets are not independent repetitions.

Evidence: ../../reports/main_pairwise_analysis.md; ../external_baselines/online_evidence/ALL_CONTROLS.md and all_controls_paired_runs.csv.

## ECmL — Feedback 2: Table 2 position and uncertainty mismatch

The reviewer is correct: averaging baseline insertion positions and selecting the iAdapter tail do not define comparable aggregates. We cannot retrospectively determine each original Table 2 error bar without the matching logs. The proposed revision withdraws the unverified superiority statement and replaces it with separately labeled new evidence. All internal adapters use the same fixed second-layer tail and seed-specific backbone. No insertion-position average is mixed with seed variation; all reported SDs refer to optimization seeds.

The author GPT4MTS architecture is evaluated as a separate system, using pinned public source and pretrained weights. It has no analogous adapter insertion-position search. The common-OT table shares forecast origins, targets, masks, chronological boundaries and availability cutoffs. It does not equalize architecture, trainable parameter count, text encoding/history or training schedule across systems. GPT4MTS uses historical CLS/512 text representations; iAdapter uses selected pooled mean/64 representations. Consequently this is a common-task system comparison, not an isolated MoE or equal-text-budget intervention. VoT is omitted because the supplied generated priors lack a recoverable origin/availability provenance chain. Missing provenance is not evidence that leakage actually occurred.

Evidence: ../external_baselines/STATIC_MATRIX_FINDINGS.md; ../external_baselines/INTEGRATION_REPORT.md; ../external_baselines/vot_prior_audit.json; ../../submission/audit/REPORT.md.

## ECmL — Feedback 3: text-gradient ablation and decisiveness

We do not have evidence that every component is necessary. In the revised original online matrix, removing routing text-gradient conditioning lowers mean MSE on Energy, Security and SocialGood. All paired differences and seed variability are supplied rather than explaining these outcomes away as noise. A development study additionally isolates routing text gradients and the entropy decisiveness term in a 2×2 design over three seeds on Energy and SocialGood. The no-entropy candidate was extended to five seeds: relative to full iAdapter, mean MSE decreases by about 5.74% and 1.03% in the two domains, with improvements in 3/5 and 5/5 seeds. In the two newly added Energy seeds, both MSE and MAE worsen. This reversal limits the original three-seed observation.

The original key ablations also disable all text feedback, numerical feedback, distribution descriptors, entropy, expert heterogeneity and learned fusion under explicit path definitions. A single-MLP control tests a broader architectural simplification, not the isolated effect of removing a particular expert. Individual expert semantics and universal component necessity remain unproven. Concentrated routing (22/25 full-model H6 runs assigning over 99% mean weight to one expert) motivates this more cautious interpretation. We do not infer the cause of the gradient-ablation result from that correlation alone.

Evidence: ../../reports/component_verdicts.csv; ../../reports/method_appendix.md; ../review_sprint/REVIEW_UPDATE_FIVE_SEED.md; ../review_sprint/five_seed_evidence/.

## ECmL — Feedback 4: reproducibility and efficiency

The replacement method appendix specifies expert computations, gate inputs, the distribution descriptor, regularization, feedback updates and ablation boundaries. The revised numerical backbone is the minimal ITransformerZ implementation, dimension 32 and two layers, not an asserted reproduction of an undisclosed original configuration. The internal text representation uses the full frozen 12-layer BERT-base with attention masks and masked mean pooling. Cached frozen features are passed through the current trainable projection during replay. Missing-text outputs are zeroed after projection using a binary availability indicator.

Numerical inputs are chronologically forward-filled, with a fixed leading-missing fallback; normalization is fitted on the training prefix only. Original missing targets are masked in both training and evaluation. Selected text IDs are saved per origin. Text availability is approximated by dataset end_date, so we cannot claim verified real publication-time causality. The adapter is warmed on the training stream; evaluation starts with empty optimizer, queue and feedback states and the retained warm parameters. Validation selects checkpoints without carrying validation-adapted weights into evaluation.

On the RTX 4060 Laptop GPU, the original Economy/Energy full configurations have 166,896/167,088 trainable adapter-plus-projection parameters, versus 26,854/27,238 numerical-backbone parameters. Cached prediction medians are 3.239/3.561 ms and update medians 9.163/9.889 ms; fresh-text prediction medians including BERT are 10.810/10.632 ms. These isolated measurements use 50 warm-ups and 200 measured iterations; P95, peak allocated GPU memory and CPU/cache payloads are supplied separately. They exclude some executor/logging overhead and are not full deployment throughput. Small absolute parameter count does not justify calling the adapter parameter-light relative to this backbone. Later single-MLP simplification reduces cost but does not consistently outperform the simpler text-addition control.

Evidence: ../../reports/efficiency.md and efficiency.json; ../../reports/text_encoding.json; ../../reports/method_appendix.md; ../review_sprint/cost_evidence/REPORT.md. Hardware and parameter conventions must replace the unverified older deployment description.

## LCo1 — P1: replay and delayed updates

For an origin s, the target is the next H observations and becomes fully mature at s+H. At each current origin t, the system first emits and saves its prediction, then consumes each previously unused record with s+H≤t. Each pending record retains the original input/text IDs and original detached numerical/textual feedback context. Replay uses that context with current trainable parameters. Any new feedback affects only later forecasts; scores always use the original emitted predictions. The last H unconsumed records are not flushed after scoring ends.

The new online common-OT comparison uses one event loop and maturity queue for all three systems. Internal adapters preserve their gradient-feedback mechanism; the GPT4MTS extension updates only its declared trainable parameter set and has no gradient-feedback cache. This aligns target maturity and update opportunity, not every method's computation. All 25 dataset/seed groups passed original-CSV target/mask and once-only maturity checks. The first seven forecasts at H=6 match their same-weight static counterparts before updates can influence predictions.

Evidence: ../external_baselines/ONLINE_EXTENSION_PLAN.md; ../../iadapter/shared_online.py; ../external_baselines/online_evidence/validation.json; protocol-tests.txt.

## LCo1 — P2: external methods and protocol alignment

We distinguish author-model GPT4MTS, our fixed online extension, and the submitted GPT4MTS_z proxy. The extension starts from exactly the static selected parameters and uses Adam with learning rate 0.001, weight decay 0.0001, gradient clipping at one and batch one on mature targets. This is our declared evaluation extension, not the author's original online algorithm. Text history and encoding differences remain explicit, as described above. We have not evaluated native VoT under a trustworthy provenance chain and cannot claim all three original methods are now fully matched.

Critically, this fixed GPT4MTS online policy increases both MSE and MAE in all 25 runs relative to its own static reference. The cause has not been isolated. We retain the stronger static GPT4MTS results beside the extension and do not use the degraded extension to claim superiority over GPT4MTS. Relative to static GPT4MTS, online iAdapter has lower mean MSE only on Climate and lower mean MAE on Security and Climate. No post-result learning-rate search was conducted.

Evidence: ../external_baselines/online_evidence/ALL_CONTROLS.md; all_static_online_runs.csv; all_controls_paired_summary.csv in the same directory.

## LCo1 — P3: reliability of small effects

All seed-level outcomes, mean errors, sample SD and paired differences are included. The approximately 0.095%/0.242% common-OT MSE improvements over numerical online adaptation coexist with worse mean MAE in those domains. Additional optimization seeds did not provide independent-data confirmation; in Energy they reversed some early component findings. We therefore label the subsequent 70%–80% studies as exploratory development. We do not select the best seed, pool heterogeneous protocols into one win rate, or claim significance or non-inferiority. Independent new periods/data would be required before asserting a robust new superiority claim.

## LCo1 — P4: experts, descriptors, gradients and regularizer

The revised ablation table includes a five-independent-MLP homogeneous control with parameter counts, uniform fusion, no numerical gradient, no distribution descriptor, no text, no text gradients, no routing text gradients, and no entropy. These changes are defined by computational paths in the method appendix. A later single-MLP and projection-width study examines simplification and cost; neither is presented as isolating the semantic role of each original expert. No compressed candidate meets the predeclared accuracy/parameter criteria in both domains, so that branch was stopped rather than tuning widths until it passed. Effective expert count is a diagnostic of the gate distribution, not an independently trained component; we describe its computation and avoid treating it as a separately validated mechanism.

The current-content interventions compare correct text, training-mean text, historical donor text and zero text at the same model state while retaining history. They do not establish a stable content advantage; the full model's mean prediction changes for mean/donor substitutions are approximately 6.4e-5–9.2e-5 in standardized output units. This does not establish that the original text contains no useful information, because prior feedback and training history remain fixed. Accordingly no claims about useful text types or causal semantic expert specialization are made. State-change plots are descriptive historical-window analyses, not validated concept-drift detection.

Evidence: ../review_sprint/REVIEW_UPDATE_FIVE_SEED.md; ../review_compression/REVIEW_FINDINGS.md; ../../reports/state_paired_analysis.md; ../../reports/figure-catalog.md.

## LCo1 — P5: exact data, horizons and table organization

The proposed revision separates four evidence blocks. R1: the original revised five-domain H6 three-method comparison (75 runs), two-domain key ablations plus the Security routing-gradient check (85), and Economy/Energy H12 (30), totaling 190. These use the original revised 60/20/20 chronological protocol. R2: Energy/SocialGood H6 development component/content studies, with 60% training, 60%–70% checkpoint selection and 70%–80% evaluation; the 2×2 study has three seeds, selected candidates five. R3: the same two-domain three-seed projection compression study. R4: five-domain OT-only H6 external static/online comparison with five seeds under the development split, 75 static and 75 online cells. Reused controls are explicitly identified and not counted as new runs. All reports specify numerical target channels; these blocks must not be pooled.

No ten-domain, multiple-backbone, multiple-encoder or arbitrary-delay robustness claim is inferred from this evidence. The original broad-table records, native VoT generation provenance, true text publication-time records and independent new confirmation data remain unresolved. The present delivery contains replacement paragraphs and table/figure sources because the full LaTeX manuscript was not supplied.

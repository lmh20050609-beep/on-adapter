# iAdapter 修订整合稿：实验、讨论及摘要替换

这是基于真实结果的章节替换稿，尚未合入原 LaTeX，也未加入待跑的 N1/N3 数字。原始投稿宽范围表格不会与新实验拼接。本稿保留原 IG-SMoE 作为被验证的方法，同时将低维文本接入作为独立的后续研究。

## Compact outline

- Define delayed, origin-specific adaptation of a frozen numerical forecaster.
- Specify the executable expert, gate, text and replay protocol.
- Report the five-domain chronological evaluation with matched seeds.
- Separate component evidence, complete-system baselines and development diagnostics.
- Present the restricted five-dimensional text gains with all stronger controls.
- State remaining provenance and confirmation gaps; reserve additional-domain results.

## Abstract replacement — problem, design and evidence

We study text-assisted online adaptation of a frozen time-series forecaster under delayed multi-step supervision. iAdapter applies a residual correction to an intermediate representation through a soft mixture of heterogeneous experts. Our revised implementation records the input, selected text and feedback at each forecast origin, emits predictions before updates, and consumes each mature target once. A controlled evaluation on five datasets and five optimization seeds quantifies dataset-dependent benefits, component contributions and online costs. The full model attains the lowest mean MSE on two of five datasets, while reduced variants often improve its error. A separate development study explores whether compact text representations improve simple additive adaptation. With the same five-dimensional interface and initialization, a fixed language-model representation reduces mean MSE relative to BERT-PCA features by 3.14% on Energy and 1.54% on SocialGood; it also improves SocialGood over numerical online adaptation by 2.34%. We retain stronger numerical and static multimodal controls, adverse cases and seed variability. Text record end dates approximate availability, and true publication-time provenance remains unverified. The results support conditional progress in text-assisted adaptation and a reproducible delayed-update protocol.

## Method clarification — interface and reproducibility

At origin t, the numerical input consists of the preceding L observed rows, including t. A frozen minimal ITransformerZ produces the latent representation Z_t. The adapter generates a residual R_t and the original frozen head maps Z_t+R_t to predictions for the next H observations. All internal controls use the second-layer tail, d_model=32 and two layers. Numerical inputs are forward-filled from history, with zero as the fixed leading-missing fallback; normalization uses the training prefix. Original missing targets are excluded from loss and scoring. This implementation is disclosed rather than asserted to reproduce an unspecified original configuration.

Text records from the current and preceding merged rows are cleaned and combined using the documented selection rule. Their IDs are saved. Internal features use a frozen 12-layer BERT-base, attention masks, a maximum of 64 tokens and masked mean pooling, including unmasked special tokens. Features are cached on CPU/disk. Replay applies the current trainable projection to the saved frozen feature; it does not reuse an obsolete projection output. Missing-text outputs are multiplied by a binary availability mask after projection. End_date is an availability proxy, not verified publication time.

The full model evaluates all five experts at each origin. Their implemented branches are a zero-initialized trainable linear correction, a rank-two output bottleneck, spectral filtering over latent coordinates, latent centering with numerical feedback, and text-conditioned affine modulation. These are architectural biases; spectral and fluctuation names do not establish seasonal decomposition or temporal high-pass filtering. The gate uses local and variable-averaged latent states, numerical feedback, text, textual feedback and a historical-window distribution descriptor. Softmax temperature is one. Effective expert count is 1/Σw_k² and measures gate concentration; it does not make execution sparse.

The loss is masked MSE plus 0.02 times mean gate entropy in the full model. Minimizing this term encourages concentration, rather than load balancing. Adam uses lr=0.001, weight decay=0.0001 and clipping at one, with online batch one. Numerical feedback uses EMA=0.9; textual feedback is the latest mature-origin loss gradient. The exact branch equations, dimensions, gate terms and ablation paths are provided in the implementation appendix (`reports/method_appendix.md`).

For a saved origin s, the complete H-step target matures at s+H. At every current origin t the executor first saves the issued forecast, then consumes any unused mature record. Each record retains its original numerical context, text IDs and detached feedback snapshots. Replay uses those snapshots with current trainable parameters. Newly computed feedback only affects later forecasts. For H=6, the first test-origin update cannot influence a prediction until the eighth forecast. The final H pending records are left unconsumed when scoring ends. Metrics use only the initially issued predictions.

Backbone training uses the training prefix and checkpoint selection uses validation. The adapter is warmed once on the training stream. Evaluation retains warmed parameters but clears optimizer, feedback and queue state. Validation-adapted parameters are not carried into evaluation. The backbone remains frozen during warm-up and evaluation. Saved forecasts are checked against original CSV targets and masks, and each mature origin is consumed once.

## Experimental organization — distinct evidence blocks

The R1 revision evaluates Economy, Energy, Security, SocialGood and Climate under chronological 60/20/20 splits. It uses seeds 2023–2027, H=6, L=48 on Energy and L=36 elsewhere. Its main comparison contains 75 evaluations; eight additional variants on Energy/SocialGood plus the Security routing-gradient check add 85; Economy/Energy H=12 adds 30. The complete 190 evaluations reuse 35 seed/dataset/horizon backbone configurations. All numerical channels selected by the disclosed loader are predicted. MSE and MAE are computed on the same observed target elements within each comparison.

Later OT-only studies use 60% training, 60–70% checkpoint or predeclared readout selection and 70–80% development assessment. These periods have been inspected during development. Their three- and five-seed studies are separate from R1 and from each other. They do not provide untouched-data confirmation. The prospective Agriculture/Health extension uses 60/20/20 with a candidate locked before its test evaluation; it has no forecast results yet.

Five optimization seeds describe variability conditional on a fixed chronological stream. Overlapping forecast windows are not treated as independent samples. Tables retain all seeds, sample SD and paired seed differences. Relative changes are ratios of dataset-specific mean errors unless otherwise specified. No significance test or cross-domain raw-error pooling is used. A gain below 2% remains an observed improvement; the earlier engineering screen is not a statistical threshold.

## R1 results — matched five-domain evaluation

Table 1. Revised chronological 60/20/20 evaluation, five seeds. Lower MSE/MAE is better. These are all-channel errors under R1, not the OT-only development scores below.

| Dataset | H | Method | MSE (mean ± SD) | MAE (mean ± SD) | n |
|---|---|---|---|---|---|
| Economy | 6 | Numeric adapter | 0.301191 ± 0.017567 | 0.407477 ± 0.011565 | 5 |
| Economy | 6 | Text addition | 0.303267 ± 0.017749 | 0.408810 ± 0.012572 | 5 |
| Economy | 6 | iAdapter | 0.302811 ± 0.020286 | 0.408696 ± 0.012974 | 5 |
| Energy | 6 | Numeric adapter | 0.094009 ± 0.007192 | 0.208535 ± 0.004241 | 5 |
| Energy | 6 | Text addition | 0.097347 ± 0.007069 | 0.212689 ± 0.006381 | 5 |
| Energy | 6 | iAdapter | 0.097389 ± 0.046207 | 0.218451 ± 0.052828 | 5 |
| Security | 6 | Numeric adapter | 91.515101 ± 2.035356 | 4.893063 ± 0.086249 | 5 |
| Security | 6 | Text addition | 92.664365 ± 1.535644 | 4.957522 ± 0.051339 | 5 |
| Security | 6 | iAdapter | 91.216356 ± 3.076448 | 4.826188 ± 0.312196 | 5 |
| SocialGood | 6 | Numeric adapter | 0.866542 ± 0.030956 | 0.410124 ± 0.020556 | 5 |
| SocialGood | 6 | Text addition | 0.870328 ± 0.025706 | 0.410298 ± 0.015874 | 5 |
| SocialGood | 6 | iAdapter | 0.871040 ± 0.028510 | 0.416145 ± 0.016762 | 5 |
| Climate | 6 | Numeric adapter | 1.129030 ± 0.018063 | 0.867883 ± 0.008703 | 5 |
| Climate | 6 | Text addition | 1.130308 ± 0.017720 | 0.867712 ± 0.008753 | 5 |
| Climate | 6 | iAdapter | 1.128200 ± 0.017355 | 0.868359 ± 0.007927 | 5 |
| Economy | 12 | Numeric adapter | 0.451303 ± 0.019451 | 0.509476 ± 0.012557 | 5 |
| Economy | 12 | Text addition | 0.463131 ± 0.019361 | 0.515615 ± 0.010509 | 5 |
| Economy | 12 | iAdapter | 0.458627 ± 0.021881 | 0.512689 ± 0.011770 | 5 |
| Energy | 12 | Numeric adapter | 0.197950 ± 0.030876 | 0.325344 ± 0.020402 | 5 |
| Energy | 12 | Text addition | 0.207061 ± 0.022130 | 0.334630 ± 0.016791 | 5 |
| Energy | 12 | iAdapter | 0.220617 ± 0.056771 | 0.343069 ± 0.031149 | 5 |

SD is across five seeds on a fixed chronological test stream.

The full model has lower mean H=6 MSE on Security and Climate and lower mean MAE on Security. The gains coexist with adverse mean errors on the remaining domains. Energy illustrates why means and individual seeds should both be shown: full beats numerical adaptation in four of five seeds, but one large adverse outcome makes its overall mean MSE worse. At H=12, full improves over simple text addition on Economy, while numerical adaptation remains stronger in mean error. These outcomes support dataset- and comparator-specific improvements, not universal robustness across horizons.

## Components — identify contributions instead of presuming necessity

The revised ablations separately remove text, expert/routing text gradients, numerical gradients, the descriptor and entropy, and compare homogeneous experts and uniform fusion. Every name denotes a specific computational path. Removing routing text gradients lowers mean MSE in the three evaluated original ablation domains; paired variability and adverse seeds remain in the table. Homogeneous experts can also improve mean error. Thus neither heterogeneity nor each feedback path is established as universally necessary. Individual expert semantics remain unverified because leave-one-expert-out experiments were not run.

A same-start deployment study separates parameter updates U from feedback refresh F. Joint online U1F1 improves the full model over its own warmed static state by 28.71% MSE on Energy and 2.53% on SocialGood. These are useful conditional adaptation gains. U1F1 nevertheless has 1.58% and 0.47% higher mean MSE than U1F0. Feedback-only deployment also raises average error. The contrast indicates that useful parameter adaptation does not require a positive average contribution from deployment feedback refresh in these settings. It does not remove feedback used during earlier warm-up.

Gate and historical-window analyses are reported as diagnostics. In 22/25 original full H=6 runs, one expert exceeds 99% mean weight. This concentration limits claims of dynamic semantic specialization. Aligned forecast, historical state-change and gate plots include the full predefined stream, text-selection proxy and target maturity. Their state-change score is not validated concept-drift ground truth, and a correlation is not a mechanism proof.

## Complete-system multimodal baselines

The separate five-domain OT comparison retains both static author-model GPT4MTS and its explicitly defined online extension. The extension starts from the same selected static parameters and uses the shared predict-then-update maturity queue. It has no gradient-feedback cache. It is our evaluation policy, not the author's original online algorithm. GPT4MTS consumes historical CLS features with a token limit of 512, whereas internal adapters consume current selected mean-pooled features with a token limit of 64. Thus common targets and origins enable a complete-system comparison, while architecture, text budget and training differences remain explicit.

The fixed online GPT4MTS policy raises both MSE and MAE relative to its own static form in all 25 runs. We retain the stronger static results rather than using the degraded extension to inflate a ranking. Online full iAdapter has lower mean MSE than numerical online adaptation only on Security and Climate; against static GPT4MTS it has lower mean MSE only on Climate. No learning-rate search was conducted after seeing these scores. Native VoT has not yet been evaluated through a trustworthy origin-level prior-generation chain. Its source audit and required alignment steps are recorded separately; the original Table 2 position aggregates remain unrecovered.

## Compact text follow-up — actual local progress

The development input audit finds that the newest selected record supplies no readable content token in 139/148 Energy origins and 16/90 SocialGood origins. This motivates a controlled recent-record-first order study with the same retained characters and token budget; its performance is pending. A train-only PCA/Ridge probe adds content beyond availability metadata and lowers Energy mean MSE by 4.43% in all three seeds, while raising SocialGood mean MSE by 6.69%. Energy's result shows a locally readable signal, although its error remains worse than the bare numerical backbone.

The E003 follow-up feeds raw BERT-PCA5 and fixed LLM5 labels through the same 5→96→64 additive projection with identical initial parameters. The language model is a local frozen 0.5B model with a fixed prompt, greedy generation and five labels. Its 1538 unique outputs have valid formats, while two target-grounding mistakes in a 30-training-text preview are retained. The features are therefore tested as an imperfect representation, not described as verified causal knowledge. No prompt/model search is used. PCA and scaling use only the training prefix; missing features are zeroed after projection.

Table 2. E003 OT-only development assessment (60/10/10 prefix, H=6, seeds 2023–2025). Numerical and GPT4MTS rows reuse exact B1 controls; they are not new trainings. GPT4MTS is a complete-system reference. SD is across three matched optimization seeds.

| Dataset | Method | MSE ↓ (mean ± SD) | MAE ↓ (mean ± SD) |
|---|---|---|---|
| Energy | frozen_backbone | 0.013927 ± 0.000030 | 0.090502 ± 0.001564 |
| Energy | numeric_online | 0.014097 ± 0.000442 | 0.090025 ± 0.002486 |
| Energy | gpt4mts_static | 0.015489 ± 0.000557 | 0.092406 ± 0.001811 |
| Energy | raw_BERT_PCA5_online | 0.014697 ± 0.000694 | 0.091779 ± 0.001747 |
| Energy | LLM5_static | 0.014138 ± 0.000327 | 0.090032 ± 0.000785 |
| Energy | LLM5_online | 0.014237 ± 0.000164 | 0.090377 ± 0.000205 |
| SocialGood | frozen_backbone | 0.133175 ± 0.018801 | 0.211382 ± 0.015181 |
| SocialGood | numeric_online | 0.133797 ± 0.016984 | 0.211547 ± 0.015039 |
| SocialGood | gpt4mts_static | 0.088090 ± 0.005238 | 0.186882 ± 0.002546 |
| SocialGood | raw_BERT_PCA5_online | 0.132720 ± 0.014202 | 0.209849 ± 0.011744 |
| SocialGood | LLM5_static | 0.132819 ± 0.017523 | 0.209625 ± 0.014407 |
| SocialGood | LLM5_online | 0.130672 ± 0.015281 | 0.208180 ± 0.012376 |

Table 3. LLM5-online paired effects on the same E003 origins. Δ is candidate minus control; negative values favor LLM5. Percent changes use the ratio of mean errors, while SD is across individual paired seed differences. Improved-seed counts are descriptive and are not significance tests.

| Dataset | Control | Δ MSE mean ± SD | Δ MAE mean ± SD | MSE change (%) | MAE change (%) | MSE improved seeds |
|---|---|---|---|---|---|---|
| Energy | raw_BERT_PCA5_online | -0.0004609 ± 0.0005506 | -0.0014016 ± 0.0015510 | -3.14 | -1.53 | 2/3 |
| Energy | numeric_online | +0.0001398 ± 0.0006035 | +0.0003523 ± 0.0026032 | +0.99 | +0.39 | 1/3 |
| Energy | LLM5_static | +0.0000986 ± 0.0001782 | +0.0003453 ± 0.0006921 | +0.70 | +0.38 | 1/3 |
| SocialGood | raw_BERT_PCA5_online | -0.0020480 ± 0.0018446 | -0.0016686 ± 0.0012497 | -1.54 | -0.80 | 3/3 |
| SocialGood | numeric_online | -0.0031252 ± 0.0029077 | -0.0033668 ± 0.0035960 | -2.34 | -1.59 | 2/3 |
| SocialGood | LLM5_static | -0.0021472 ± 0.0037699 | -0.0014445 ± 0.0035791 | -1.62 | -0.69 | 2/3 |

LLM5 reduces mean MSE relative to raw PCA5 by 3.14% on Energy and 1.54% on SocialGood, with improvements in two and three seeds. The 1.54% effect remains positive empirical progress despite falling below the earlier engineering screen. SocialGood additionally improves over numerical online adaptation by 2.34% MSE and 1.59% MAE. Energy instead worsens by 0.99% MSE against numerical online adaptation, and its static LLM form is stronger than its online form. The bare numerical backbone remains strongest in Energy mean MSE, and static GPT4MTS remains much stronger on SocialGood. The new result therefore supports a restricted representation comparison and conditional text-assisted progress.

Historical donor substitution at the same online state changes LLM mean MSE by approximately 1.3e-6 on Energy and 1.73e-4 on SocialGood. Two Energy seeds actually benefit from donor text. These small mixed sensitivities do not justify assigning the average gain to semantic correctness. The raw and LLM branches have the same trainable capacity; their representation difference is an inspectable intervention, but the current development periods and noisy labels constrain generalization and mechanism claims.

## Efficiency — measured cost and accounting

Original full adapter-plus-projection parameters are 166,896 on Economy and 167,088 on Energy, compared with numerical backbone sizes of 26,854 and 27,238. The absolute adaptation module is small, but it is roughly six times this minimal backbone. Original cached prediction medians are 3.239/3.561 ms and update medians are 9.163/9.889 ms. Fresh-text predictions including a real BERT forward take 10.810/10.632 ms. Measurements use 50 warm-ups and 200 iterations on an RTX 4060 Laptop; P95, allocated GPU memory and CPU/cache payloads are supplied separately. They are isolated microbenchmarks and omit some queue/logging overhead.

E003 uses 21,632 trainable parameters per text adapter, compared with 94,880 for the OT simple-text branch with a 768-dimensional input: a 77.20% reduction. This is not an accuracy-preserving comparison to the original full model or proof of lower complete-system cost. Eighteen later measurement rounds report cached prediction/update latency, GPU memory and cache sizes with isolated model copies. The same-capacity raw and LLM branches have the same architecture, and observed timing differences on the shared GPU should not be interpreted as intrinsic architectural speedups. Frozen BERT and approximately 0.99 GB of language-model allocated GPU memory during generation are accounted for separately from online costs.

## Prospective extension — no numbers before execution

[TODO N1] Report the fixed recent-record-first results only after the current-order diagnostic reproduces old forecasts and all new forecasts pass the original-CSV audit. Keep the same budget, selected records, character inventory, warm-up and seed-specific backbone.

[TODO N3] Add a separate locked-candidate table for Agriculture and Health after all 50 evaluation cells, 10 shared backbones and 30 warm-ups complete. Include frozen, numerical online, raw PCA5 online, LLM static and LLM online controls with five-seed paired differences. Agriculture is monthly U.S. retail broiler composite. Health is national unweighted ILI outpatient percentage; five duplicate calendar intervals contain distinct YEAR/WEEK reports, so H=6 denotes the next six reports. No extension forecast result is presently available.

## Limitations and closing discussion

The revised evidence preserves both modest gains and failures. It supports a reproducible frozen-backbone delayed-update interface, conditional online parameter-adaptation gains and a restricted same-capacity text-representation improvement. It does not establish that all expert or gradient paths are necessary, that generated labels are semantically correct, or that the original full method generally outperforms strong multimodal systems. Additional domains can test a candidate selected before their evaluation, but cannot restore missing original logs or actual publication times. Original broad performance, arbitrary-delay robustness and multi-backbone/encoder claims require their own traceable evidence.

## Local writing check and claim map

| Claim in this integration | Evidence | Status |
|---|---|---|
| Full R1 lowest mean H6 MSE in 2/5 domains | reports/main_table.md, runs.csv, original_csv_audit.json | Supported, fixed five-domain protocol |
| Forecast saved before once-only mature update | engine/shared_online source and prior passed tests | Supported within end_date proxy timeline |
| Full joint online/static gains, feedback separation | replan_execution/evidence/b1 | Supported, three-seed development |
| Same-capacity E003 mean improvement | E003 summary, paired_summary, original-CSV validation | Supported, restricted development comparison |
| 21,632 parameters and 77.20% reduction | E003/cost config and model parameter counts | Supported for the specified simple branch |
| Recent-first improves forecasts | No N1 GPU output | Pending, not asserted |
| Locked candidate extends to Agriculture/Health | No N3 GPU output | Pending, not asserted |
| Fully reproduced native VoT / original Table 2 | Prior provenance and original records missing | Not asserted |
| True text publication-time causality | No independent publication metadata | Not asserted |

Reverse outline: task and interface → chronology and source truth → separate experiment blocks → main results → component diagnosis → stronger external systems → compact text progress → complete cost → pending confirmation and limits. Definitions of static, online, full and simple-text remain distinct. All new numbers are copied or computed from prior verified evidence; pending stages have explicit TODO markers. This file is an integration draft, not a submitted rebuttal or compiled final paper.

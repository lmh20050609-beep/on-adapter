# 逐点 review 回复整合稿

这是可合入回复的英文素材，尚未提交。回应对应原 review 的 ECmL 四点及 LCo1 五点。N1/N3 已完成并通过原预测/目标审计；下文保留成功和退化，VoT 接入检查与正式基线比较明确分开。原 Table 1–3 的来源缺口仍显式保留。

## ECmL F1 — consistency, significance and small effects

We have replaced the consistency and significance wording with the observed dataset-specific comparisons. No paired significance test was conducted. The revised five-domain H=6 evaluation reports all five seeds, sample SD and paired differences; full iAdapter has the lowest mean MSE on two domains and MAE on one. Energy improves over numerical adaptation in four seeds but has worse overall mean error because of one large adverse outcome. We retain this variability rather than reporting only favorable seeds.

We also report positive incremental results directly. In the separate three-seed OT development follow-up, the same-capacity LLM5 representation reduces online mean MSE over raw BERT-PCA5 by 3.14% on Energy and 1.54% on SocialGood; mean MAE falls by 1.53% and 0.80%. SocialGood improves over numerical online adaptation by 2.34% MSE and 1.59% MAE, whereas Energy worsens by 0.99% and 0.39%. The modest improvement is useful evidence within this comparator and split, although it is not independent confirmation or general superiority. The earlier 2% engineering screen is distinguished from statistical significance.

Evidence: previous delivery `reports/main_table.md`, `main_pairwise_summary.csv`, and `refine-logs/replan_execution/evidence/E003/{summary.csv,paired_summary.csv,validation.json}`; integrated Tables 1–3.

## ECmL F2 — mismatched insertion-position statistics

The submitted Table 2 uses incompatible sources of variation, and its original position-level records have not been recovered. We cannot repair those aggregates by relabeling new runs. The new internal comparison shares each seed-specific numerical backbone, second-layer tail, input and valid scoring targets. Its SD measures optimization-seed variation conditional on the fixed split. Complete-system GPT4MTS results are presented separately with common OT targets and origins; their architecture and text budget differ. We withdraw the original Table 2 superiority claim until traceable matched-position records are available.

Evidence: original-CSV audits, fixed configs, external static/online control tables; `VOT_SOURCE_AND_INTEGRATION.md` states the remaining native VoT requirements.

## ECmL F3 — routing gradients, entropy and component necessity

The revised matrix includes no text gradients, no routing text gradients, no numerical gradients, no distribution descriptor, no entropy, homogeneous experts and uniform fusion. The routing-gradient removal has lower mean MSE in Energy, Security and SocialGood; all paired differences and SD are supplied. We do not assume these effects are outside seed variability or assign them a cause without isolation.

A same-start U×F study separates parameter adaptation from deployment feedback refresh. Joint online adaptation improves the full model over its warmed static state by 28.71% MSE on Energy and 2.53% on SocialGood, but feedback refresh adds no average benefit over parameter updates alone in these settings. These observations preserve the useful adaptation gain while narrowing the gradient-cache claim. Entropy has a separate switch; effective expert count is a diagnostic rather than an independently trained component. Individual expert semantic roles remain unverified.

Evidence: `reports/component_verdicts.csv`, `method_appendix.md`, and `refine-logs/replan_execution/evidence/b1/`.

## ECmL F4 — omitted implementation and efficiency

The implementation appendix gives exact branch computations, gate inputs, descriptor, feedback refresh, optimizer, entropy coefficient and ablation boundaries. Numerical processing is chronological; normalization uses the training prefix, input filling uses history only, and original missing targets remain masked. Frozen contextual BERT features use all 12 layers, attention masks and masked mean pooling at a 64-token limit. The selected text IDs and feedback snapshots are saved per origin. Replayed records use current projection parameters and original context. Text availability uses the dataset end_date proxy; we cannot claim verified publication-time causality.

Costs are measured on the RTX 4060 Laptop. Original full adapter/projection counts are 166,896/167,088 on Economy/Energy, approximately six times the small numerical backbone, so relative parameter-lightness is not established. Cached prediction medians are 3.239/3.561 ms, updates 9.163/9.889 ms, and fresh-text predictions with BERT 10.810/10.632 ms. P95, allocated GPU memory and logical CPU/cache payloads accompany the table. E003 reduces its simple text branch from 94,880 to 21,632 trainable parameters; frozen text/generation costs are separate. Isolated microbenchmarks are not total deployment throughput.

Evidence: `reports/efficiency.json`, `text_encoding.json`, and `refine-logs/replan_execution/evidence/cost/`.

## LCo1 P1 — replay and delayed supervision

For origin s, the next H observations mature at s+H. At each current origin the executor first emits and saves its forecast, then consumes each unused mature record once. Replay retains the original numerical context, text IDs and feedback snapshots; new feedback affects later forecasts. The last H pending records are not flushed after evaluation. Only issued forecasts are scored against observed original CSV targets.

All new internal and GPT4MTS online comparisons use the same maturity executor. This aligns update opportunity, target masks and scoring origins while preserving each method's declared trainable parameters and feedback mechanism. Static systems are retained as distinct references, rather than treated as online methods.

## LCo1 P2 — GPT4MTS, VoT and fairness

Author-model GPT4MTS, the submitted GPT4MTS_z proxy and our fixed online extension are named separately. GPT4MTS consumes historical CLS text with up to 512 tokens; internal adapters use selected current mean-pooled text with 64. We therefore describe a complete-system comparison on a common task, not equal architecture or text consumption. Its fixed online extension is worse than its own static form in all 25 runs; the stronger static baseline remains in every relevant table.

The native VoT three-stage runtime integration smoke has passed; its full matched baseline evaluation is still pending. The source audit identifies a released-prior directory mismatch, 70/10/20 versus our 60/20/20 split, an embedding-version difference, and missing origin/text/retrieval generation metadata in published priors. These are concrete replication/alignment requirements, not proof of leakage or failure of VoT. We do not use those priors as ground truth or claim that Table 2 has been recovered.

## LCo1 P3 — robustness of small improvements

We report both mean effects and matched-seed SD, retaining all outcomes. Low or variable improvements are identified as such, rather than erased by an engineering threshold. Three seeds in the E003 development study improve SocialGood MSE over raw PCA5, while two improve Energy; comparison to numerical adaptation and stronger complete systems remains visible. Repeated optimization seeds do not turn a fixed inspected period into independent data. A locked Agriculture/Health extension is implemented and code-reviewed, but has no GPU results yet and cannot be cited as confirmation.

## LCo1 P4 — expert roles and useful textual signals

The full component table isolates the specified input and regularization paths and includes parameter-reported homogeneous experts. Removing one path can improve mean error, so universal necessity is not claimed. The spectral branch operates over latent coordinates, and latent centering is not demonstrated temporal decomposition. Strong gate concentration is reported without inferring semantic specialization.

Content probes distinguish availability metadata from readable text. A train-only content residual probe improves Energy over metadata while worsening SocialGood. The fixed LLM5 simple-fusion comparison offers restricted positive evidence, but same-state donor effects are small and mixed and do not prove semantic causation. An input audit shows that Energy's latest selected record is unread by the 64-token encoder in 139/148 development origins. The completed equal-budget record-order study shows a small SocialGood mean gain and an Energy regression. It is exploratory evidence, with all seed differences retained.

## LCo1 P5 — datasets, horizons and table organization

The 190-run R1 block uses five domains, H=6, all disclosed numerical target channels and 60/20/20; H=12 is limited to Economy/Energy. The external OT block uses five domains, five seeds and the inspected 60/10/10 prefix protocol. Later three-seed OT component/readout/E003 studies remain separate. Reused controls are marked and not counted as new trainings. The locked Agriculture/Health OT H=6 evaluation at 60/20/20 has completed all fifty cells with five seeds, ten shared backbones and thirty warm-ups; its separate tables retain the stronger static controls. Health contains distinct reports with duplicate calendar intervals, so H denotes the next reports rather than exact calendar weeks at every origin.

The revision does not restore ten-domain SOTA, arbitrary-delay, multi-backbone or multi-encoder claims from these narrower experiments. Full LaTeX sources, original broad-table provenance, true publication times and a trustworthy native VoT generation chain remain the materials needed to finish the corresponding claims.

## Coverage and current status

All nine specific questions have a response above. WSL/CUDA have recovered, code and the locked plan are synchronized through ef22b671, and completed N1/N3 forecasts pass the source-target checks. Their numbers are integrated in separate tables. Native VoT integration passes but trusted prior generation and full comparison remain unfinished. This is replacement text; it has not been submitted.


## 2026-10-01 completed continuation evidence

For the concerns about small and inconsistent gains, the additional-domain results retain all controls. LLM5-online lowers Health mean MSE by 7.99% versus numerical online adaptation and 6.06% versus raw PCA5 online, but its own static form is stronger by 8.13%. Agriculture shows adverse mean error. The recent-record-first study retains a small SocialGood gain and Energy regression; it is exploratory. These results sharpen the scope of text-assisted progress and do not restore consistency or significance claims.

For reproducibility and baseline control, all fifty additional-domain cells passed original-CSV target/mask and matched-origin checks, using ten shared backbones and thirty warm-ups. The fixed candidate uses the original text ordering. A native VoT three-stage real-target integration check also passes after the recorded missing dependency was installed, but prior-generation availability remains unknown. No formal VoT performance comparison is claimed from the smoke.

Full tables, paired seed differences, generation costs and the repeated-calendar handling for Health are included in the updated manuscript replacement. The new first-principles calibration route is planned separately and supplies no current performance evidence.

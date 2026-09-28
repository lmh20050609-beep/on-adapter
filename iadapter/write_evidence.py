"""Write manuscript replacement paragraphs and responses from verified outputs."""
import json
import pandas as pd
from .config import ROOT,MAIN,DOMAINS
from .analyze import LABEL,md_table

def main():
    out=ROOT/'reports'
    valid=json.loads((out/'validation.json').read_text()); assert valid['completed_runs']==190
    assert valid.get('original_csv_audit_passed')==190, 'Run iadapter.analyze before generating manuscript evidence'
    summary=pd.read_csv(out/'summary.csv'); paired=pd.read_csv(out/'paired_summary.csv')
    eff=json.loads((out/'efficiency.json').read_text()); assert len(eff)==6
    main=summary[(summary.horizon==6)&summary.variant.isin(MAIN)]
    mse_winners=main.loc[main.groupby('dataset').mse_mean.idxmin(),['dataset','variant']]
    mae_winners=main.loc[main.groupby('dataset').mae_mean.idxmin(),['dataset','variant']]
    mse_wins=int((mse_winners.variant=='full').sum()); mae_wins=int((mae_winners.variant=='full').sum())
    contrasts=paired[(paired.horizon==6)&paired.variant.isin(['numeric_add','text_add'])]
    comparison=contrasts[['dataset','variant','delta_mse_mean','delta_mse_sd','mse_reduction_mean']].copy()
    comparison['variant']=comparison.variant.map(LABEL)
    for c in ['delta_mse_mean','delta_mse_sd']: comparison[c]=comparison[c].map(lambda x:f'{x:.6f}')
    comparison['mse_reduction_mean']=comparison.mse_reduction_mean.map(lambda x:f'{x:.2f}%')
    ab=paired[(paired.horizon==6)&~paired.variant.isin(['numeric_add','text_add'])].copy()
    # Positive delta means removing/replacing the component has lower error.
    ab['simpler_better_mean_mse']=ab.delta_mse_mean>0
    ab.to_csv(out/'component_verdicts.csv',index=False)
    no_route=ab[ab.variant=='no_route_text_gradient']
    n_route_better=int(no_route.simpler_better_mean_mse.sum())
    pairwise=pd.read_csv(out/'main_pairwise_summary.csv')
    text_effects=pairwise[(pairwise.horizon==6)&(pairwise.candidate=='text_add')&(pairwise.control=='numeric_add')]
    text_mse_wins=int((text_effects.delta_mse_mean<0).sum())
    text_mae_wins=int((text_effects.delta_mae_mean<0).sum())
    (out/'main_pairwise_analysis.md').write_text('# All main-method paired contrasts\n\nDelta = candidate minus control; negative values favor candidate. Five matched seeds share a backbone and scoring targets. SD is computed over seed differences. Improved-seed counts do not imply significance.\n\n'+md_table(pairwise.round(6))+'\n',encoding='utf-8')
    (out/'method_appendix.md').write_text((ROOT/'docs/METHOD_APPENDIX.md').read_text(encoding='utf-8'),encoding='utf-8')
    routing=pd.read_csv(out/'routing_summary.csv')
    seed_pairs=pd.read_csv(out/'main_pairwise.csv')
    failure_pool=seed_pairs[(seed_pairs.horizon==6)&(seed_pairs.candidate=='full')&(seed_pairs.control=='numeric_add')]
    failures=failure_pool.loc[failure_pool.groupby('dataset').delta_mse.idxmax()].copy()
    failures=failures.merge(routing[(routing.horizon==6)&(routing.variant=='full')],on=['dataset','horizon','seed'],validate='one_to_one')
    names=['Zero-init','Low-rank','Spectral','Fluctuation','TextFiLM']
    expert_cols=[f'expert_{i}_mean_weight' for i in range(5)]
    failures['dominant_expert']=failures[expert_cols].idxmax(axis=1).map({col:name for col,name in zip(expert_cols,names)})
    failures['dominant_mean_weight']=failures[expert_cols].max(axis=1)
    failure_table=failures[['dataset','seed','delta_mse','delta_mae','dominant_expert','dominant_mean_weight']]
    (out/'failure_cases.md').write_text('# Failure-case audit\n\nFor each H6 dataset, this table selects the seed with the largest full-minus-numerical MSE difference among all five registered seeds. This is explicitly a post-hoc adverse-case selection, not a representative random sample or a model-selection rule. Positive differences mean degradation. No selected run is removed from the main table.\n\n'+md_table(failure_table.round(6))+'\n\nThe dominant-expert column describes concurrent average routing concentration. It does not establish why a run failed: the learned backbone/adapter, optimization randomness, feedback and gate can interact. A route-disabled or matched-seed ablation changes the learning trajectory and is not a controlled intervention on an otherwise identical final model. Report the paired mean and SD alongside these cases, and do not explain every small difference as either noise or negative transfer.\n\nThe primary all-seed contrasts and improvement counts are in main_pairwise_analysis.md. The near-single-expert mechanism figures use fixed seed 2023 and were not chosen from this adverse-case table. Upstream text provenance remains insufficient to diagnose source-level semantic causes.\n',encoding='utf-8')
    routed=routing[(routing.variant=='full')&(routing.horizon==6)]
    concentrated=int((routed[[f'expert_{i}_mean_weight' for i in range(5)]].max(axis=1)>.99).sum())
    small=main.copy()
    small['MSE']=small.apply(lambda r:f'{r.mse_mean:.6f} ± {r.mse_sd:.6f}',axis=1)
    small['MAE']=small.apply(lambda r:f'{r.mae_mean:.6f} ± {r.mae_sd:.6f}',axis=1)
    small['method']=small.variant.map(LABEL)
    table=md_table(small[['dataset','method','MSE','MAE','n']])
    effrows=[]
    for r in eff:
        effrows.append({'dataset':r['dataset'],'method':LABEL[r['variant']],'numeric_backbone_params':r['backbone_parameters'],'trainable_params':r['trainable_parameters'],'adapter_to_backbone_ratio':round(r['trainable_parameters']/r['backbone_parameters'],2),'total_params_including_BERT':r['total_parameters'],'cached_pred_ms':round(r['prediction_cached_text']['median_ms'],3),'update_ms':round(r['update_fixed_mature_record']['median_ms'],3),'fresh_text_e2e_ms':round(r['fresh_text_end_to_end_prediction']['median_ms'],3),'adapter_peak_MiB':round(r['adapter_peak_cuda_allocated_bytes']/2**20,2)})
    ef=pd.DataFrame(effrows); ef.to_csv(out/'efficiency_table.csv',index=False); ef.to_latex(out/'efficiency_table.tex',index=False,escape=True)
    param_table=ef[['dataset','method','numeric_backbone_params','trainable_params','adapter_to_backbone_ratio','total_params_including_BERT']]
    latency_table=ef[['dataset','method','cached_pred_ms','update_ms','fresh_text_e2e_ms','adapter_peak_MiB']]
    param_table.to_latex(out/'parameter_table.tex',index=False,escape=True)
    latency_table.to_latex(out/'latency_table.tex',index=False,escape=True)
    (out/'efficiency.md').write_text('# 实测效率\n\n## 参数\n\n'+md_table(param_table)+'\n\n完整适配器加文本投影的可训练参数约为这个 minimal 数值骨干的 6 倍。绝对规模较小，不代表参数量相对数值骨干更少；不能以包含冻结 BERT 的总参数作分母掩盖这一点。\n\n## 延迟与显存\n\n'+md_table(latency_table)+'\n\n单位：毫秒，表内为中位数。完整 P95、50 次预热/200 次测量记录见 efficiency.json。固定真实窗口的隔离微基准，不修改正式实验状态；使用同一台共享 RTX 4060 Laptop GPU。Update 包含回放损失、梯度探针和优化器，不含队列管理/日志 I/O。Adapter peak 是本进程 PyTorch 峰值 allocated memory，不是整卡用量，也不等于 CPU 文本缓存。队列数字统计张量与文本 ID 内容，不含 Python 对象管理开销；text_cache_bytes 为 CPU embedding 数组大小，GPU 副本计入峰值显存。新文本端到端预测包括真实 BERT 前向；缺省在线实验使用已缓存的冻结表示。\n',encoding='utf-8')
    report=f'''# iAdapter 修订实验结果

## 完成情况

190/190 次预定适配评估已完成；35 个骨干配置；所有保存的预测均已重新计算指标，并核对同组原点、原始目标、有效掩码及共享骨干一致。代码提交：`{valid['code_commit']}`。没有使用旧 finally_mm.csv，也没有通过选择 seed 或追加方法改善结论。

## H=6 主结果

{table}

完整 iAdapter 在三个受控方法中取得 **{mse_wins}/5 个数据集最低平均 MSE、{mae_wins}/5 个最低平均 MAE**。这是五个固定数据流上五 seed 的描述性结果，不是十数据集 SOTA 证明，也不表示统计显著。

### 配对差值

{md_table(comparison)}

delta_mse_mean = iAdapter MSE − 对照 MSE，负值有利于 iAdapter；相对改善先在共享骨干的配对 seed 内计算，再取平均。不同域的原始 MSE 不直接混合平均。逐 seed 百分比的平均与原始 MSE 均值的相对变化不是同一个统计量，可能出现符号不同；主表最优计数始终按原始 MSE/MAE 均值判断。

### 文本本身是否有帮助

简单文本加法相较数值加法，在 H6 的 {text_mse_wins}/5 个域降低平均 MSE、{text_mae_wins}/5 个域降低平均 MAE。三种主方法的全部配对差值见 main_pairwise_analysis.md；逐 seed 数据见 main_pairwise.csv。这个结果不能用于预设文本一定有益。

低/中/高历史状态变化组内的配对差值另见 state_paired_analysis.md。每组比较同 seed、同原点、同有效目标集合；分组分数和验证集阈值已对照原始 CSV 核验。分组解释属于描述性分析，不能据此选择正式模型或声称文本/漂移因果机制。

各域最不利 seed 的明确事后案例见 failure_cases.md；它们按完整模型相对数值加法的 MSE 增量选择，全部仍在主结果中。路由集中与误差并列展示，不被解释为已经证实的失败原因。

## 消融结论

移除路由文本梯度，在 Energy、SocialGood、Security 三域中有 **{n_route_better}/3** 的平均 MSE 低于完整模型。完整逐项结果见 component_verdicts.csv 和 ablation_table.csv。不能据此宣称全部组件稳定必要；也不能将任一小差值自动解释成负迁移或随机波动。没有进行显著性检验。

## 效率和机制

效率见 efficiency.md；原始中位数、P95 和内存分类见 efficiency.json。完整 BERT 的批处理编码成本另见 text_encoding.json。

figures/main_paired_effects.pdf 展示全部主实验配对效果。Energy 与 SocialGood 机制图使用预定 seed=2023 的完整测试流，展示多步预测中的第一步及变量平均 gate，未挑选模型表现最好的时间片。状态变化是历史窗口的描述性统计，不是概念漂移真值；专家权重不是因果功能认证。

**路由诊断：25 次 H6 完整模型运行中，{concentrated} 次存在某一专家的平均权重超过 99%。** 这是明确的强集中现象，不能把这两幅几乎单色的机制图写成“随漂移切换专家”的证据。不同 seed 的主导专家可能不同。阈值 99% 仅用于描述已记录的 gate，不用于挑选模型或删除实验；详见 routing_summary.csv。

## 不能恢复的原稿证据

- 本轮骨干为 minimal ITransformerZ，数值基线为 ADAPT-Z-style 适配，不能冒充原方法完整复现。
- 未取得 GPT4MTS/VoT 匹配协议结果，因此旧 Table 2 优越性声明仍需撤下或待来源恢复后重新比较。
- 文本使用当前数值聚合期结束时的 end_date 代理可用性，未核验真实新闻发布时间。
- 本轮只有五域 H6，以及 Economy/Energy H12；不支持十域、多骨干、多编码器、普遍反馈延迟鲁棒性声明。
- 五 seed 只刻画固定数据流上的训练随机性；不代表对新数据集的泛化置信区间。
- 当前材料没有 LaTeX 正文源码，提供可替换段落和 LaTeX 表图片段，不提供已编译完整修订论文。

## 审稿意见状态

已实质处理：本轮比较协议、指标来源、回放一致性、关键消融、完整文本编码、参数与在线成本。
部分处理：文本与状态变化关系、组件必要性、性能一致性；结论随数据限制。
尚未完成：原多模态方法匹配比较及原表完整来源恢复，不能宣称两份 review 已全部解决。
'''
    (out/'analysis-report.md').write_text(report,encoding='utf-8')
    manuscript=f'''# Manuscript replacement text

These are replacement paragraphs for a revised evaluation. They do not assert that the original submission already used this implementation. Numerical tables are supplied separately; no LaTeX manuscript source was available.

## Outline

1. Define period-end forecast origins and delayed supervision.
2. Describe fixed-backbone residual adaptation and immutable replay contexts.
3. Disclose contextual text encoding, missing-target masks and initialization.
4. Present controlled main comparisons, component limits and measured costs.
5. State unresolved original comparisons and availability limitations.

## Abstract: replacement empirical scope

We study text-assisted online adaptation of a frozen time-series forecaster under delayed multi-step supervision. At each forecast origin, the adapter forms a latent residual from numerical features, available text representations and historical gradient feedback. Mature targets update the adapter using the original forecasting context; the resulting feedback is used only for later forecasts. We evaluate the revised implementation on five paired datasets using five matched seeds and a common online protocol. At horizon six, iAdapter obtains the lowest mean MSE on {mse_wins} of five datasets and the lowest mean MAE on {mae_wins} of five datasets among the three controlled alternatives. Component ablations and measured inference/update costs characterize the method's benefits and limitations. Text availability is approximated by dataset record end dates rather than verified publication timestamps.

## Method: forecasting timeline and replay

A forecast origin is the end of the currently observed numerical aggregation period. The input contains the preceding L observations, and the target comprises the following H observations. The system first emits and records its forecast. It then consumes each previously unused origin whose entire target period has elapsed. Each pending record stores an origin identifier, the selected text record identifiers, the frozen numerical representation and detached numerical/textual feedback snapshots. During replay, the current adapter parameters operate on this saved origin context. A zero-residual probe provides numerical feedback, while the prediction loss provides sensitivity with respect to the projected text representation. After the adapter update, the numerical cache is updated with an exponential average and the latest text feedback is stored for subsequent origins. New feedback is not substituted into the replay that produced it. The final H pending origins are not flushed after the last evaluated forecast.

The complete expert computations, gate equations, eight-dimensional descriptor, optimizer chronology and precise ablation boundaries are provided in method_appendix.md for direct manuscript integration. In particular, the spectral operation and feature centering act along latent coordinates; their names do not establish temporal seasonal or fluctuation decomposition. The zero-initialized expert remains trainable, and all five experts are evaluated under soft routing.

## Related-work addition: Dynamic TMoE

Dynamic TMoE couples drift detection, heterogeneous experts and temporal memory routing, with expert-pool evolution restricted to training. Our revised setting instead evaluates text-assisted parameter updates during deployment, with a frozen numerical backbone and delayed target availability. The presence of heterogeneous experts or historical routing information alone is therefore not treated as a unique contribution. We include a homogeneous expert-pool control and distinguish structural biases from demonstrated semantic specialization. Reference: [Dynamic TMoE, Sections 3.6 and Appendix I](https://arxiv.org/html/2605.20678v1).

## Text and preprocessing

We encode the selected text with the complete frozen 12-layer BERT-base model, using an attention mask and masked mean pooling with a maximum of 64 tokens. Frozen representations are cached and shared across methods; a 64-dimensional projection remains trainable. Text is selected from the two most recent known merged records. If no informative text is available, the projected representation is set to zero. Record end dates serve as an availability proxy; they do not establish the true publication times of the underlying sources. Numerical inputs are forward-filled using observed history, with a fixed zero policy for leading missing values. Standardization is fitted on the training prefix only. Missing original targets are excluded from both the loss and evaluation by a shared validity mask.

## Experimental setup

We use Economy, Energy, Security, SocialGood and Climate, with chronological 60/20/20 splits. The backbone is the disclosed minimal ITransformerZ implementation with dimension 32, two encoder layers, four attention heads and feedforward dimension 128. The lookback is 48 for Energy and 36 otherwise. All adapters use the same tail insertion location and each seed-specific pretrained backbone. Backbone training uses Adam with learning rate 0.001, batch size 32 and at most 30 epochs; validation masked MSE selects the checkpoint with patience five. The adapter is warmed for one chronological training-stream pass. Its optimizer, feedback and queue are reset before testing, while its warmed parameters are retained. No validation-set parameter updates are performed. Online Adam uses learning rate 0.001, weight decay 0.0001 and gradient clipping at one. Seeds are 2023 through 2027. Experiments run serially on an NVIDIA RTX 4060 Laptop GPU with 8 GB memory.

## Results and component interpretation

Table R1 reports mean and sample standard deviation across five matched runs for numerical additive adaptation, additive text fusion and iAdapter. The numerical method is an ADAPT-Z-style controlled adapter and is not presented as a reproduction of the original method's complete recipe. At H=6, iAdapter has the lowest mean MSE on {mse_wins}/5 datasets and lowest mean MAE on {mae_wins}/5 datasets. Gains and degradations are both retained. Table R2 isolates text feedback, numerical feedback, the distribution descriptor, entropy regularization, expert heterogeneity and learned fusion. Removing routing text gradients lowers mean MSE on {n_route_better}/3 evaluated domains; this evidence does not support a universal necessity claim for that component. Additional H=12 results cover Economy and Energy only. We make no statistical significance or universal robustness claim from these experiments.

Routing diagnostics show that {concentrated} of 25 full-model H6 runs assign more than 99% average weight to one expert. This post-hoc descriptive threshold is not used for model selection. The representative mechanism plots therefore illustrate strongly concentrated routing rather than demonstrating adaptive temporal specialization.

The additive text control lowers mean MSE on {text_mse_wins}/5 H6 domains and mean MAE on {text_mae_wins}/5 relative to the numerical additive control. All three pairwise contrasts are reported with mean and sample standard deviation of matched-seed differences. Historical state-change groups use validation-only thresholds, and group comparisons retain the same forecast origins and valid targets within each seed. These descriptive contrasts test whether gains occur within measured regimes; they do not establish the semantic usefulness of a particular text type.

## Efficiency and limitations

Table R3 distinguishes trainable parameters, cached-text prediction latency, delayed-update latency and fresh-text end-to-end latency. Measurements use isolated copies with 50 warm-up and 200 measured iterations; medians and 95th percentiles are reported. GPU allocated memory, replay/gradient state and CPU text caches are reported separately. These results characterize this implementation on the measured device, not hardware-independent cost. The evaluation does not recover the original GPT4MTS/VoT comparison, and it does not establish true publication-time availability, causal expert specialization or generalization across the original ten-domain, multi-backbone setting.

The trainable adapter and text projection together contain roughly six times as many parameters as the small numerical backbone used here. Their small absolute size does not establish parameter efficiency relative to that backbone, and including the frozen BERT parameters in the denominator would obscure this comparison.

## Claim–evidence map and self-check

| Claim | Evidence | Status |
|---|---|---|
| Origin-aligned mature-label updates | protocol tests; updates.jsonl; validation.json | Supported for implemented protocol |
| Contextual BERT rather than embedding-only pooling | prepare.py; cache/direct checks | Supported |
| H6 performance counts | summary.csv; paired_differences.csv | Descriptive, fixed streams |
| Every module improves performance | component_verdicts.csv | Not asserted |
| Lightweight online deployment | efficiency.json | Limited to measured cost dimensions and hardware |
| Original multimodal SOTA comparison | Missing matched GPT4MTS/VoT runs | Unresolved |
| Publication-time leakage guarantee | Only end_date proxies available | Not asserted |

Self-check: scope and terminology match the implementation; comparisons share data and backbone; negative results are preserved; explanatory hypotheses are not stated as established mechanisms; missing original comparisons remain explicit.
'''
    (out/'manuscript_replacements.md').write_text(manuscript,encoding='utf-8')
    response=f'''# Point-by-point response draft

This draft refers to the NEW revision evidence package. Original Tables 1–3 are not presented as reproduced. Replace R1/R2/R3 with final manuscript table numbers after integration. Both reviews' unresolved concerns remain explicit.

## Reviewer ECmL

### W1 / Feedback 1 — consistency and significance

We agree that the original blanket consistency/significance statement was stronger than the available evidence. We have removed that statement. The revised controlled evaluation reports five seeds, paired differences and both improvements and regressions. At H=6, the full model has the lowest mean MSE on {mse_wins}/5 datasets and MAE on {mae_wins}/5 among the three controlled alternatives. We do not have a verified paired-significance analysis supporting the original statement, and the new package makes descriptive claims only. Evidence: main_table.md, paired_differences.csv, stats-appendix.md. Manuscript: Results and experimental setup replacements.

### W2 / Feedback 2 — incompatible position/run aggregates

We agree that averaging baseline positions while reporting run variability for a selected iAdapter position is not a matched comparison. The new controlled experiments fix the backbone and tail insertion position, share each pretrained seed-specific checkpoint, and report only across-seed variability. We have not recovered matched GPT4MTS/VoT runs; therefore the old Table 2 superiority claim must be withdrawn from the verified revision rather than relabelled as validated. Evidence: config.json in each run, validation.json. Remaining limitation: external multimodal baseline comparison is unresolved.

### W3 / Feedback 3 — text-gradient counterexamples and unisolated components

We separated removal of all text gradients from removal of routing-only text gradients. The latter retains gradient-conditioned expert inputs while using unmodulated text in the gate and removing the direct textual-gradient logit term. Eight variants are evaluated on Energy and SocialGood, with an additional routing-gradient ablation on Security. The routing-gradient removal has lower mean MSE on {n_route_better}/3 domains. We report paired variation and do not assume these differences are noise or claim every component is necessary. We additionally isolate the entropy term, numerical feedback, distribution descriptor, homogeneous experts and uniform fusion. Evidence: ablation_table.csv, component_verdicts.csv, per-run configs.

### W4 / Feedback 4 — implementation, chronology and efficiency

We provide fixed expert/update hyperparameters, original-context replay, explicit maturity queues and saved per-origin traces. During verification we found a historical text/numerical replay misalignment and differences between stored code variants; the new implementation corrects these issues and is evaluated independently of historical CSVs. Input preprocessing is causal under a fixed training-fitted scaler and invalid original targets are masked. Complete contextual BERT is used. We report parameter counts, cached-text and fresh-text latency, update latency, GPU allocation and replay/text cache memory. Record end_date is an availability proxy, not a verified publication timestamp. Evidence: PROTOCOL.md, DECISIONS.md, validation.json, efficiency.json, text_encoding.json.

The accompanying method_appendix.md now specifies each branch's affine dimensions, activations, initialization, gate inputs, entropy sign and origin-level pseudocode. Its spectral transform is explicitly along latent coordinates, not the time axis. The source audit additionally verifies historical-window state scores and validation-only grouping thresholds.

## Reviewer LCo1

### W1 / P3 — small and inconsistent improvements

We have narrowed the claims and report all five seeds with matched differences. The result counts are {mse_wins}/5 for mean MSE and {mae_wins}/5 for mean MAE in the H6 controlled comparisons. Negative outcomes remain in the tables. We do not infer statistical significance or broad generalization from small average gains. Evidence: main_table.md and stats-appendix.md.

### W2 / P4 — individual components

The revised ablation matrix separates text, all text gradients, routing-only text gradients, numerical gradients, the distribution descriptor, the decisiveness regularizer, expert heterogeneity and learned fusion. Homogeneous MLP experts approximately match the heterogeneous expert parameter budget, with the exact difference recorded per run. Effective expert count is a diagnostic of soft weights, not a reduction in evaluated expert count. Evidence: ablation_table.csv, component_verdicts.csv and expert_matching fields in metrics.json.

### W3 / P1 / P2 — replay and controlled protocols

Each method uses the same forecast-origin timeline and valid target set. Forecasts precede updates; only fully matured origins are consumed, exactly once. Replays retain their original text IDs and both feedback snapshots, and newly computed feedback affects only later forecasts. The numerical backbone, insertion location and seed-specific checkpoint are shared across the three main alternatives. This controls the new internal comparisons. GPT4MTS/VoT were not re-evaluated in this local package, so their original comparison remains unresolved and is not used to claim superiority. Evidence: updates.jsonl, prediction_context.json, validation.json.

### W4 — useful textual signals and routing under shifts

We add descriptive low/medium/high state-change analyses using historical-window scores and thresholds selected from validation data, plus full-stream Energy/SocialGood routing figures with fixed seed 2023. These figures align first-step forecasts, historical state-change scores, text availability and expert weights. They do not establish semantic causality or prove that any particular type of text is universally useful. Finer text-content interventions remain outside this limited revision package. Evidence: state_groups.csv, figure-catalog.md, mechanism figures.

For each state group we additionally report all three matched-seed contrasts, including additive text versus numerical adaptation, with paired-difference standard deviations and the number of improving seeds (state_paired_analysis.md). Across all H6 origins, additive text improves mean MSE on {text_mse_wins}/5 domains and mean MAE on {text_mae_wins}/5 (main_pairwise_analysis.md). We therefore do not assume that access to the supplied text is beneficial by itself.

The diagnostics also reveal strong routing concentration: {concentrated} of 25 full-model H6 runs assign more than 99% mean weight to one expert. We therefore do not use these visualizations as evidence of adaptive temporal specialization. This is a limitation of the measured configuration, not a successful mechanism claim.

### P5 — datasets, horizons and table organization

The revised main table covers Economy, Energy, Security, SocialGood and Climate at H=6, with five seeds and a shared protocol. The H=12 extension covers Economy and Energy. Core ablations cover Energy and SocialGood at H=6, plus the routing-text-gradient ablation on Security. All tables explicitly name scope and aggregation. This reduced, traceable evaluation replaces broad unsupported claims; it does not reconstruct the old ten-domain evaluation. Evidence: manifest.json, summary.csv and experiment setup replacement.
'''
    (out/'reviewer_responses.md').write_text(response,encoding='utf-8')
    (out/'claim_changes.md').write_text(f'''# 主张修改摘要

- 保留：固定骨干、文本辅助残差适配、明确的延迟监督队列；已有代码和逐原点记录支持。
- 收紧：H6 三方法比较中 {mse_wins}/5 个最低平均 MSE、{mae_wins}/5 个最低平均 MAE；具体域和负收益完整列出。
- 不再宣称：所有组件稳定有效、所有场景显著领先、十域 SOTA、真实发布时间完全核验、因果专家分工、普遍延迟鲁棒性。
- 效率：以本机实际参数/延迟/内存数据替代笼统 lightweight 声明。
- 待补：原表运行来源及 GPT4MTS/VoT 匹配比较，取得原稿 LaTeX 后再正式整合排版。
''',encoding='utf-8')
    encoding={d:json.loads((ROOT/'cache'/f'{d}_bert_contextual.json').read_text()) for d in DOMAINS}
    (out/'text_encoding.json').write_text(json.dumps(encoding,indent=2))
    print(f'WRITING_COMPLETE mse_wins={mse_wins}/5 mae_wins={mae_wins}/5 routing_removal_better={n_route_better}/3')

if __name__=='__main__': main()

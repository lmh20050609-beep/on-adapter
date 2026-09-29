"""Render a bounded, descriptive review response from the verified development sprint."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .review_sprint import DEFAULT_RUN,development_data
from .analyze import md_table

def main(base=DEFAULT_RUN):
    base=Path(base);out=base/'reports';proof=json.loads((out/'validation.json').read_text())
    assert proof['status']=='passed' and proof['completed']==42 and not proof['test_80_100_evaluated']
    summary=pd.read_csv(out/'main_summary.csv');pairs=pd.read_csv(out/'paired_summary.csv');raw_pairs=pd.read_csv(out/'paired_runs.csv');probes=pd.read_csv(out/'text_probe_summary.csv')
    labels={'numeric_add':'数值加法','text_add':'文本加法','full':'完整模型','no_route_text_gradient':'关闭路由文本梯度','no_decisiveness':'关闭熵正则','no_route_no_decisiveness':'两者都关闭','single_mlp':'单 MLP'}
    display=summary[['dataset','variant','mse_mean','mse_sd','mae_mean','mae_sd','parameters']].copy()
    display['MSE mean ± SD']=[f'{m:.6f} ± {s:.6f}' for m,s in zip(display.mse_mean,display.mse_sd)]
    display['MAE mean ± SD']=[f'{m:.6f} ± {s:.6f}' for m,s in zip(display.mae_mean,display.mae_sd)]
    display['variant']=display.variant.map(labels);display=display[['dataset','variant','MSE mean ± SD','MAE mean ± SD','parameters']]
    effect=[]
    for d,block in summary.groupby('dataset'):
        b=block.set_index('variant');reference=b.loc['full']
        for v in ['no_decisiveness','no_route_text_gradient','no_route_no_decisiveness','single_mlp']:
            q=pairs[(pairs.dataset==d)&(pairs.candidate==v)&(pairs.control=='full')].iloc[0]
            effect.append({'dataset':d,'candidate':v,'MSE_change_of_means_percent':100*(b.loc[v,'mse_mean']/reference.mse_mean-1),'paired_MSE_delta_mean':q.delta_mse_mean,'paired_MSE_delta_SD':q.delta_mse_sd,'MSE_improved_seeds':int(q.mse_improved_seeds),'parameter_reduction_percent':100*(1-b.loc[v,'parameters']/reference.parameters)})
    effect=pd.DataFrame(effect);effect.to_csv(out/'decision_effects.csv',index=False)
    factorial=pd.read_csv(out/'factorial_interaction.csv').groupby(['dataset','metric']).interaction.agg(['mean','std']).reset_index()
    donor_rows=[]
    for d in ['Energy','SocialGood']:
        data,_=development_data(d,True)
        for seed in [2023,2024,2025]:
            for v in ['full','text_add']:
                traces=json.loads((base/f'runs/{d}_h6_s{seed}_{v}/text_interventions.json').read_text())
                valid=[r for r in traces if r['donor']>=0]
                identical=sum(data.texts[r['origin']]==data.texts[r['donor']] for r in valid)
                cosine=[]
                for r in valid:
                    a=data.embeddings[r['origin']];b=data.embeddings[r['donor']]
                    cosine.append(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
                donor_rows.append(dict(dataset=d,seed=seed,variant=v,eligible_origins=len(valid),identical_text_donors=identical,frozen_bert_cosine_mean=float(np.mean(cosine))))
    donor=pd.DataFrame(donor_rows);donor.to_csv(out/'donor_content_audit.csv',index=False)
    entropy=raw_pairs[(raw_pairs.candidate=='no_decisiveness')&(raw_pairs.control=='full')]
    mse_improved=int((entropy.delta_mse<0).sum());mae_improved=int((entropy.delta_mae<0).sum())
    probe_display=probes[probes.condition!='correct'][['dataset','variant','condition','delta_mse_mean','delta_mse_sd','prediction_divergence_mae','eligible_origins','total_origins']].copy()
    for col in ['delta_mse_mean','delta_mse_sd','prediction_divergence_mae']:probe_display[col]=probe_display[col].map(lambda x:f'{x:.3e}')
    findings=f'''# iAdapter 第一轮 review 开发实验

## 完成与实验边界

2 个 Economy sanity 通过；42/42 个正式开发流完成；6 个正式骨干；48 个同状态文本诊断序列。正式单元耗时求和 {proof['total_cell_seconds']/60:.2f} 分钟，包含预热/诊断和部分骨干训练，不等同于纯训练 GPU 时间或单次在线延迟。实验源提交 `{proof['experiment_commit']}`。

Energy/SocialGood、H6、每设置3个种子；前60%训练，60%–70%选择骨干，70%–80%在线开发比较。旧80%–100%不参与本轮评分。这个开发区间并非独立新测试集，绝对MSE不得与旧190次测试主表直接比较。原预测和目标经原始CSV核验；未使用模型输出作为真值。

## 主结果

{md_table(display)}

## 可以直接回应组件意见的发现

1. 去掉熵正则，两域平均MSE/MAE都降低；在6个配对单元中，MSE改善 {mse_improved}/6，MAE改善 {mae_improved}/6。该干预贯穿训练流预热与在线开发更新，不单独定位是哪一阶段起作用。不能据此宣称所有数据和horizon都应删除正则。
2. 单独删除路由文本梯度，两域平均MSE都上升。它与旧测试区间中的结果方向不同，应并列披露，不能据此确定差异由划分、初始化或其他变化中的哪一项导致。
3. 两者同时删除也优于完整模型；但它不在两个域都优于仅删除熵正则。组件作用存在条件依赖，不能简单把各单独消融收益相加。
4. 单MLP在Energy改善，SocialGood平均MSE略退化但MAE改善；相对完整模型减少约36%可训练参数。它是结构和容量共同改变的简洁性控制，不是参数匹配的“MoE必然无用”证明。

以下百分比为各方法原始MSE均值之比，与逐seed百分比均值不同；负值有利于候选。

{md_table(effect.round(6))}

### 2×2交互

交互=both-off−route-off−entropy-off+full，在每seed计算，再汇总均值和样本SD。n=3，不作显著性交互结论。

{md_table(factorial.round(6))}

## 当前文本内容干预

下表 ΔMSE=控制输入−正确文本，正值才表示正确内容更好；divergence是与正确文本预测的平均绝对差，单位为标准化目标。所有比较共用当时的模型、数值输入、反馈状态和目标。每个Energy流143个原点、SocialGood流85个原点，全部有合法历史长度匹配donor。

{md_table(probe_display)}

替换为训练均值或历史文本时预测变化很小，当前证据没有显示正确文本在两个域的full/text_add上稳定优于这些控制。{int(donor.identical_text_donors.sum())} 个donor与参考文本完全相同；所选历史文本确实不同，不能用“只换了同一条文本”解释本次小变化。但模型可能压缩/忽略内容，也可能是当前文本只提供有限增量信息，本试验不单独定位根因。

置零当前文本会产生更大变化，且Energy和SocialGood的误差方向相反。置零同时改变当前可用性和分支激活，不能作为语义利用的独立证据。全部干预保留历史文本反馈，因此本结论只关于给定历史状态下的当前文本内容，不等同于移除全历史文本。

## 对下一步的决定

- 保留“无熵正则”和“单MLP”作为进一步验证候选；开发结果不覆盖原full定义或原190次结果。
- 暂不依据旧测试单独删除路由文本梯度；组合效果和跨区间不一致必须报告。
- C1 没有达到“正确内容在两个域稳定优于均值与历史donor”的门槛，因此不自动启动更复杂的可信度网络或文本残差模块，也不把渠道消融当作已证明必要。
- 下一项低成本工作优先验证候选的实际在线成本与文本投影容量；原生外部基线、publication time和独立确认仍需另行补齐。

## Review覆盖

ECmL W3/F3与LCo1 W2/P4：增加了路由文本梯度×熵项交互与单MLP控制，给出正负结果。LCo1 W4：增加过去文本/均值/零输入的同状态干预，限制语义解释。W1/P3：完整保留三seed差值与波动，不宣称显著或普遍领先。公平外部基线、真实文本发布时间、全部专家单独必要性未被本轮解决。
'''
    (out/'REVIEW_FINDINGS.md').write_text(findings,encoding='utf-8')
    lines=[]
    for d in ['Energy','SocialGood']:
        b=summary[summary.dataset==d].set_index('variant')
        lines.append(f"{d}: full MSE {b.loc['full','mse_mean']:.6f}; no-entropy {b.loc['no_decisiveness','mse_mean']:.6f}; no-routing-text-gradient {b.loc['no_route_text_gradient','mse_mean']:.6f}; both removed {b.loc['no_route_no_decisiveness','mse_mean']:.6f}.")
    reply=f'''# Additional reviewer-response draft: development experiments

This text describes new exploratory development evidence, not a recovered original table or an independent held-out confirmation. All statements below are conditional on Energy/SocialGood, H=6 and three seeds. The chronological train/checkpoint-selection/development intervals are 60%/10%/10%; the remaining old test interval is not evaluated here.

## ECmL W3 / Feedback 3; LCo1 W2 / P4 — isolated components

We added a matched 2×2 study of routing text-gradient conditioning and the decisiveness entropy penalty, sharing each seed-specific backbone and the initial common parameters. We also evaluated a single-MLP adapter. Removing entropy reduced mean MSE and MAE on both development domains, with MSE reductions in {mse_improved}/6 matched runs and MAE reductions in {mae_improved}/6. Removing routing text gradients alone increased mean MSE on both domains, whereas removing both terms reduced mean MSE relative to the full model. We report the paired differences and interaction rather than assume independent or universally beneficial component effects.

{' '.join(lines)}

These observations differ from the earlier routing-gradient ablation on the old test interval. We retain both outcomes and do not attribute that difference to a single cause without a controlled follow-up. The single-MLP model reduces trainable parameters but changes both structure and capacity, so it does not establish an isolated causal role for each expert. Evidence: main_summary.csv, paired_runs.csv, paired_summary.csv, factorial_interaction.csv.

## LCo1 W4 — usefulness of textual content

At each development forecast origin, we held model weights, numerical input and historical feedback fixed and replaced the current text representation with a training-prefix mean, a token-length-matched earlier text, or a zero representation. Diagnostic forwards do not update the reference model or alter its subsequent trajectory. The correct-text and replacement predictions use identical original target masks. All donor records precede their query origins and none has identical cleaned text in this experiment.

Replacing content with a mean or historical text produces small prediction changes and does not consistently worsen error. Zeroing the current representation has a larger, domain-dependent effect, but this also changes availability and is not sufficient evidence of semantic utilization. We therefore do not present branch activity or a routing plot as proof that the model extracts useful current-text semantics. Historical text feedback is held fixed, so this intervention is not a removal of all past textual information. Evidence: text_probe_runs.csv, text_probe_summary.csv, donor_content_audit.csv and saved same-origin intervention predictions.

## ECmL W1 / Feedback 1; LCo1 W1 / P3 — strength of gains

We report all three seeds and sample standard deviations of matched differences. These repeats measure optimization variability conditional on the fixed development streams; overlapping windows are not independent repeats. We make no statistical-significance or broad SOTA claim. Engineering candidates selected from these results still require independent confirmation. Original GPT4MTS/VoT comparisons and true publication-time provenance remain unresolved.
'''
    (out/'REVIEW_RESPONSES.md').write_text(reply,encoding='utf-8')
    (out/'STATS.md').write_text('Comparisons are descriptive over three matched seeds per fixed domain. Shared backbone and shared compatible initial parameter tensors establish pairing. MSE/MAE aggregate original observed target elements; windows, horizons and variables are not independent repeats. The 70%-80% block is development data, not a new unseen test. Reported percentage changes use ratios of raw metric means, labelled explicitly. No p-values, confidence intervals or noninferiority guarantees are claimed. Counterfactual error deltas use common matched-donor origins; same-state probes are dependent measurements, not extra seeds. Cell wall time includes warm-up, optional diagnostic forwards and sometimes backbone fitting, so it is not an isolated latency comparison.\n')
    print('REVIEW_REPORT_WRITTEN',out,flush=True)

if __name__=='__main__':main()

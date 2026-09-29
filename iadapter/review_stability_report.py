"""Merge only the five predeclared candidates after independent cohort audits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ROOT
from .review_sprint import DEFAULT_RUN
from .review_sprint_analysis import analyze
from .review_stability import OUTPUT,METHODS,jobs,verify_training_unchanged
from .review_cost import DEFAULT_OUTPUT
from .analyze import md_table
from .util import write_json

def main():
    implementation=verify_training_unchanged()
    original=analyze(DEFAULT_RUN);extension=analyze(OUTPUT,jobs_override=jobs(),factorial=False)
    out=OUTPUT/'reports'
    parts=[];probe_parts=[]
    for label,base in [('initial_3_seeds',DEFAULT_RUN),('new_2_seeds',OUTPUT)]:
        frame=pd.read_csv(base/'reports/runs.csv');frame=frame[frame.variant.isin(METHODS)].copy();frame['cohort']=label
        frame['experiment_commit']=json.loads((base/'reports/validation.json').read_text())['experiment_commit'];parts.append(frame)
        probe=pd.read_csv(base/'reports/text_probe_runs.csv');probe['cohort']=label;probe_parts.append(probe)
    frame=pd.concat(parts,ignore_index=True);probe=pd.concat(probe_parts,ignore_index=True)
    expected={(d,s,v) for d in ['Energy','SocialGood'] for s in range(2023,2028) for v in METHODS}
    assert len(frame)==50 and set(zip(frame.dataset,frame.seed,frame.variant))==expected
    assert not frame.duplicated(['dataset','seed','variant']).any()
    frame.to_csv(out/'five_seed_runs.csv',index=False)
    summary=frame.groupby(['dataset','variant'],sort=False).agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'),parameters=('trainable_parameters','first')).reset_index()
    assert (summary.n==5).all();summary.to_csv(out/'five_seed_summary.csv',index=False)
    paired=[]
    for (d,s),block in frame.groupby(['dataset','seed']):
        b=block.set_index('variant')
        for candidate in METHODS:
            for control in ['full','numeric_add','text_add']:
                if candidate!=control:paired.append({'dataset':d,'seed':int(s),'candidate':candidate,'control':control,'delta_mse':float(b.loc[candidate,'mse']-b.loc[control,'mse']),'delta_mae':float(b.loc[candidate,'mae']-b.loc[control,'mae'])})
    paired=pd.DataFrame(paired);paired.to_csv(out/'five_seed_paired_runs.csv',index=False)
    ps=paired.groupby(['dataset','candidate','control'],sort=False).agg(n=('seed','count'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),delta_mae_mean=('delta_mae','mean'),delta_mae_sd=('delta_mae','std'),mse_improved_seeds=('delta_mse',lambda x:int((x<0).sum())),mae_improved_seeds=('delta_mae',lambda x:int((x<0).sum()))).reset_index()
    ps.to_csv(out/'five_seed_paired_summary.csv',index=False)
    assert len(probe)==80 and not probe.duplicated(['dataset','seed','variant','condition']).any()
    probe.to_csv(out/'five_seed_text_probe_runs.csv',index=False)
    text_summary=probe.groupby(['dataset','variant','condition'],sort=False).agg(n=('seed','count'),delta_mse_mean=('delta_mse_vs_correct','mean'),delta_mse_sd=('delta_mse_vs_correct','std'),delta_mae_mean=('delta_mae_vs_correct','mean'),prediction_divergence_mae=('prediction_divergence_mae','mean'),eligible_origins=('eligible_origins','min'),total_origins=('total_origins','first')).reset_index()
    assert (text_summary.n==5).all();text_summary.to_csv(out/'five_seed_text_probe_summary.csv',index=False)
    display=summary[['dataset','variant','n','parameters']].copy()
    display['MSE mean ± SD']=[f'{m:.6f} ± {s:.6f}' for m,s in zip(summary.mse_mean,summary.mse_sd)]
    display['MAE mean ± SD']=[f'{m:.6f} ± {s:.6f}' for m,s in zip(summary.mae_mean,summary.mae_sd)]
    decisions=[]
    for dataset,block in summary.groupby('dataset'):
        b=block.set_index('variant')
        for variant in ['no_decisiveness','single_mlp']:
            p=ps[(ps.dataset==dataset)&(ps.candidate==variant)&(ps.control=='full')].iloc[0]
            decisions.append({'dataset':dataset,'candidate':variant,'mse_change_percent':100*(b.loc[variant,'mse_mean']/b.loc['full','mse_mean']-1),'mae_change_percent':100*(b.loc[variant,'mae_mean']/b.loc['full','mae_mean']-1),'MSE_improved_seeds':int(p.mse_improved_seeds),'MAE_improved_seeds':int(p.mae_improved_seeds),'within_1_percent_mean_error_tolerance':bool(b.loc[variant,'mse_mean']<=1.01*b.loc['full','mse_mean'] and b.loc[variant,'mae_mean']<=1.01*b.loc['full','mae_mean'])})
    decisions=pd.DataFrame(decisions);decisions.to_csv(out/'candidate_decisions.csv',index=False)
    cost=pd.read_csv(DEFAULT_OUTPUT/'relative_to_full.csv')
    cost=cost[cost.variant.isin(['no_decisiveness','single_mlp'])]
    cost_status=json.loads((DEFAULT_OUTPUT/'status.json').read_text());assert cost_status['status']=='completed' and cost_status['cases']==30
    highlights=[]
    for _,r in decisions.iterrows():
        highlights.append(f"- {r.dataset} / {r.candidate}：相对full平均MSE变化 {r.mse_change_percent:+.2f}%，平均MAE变化 {r.mae_change_percent:+.2f}%；MSE改善 {r.MSE_improved_seeds}/5，MAE改善 {r.MAE_improved_seeds}/5。")
    limited=text_summary[text_summary.condition.isin(['train_mean','historical_donor'])].copy()
    for col in ['delta_mse_mean','delta_mse_sd','delta_mae_mean','prediction_divergence_mae']:limited[col]=limited[col].map(lambda x:f'{x:.3e}')
    report=f'''# 五种子稳定性与成本补充

## 完成与范围

新增20/20单元完成，4个新骨干；复用初始实验中五种候选对应的30单元，形成两域×五方法×五seed=50单元。新增32个同状态文本诊断序列，与原48个合并为80个。原始2×2路由/entropy交互仍只有3seed，本报告没有将其虚报为5seed。

两批训练模型/数据/执行器及run_cell代码经AST逐项比较相同；变化仅在队列入口和分析。两批分别从原始CSV重建目标、掩码和边界，全部通过。数据仍为前60%训练、60%–70%骨干选择、70%–80%开发评估；增加种子不恢复数据独立性，不构成新测试集确认。

## 五种子主表

{md_table(display)}

## 候选结论

{chr(10).join(highlights)}

百分比使用原始误差均值之比，不是逐seed百分比均值。1%容差仅是预先固定的开发筛选规则，不是统计非劣效保证。负值有利于候选，所有退化保留在表中。具体paired均值/SD见five_seed_paired_summary.csv。

## 独立在线成本

下表为30组隔离微基准的相对full变化，正值表示减少。参数采用实际计数；计时为同设备三轮中位数的中位数。每轮50次预热、200次测量，seed2023预热权重副本。没有将新增优化种子当作计时轮次。

{md_table(cost.round(2))}

无熵正则不减少参数；其微小或混合的时间变化应结合轮间范围解释。single_mlp保留相同文本投影但缩减专家/门控结构，缓存预测与更新的节省不能直接等同于含BERT的新文本端到端节省。完整P95、内存口径和轮间范围见profile_output/review_sprint_cost_20260929/REPORT.md。

## 五种子当前文本内容诊断

差值为控制文本减正确文本；正值才表示正确内容误差更低。每个原点的模型/数值/历史反馈固定，诊断不改变参考轨迹。

{md_table(limited)}

这些同状态干预针对当前文本，不是删除全部历史文本或证明原始文本不含预测信息。文本分支是否提供增益与其是否利用具体内容是不同问题；不能用置零造成的变化独立证明语义利用。

## 对review的更新

- ECmL W3/F3、LCo1 W2/P4：候选的组件删减结果现在有5seed，显示是否维持原3seed方向；不宣称所有模块都必要。
- ECmL W4/F4：补充候选的参数、独立预测/更新/新文本成本和内存；正式预测状态未改变。
- LCo1 W4：文本内容敏感性复核扩大到5seed，仍报告控制条件和边界。
- W1/P3：报告全部种子及配对波动，不以窗口数放大统计样本量。
- 原生GPT4MTS/VoT公平比较、真实publication time和独立数据确认仍未由本轮解决。
'''
    (out/'STABILITY_AND_COST.md').write_text(report,encoding='utf-8')
    write_json(out/'combined_validation.json',{'status':'passed','initial_cohort_commit':original['experiment_commit'],'extension_commit':extension['experiment_commit'],'new_runs':20,'combined_candidate_runs':50,'seeds':[2023,2024,2025,2026,2027],'content_probe_sequences':80,'factorial_design_seeds_remain':3,'cost_round_cases':30,'scope':'exploratory development, not independent test confirmation',**implementation})
    print('FIVE_SEED_REPORT_VALIDATED',out,flush=True)

if __name__=='__main__':main()

"""Write the fixed compression study without selecting favorable runs."""
import json
import pandas as pd
from .review_compression import OUTPUT
from .review_compression_cost import COST_OUTPUT
from .analyze import md_table

def main():
    out=OUTPUT/'reports'
    proof=json.loads((out/'validation.json').read_text())
    status=json.loads((COST_OUTPUT/'status.json').read_text())
    assert proof['status']=='passed' and proof['combined_cells']==24
    assert status['status']=='completed' and status['cases']==24 and status['scientific_artifacts_unchanged']
    summary=pd.read_csv(out/'summary.csv');decision=pd.read_csv(out/'decisions.csv')
    reference=pd.read_csv(out/'simple_reference_summary.csv');pairs=pd.read_csv(out/'paired_summary.csv')
    cost=pd.read_csv(COST_OUTPUT/'relative_to_E0.csv')
    merged=decision.merge(cost.rename(columns={'variant':'setting'}),on=['dataset','setting'],validate='one_to_one',suffixes=('_accuracy','_timing'))
    assert len(merged)==6
    assert ((merged.parameter_reduction_percent_accuracy-merged.parameter_reduction_percent_timing).abs()<1e-8).all()
    # Report both timings; do not turn a small speed difference into a significance claim.
    merged['prediction_or_update_reduction_at_least_20_percent']=(merged.prediction_cached_latency_reduction_percent>=20)|(merged.update_fixed_record_latency_reduction_percent>=20)
    merged['other_median_not_increased']=(merged.prediction_cached_latency_reduction_percent>=0)&(merged.update_fixed_record_latency_reduction_percent>=0)
    merged.to_csv(out/'accuracy_cost_join.csv',index=False)
    passed=decision.groupby('setting').accuracy_parameter_gate_pass.all()
    qualifying=passed[passed].index.tolist()
    decision_text='没有候选同时通过两域的精度与参数门槛，因此不进入五 seed 扩展或独立确认，也不追加投影宽度搜索。' if not qualifying else '两域精度与参数门槛通过设置：'+','.join(qualifying)+'；是否达到成本门槛仍需结合时延和轮间范围判断，不能直接声称独立确认。'
    display=summary[['dataset','setting','n','parameters']].copy()
    display['MSE mean ± SD']=[f'{m:.6f} ± {s:.6f}' for m,s in zip(summary.mse_mean,summary.mse_sd)]
    display['MAE mean ± SD']=[f'{m:.6f} ± {s:.6f}' for m,s in zip(summary.mae_mean,summary.mae_sd)]
    projection=pairs[((pairs.candidate=='E1')&(pairs.control=='E0'))|((pairs.candidate=='E3')&(pairs.control=='E2'))]
    (out/'REVIEW_FINDINGS.md').write_text(f'''# E-01：投影压缩的精度与成本取舍

## 决定

{decision_text}

本轮完成12个新增开发运行，复用12个原控制，共24个设置单元；另有两次sanity。成本是另外24个隔离计时单元，不计作预测性能实验。科学执行器、数据、骨干训练代码没有改动，所有原始目标/掩码/成熟事件检查通过。详见validation.json。

## 设置与全部结果

E0/E1：五专家，关闭路由文本梯度与entropy，投影中间宽96/8。E2/E3：单MLP、无gate/entropy，投影中间宽96/8。所有设置保留64维输出、相同激活/dropout/LayerNorm及专家侧文本反馈。E0是该路线固定的精简五专家基准，不是原full。

Energy/SocialGood、H6、seeds2023/2024/2025；前60%训练、60%–70%骨干选择、70%–80%开发评估。六个旧骨干和兼容形状初值逐张量匹配；改变形状的投影权重仅共享初始化规则，不能宣称参数完全相同。

{md_table(display)}

同三seed的简单参考如下，不能用之前五seed均值替代此表：

{md_table(reference.round(6))}

## 预先固定门槛

相对E0的平均MSE/MAE退化均≤1%，参数减少≥40%，之后再看缓存预测或更新中位数减少≥20%及另一项是否明显增加。下表是第一步精度/参数门槛；百分比为误差均值之比。

{md_table(decision.round(3))}

这是工程筛选阈值，不是统计非劣效边界或显著性检验。严格执行门槛不等于证明模型在所有任务不可用。不能因为某个退化只略大于1%就事后放宽阈值。

## 隔离投影压缩的效应

差值为候选减控制，负值有利。E1−E0只压缩五专家模型的投影；E3−E2只压缩单MLP模型的投影。这些对照比只拿最小模型与完整模型比较更能区分改动来源。

{md_table(projection.round(6))}

## 成本

以下所有比值由同一次24单元测量中的E0计算，正值表示减少。两域×四设置×三轮；每轮50预热/200测量，seed2023预热权重隔离副本。正式预测未改变。

{md_table(cost.round(2))}

完整原始毫秒、P95、轮间范围与缓存/显存见cost报告。参数降低与时延降低是不同结论；小参数模型未必使小批量GPU执行明显加速。缓存预测不含BERT，新文本预测包含tokenizer与完整12层BERT；不能把前者的降幅直接用于整个系统。微基准不包括队列调度、日志和磁盘写入。

## 审稿回复边界

- ECmL F4：现在可以量化投影与专家结构分别贡献的参数/延迟成本；不继续只凭冻结骨干称整个适配器相对骨干轻量。
- ECmL F3、LCo1 P4：既展示可以删减的结构，也展示压缩的精度代价；没有把全部组件宣称为必要。
- ECmL F1、LCo1 P3：所有seed与退化保留。三个seed估计固定开发流上的优化波动，不是三个独立数据集。
- 这次实验没有解决原生GPT4MTS/VoT、真实publication time或独立泛化确认，也没有新增文本语义证据。

## 下一步

本轮E路线在固定门槛处收束。更高优先级转到A路线：利用已导入的源码明确共同任务、原生与改编实现边界，准备可以运行的外部公平对照。D额外延迟仍是条件性研究方向，不能为追求正结果自动追加72次实验。

## 证据与复现

训练提交：{proof['experiment_commit']}。原始预测及事件位于experiments/review_compression_20260929/runs；复用控制位于experiments/review_sprint_20260929/runs。分析：python -m iadapter.review_compression_analysis；写报告：python -m iadapter.review_compression_write。项目根目录/home/wuwa/projects/research/iadapter，使用/home/wuwa/miniconda3/envs/llm/bin/python。报告生成不启动训练。
''',encoding='utf-8')
    print('COMPRESSION_REPORT_WRITTEN',out,flush=True)

if __name__=='__main__':main()

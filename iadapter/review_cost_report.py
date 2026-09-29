"""Descriptive round-aware cost tables; no latency significance claims."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .review_cost import DEFAULT_OUTPUT,METHODS
from .review_sprint import DEFAULT_RUN
from .analyze import md_table

PHASES=['prediction_cached','update_fixed_record','prediction_fresh_text']

def main(path):
    path=Path(path);status=json.loads((path/'status.json').read_text());config=json.loads((path/'config.json').read_text());raw=json.loads((path/'raw.json').read_text())
    assert status['status']=='completed' and status['scientific_artifacts_unchanged']
    expected={(d,v,r) for d in config['datasets'] for v in config['variants'] for r in range(config['rounds'])}
    assert len(raw)==len(expected) and {(x['dataset'],x['variant'],x['round']) for x in raw}==expected
    rows=[]
    for case in raw:
        assert case['warmup']==config['warmup'] and case['repeats']==config['repeats']
        row={k:case[k] for k in ['dataset','variant','round','origin','trainable_parameters','backbone_parameters','frozen_text_parameters','cpu_embedding_cache_bytes','replay_queue_peak_bytes','gradient_cache_bytes','original_development_peak_allocated_bytes','cached_path_peak_allocated_bytes','fresh_path_peak_allocated_bytes']}
        for phase in PHASES:
            values=np.asarray(case[phase]['sample_ms']);assert len(values)==config['repeats'] and np.isfinite(values).all() and (values>0).all()
            assert np.isclose(np.median(values),case[phase]['median_ms'],rtol=0,atol=1e-12)
            assert np.isclose(np.quantile(values,.95),case[phase]['p95_ms'],rtol=0,atol=1e-12)
            row[phase+'_median_ms']=float(np.median(values));row[phase+'_p95_ms']=float(np.quantile(values,.95))
        rows.append(row)
    rounds=pd.DataFrame(rows);rounds.to_csv(path/'rounds.csv',index=False)
    summary=[]
    for (d,v),block in rounds.groupby(['dataset','variant'],sort=False):
        row={'dataset':d,'variant':v,'rounds':len(block)}
        for key in ['trainable_parameters','backbone_parameters','frozen_text_parameters','cpu_embedding_cache_bytes','replay_queue_peak_bytes','gradient_cache_bytes','original_development_peak_allocated_bytes']:
            assert block[key].nunique()==1;row[key]=int(block[key].iloc[0])
        row['cached_path_peak_MiB']=float(block.cached_path_peak_allocated_bytes.max()/2**20)
        row['fresh_path_peak_MiB']=float(block.fresh_path_peak_allocated_bytes.max()/2**20)
        for phase in PHASES:
            med=block[phase+'_median_ms'];p95=block[phase+'_p95_ms']
            row[phase+'_median_ms']=float(med.median());row[phase+'_round_min_ms']=float(med.min());row[phase+'_round_max_ms']=float(med.max());row[phase+'_p95_ms']=float(p95.median())
        summary.append(row)
    summary=pd.DataFrame(summary);summary.to_csv(path/'summary.csv',index=False)
    relative=[]
    for d,block in summary.groupby('dataset'):
        ref=block.set_index('variant').loc['full']
        for _,r in block.iterrows():
            relative.append({'dataset':d,'variant':r.variant,'parameter_reduction_percent':100*(1-r.trainable_parameters/ref.trainable_parameters),**{phase+'_latency_reduction_percent':100*(1-r[phase+'_median_ms']/ref[phase+'_median_ms']) for phase in PHASES}})
    relative=pd.DataFrame(relative);relative.to_csv(path/'relative_to_full.csv',index=False)
    display=summary[['dataset','variant','trainable_parameters','backbone_parameters']].copy()
    for phase in PHASES:
        display[phase+' ms (range)']=[f'{r[phase+"_median_ms"]:.3f} ({r[phase+"_round_min_ms"]:.3f}–{r[phase+"_round_max_ms"]:.3f})' for _,r in summary.iterrows()]
    memory=summary[['dataset','variant','cached_path_peak_MiB','fresh_path_peak_MiB','cpu_embedding_cache_bytes','replay_queue_peak_bytes','gradient_cache_bytes']].copy()
    p95=summary[['dataset','variant']+[phase+'_p95_ms' for phase in PHASES]]
    text=f'''# 候选在线成本：独立微基准

硬件：{config['hardware']}；PyTorch {config['torch']}。两个域、五方法、seed2023预热权重副本；每方法{config['rounds']}轮、每轮{config['warmup']}次预热/{config['repeats']}次测量。统计量为轮内中位数的中位数，括号是各轮中位数范围，单位毫秒。不同轮按固定seed轮换方法顺序，不能将600个调用视为600个独立科研重复。

## 参数与计时

{md_table(display)}

参数与测量使用同一实际模型实现。full和no_decisiveness参数相同，移除正则并不减少模型参数。缓存预测不含BERT；新文本预测包含tokenizer、完整冻结12层BERT、CPU缓存格式转换、投影和数值预测。有限性检查与设备同步纳入计时。

## P95

以下是各轮P95的中位数，不是将所有调用合并后的P95。

{md_table(p95.round(3))}

## 内存

{md_table(memory.round(3))}

GPU列为本进程PyTorch峰值allocated，非整卡占用或reserved。缓存路径峰值覆盖预测与更新，包括Adam状态；fresh路径在恢复预热权重、移除微基准Adam状态后计量，包含BERT。微基准只放一个活动窗口/文本向量，原在线执行器的全流GPU张量不全部计入这里；原开发运行峰值保存在summary.csv的original_development_peak_allocated_bytes。CPU缓存、队列及梯度字节不代表完整Python进程内存，队列数字不含对象管理开销。

## 相对完整模型

正值表示参数/延迟减少；仅比较同域、同阶段，基准为三轮中位数的中位数。小幅差异若与轮间波动相近，不认定为稳定加速。

{md_table(relative.round(2))}

## 测量边界

固定第一个开发原点，使用真实成熟记录作重复更新微基准。每轮恢复相同预热权重；新文本阶段再次恢复，避免用反复优化后的局部权重测量。更新包含task loss、反馈梯度探针、熵项（若启用）、梯度裁剪、Adam和反馈缓存刷新；不含队列调度、日志和磁盘写入。微基准中的重复更新仅用于计时，不产生新的性能结果。正式42次实验文件和queue均未改变。

## 计量代码变更

| 文件 | 变更 | 内容 |
|---|---|---|
| iadapter/review_cost.py | 新增 | 隔离权重、三轮顺序轮换、三个计时阶段、参数/内存记录、正式文件不变性核对 |
| iadapter/review_cost_report.py | 新增 | 重算原始逐次时长的中位数/P95并生成对照表 |
| refine-logs/review_sprint/COST_PLAN.md | 新增 | 运行前固定的测量口径 |
| .gitignore | 修改 | profile_output/作为本地产物目录 |

这些独立脚本不修改模型训练逻辑，后续可以保留用于重测；删除profile_output只会删除计量产物，不影响科学运行，但本次保留全部原始计时。
'''
    (path/'REPORT.md').write_text(text,encoding='utf-8')
    print('COST_REPORT_VALIDATED',len(raw),'round-cases',path,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();main(a.input)

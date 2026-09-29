# E-01：投影压缩的精度与成本取舍

## 决定

没有候选同时通过两域的精度与参数门槛，因此不进入五 seed 扩展或独立确认，也不追加投影宽度搜索。

本轮完成12个新增开发运行，复用12个原控制，共24个设置单元；另有两次sanity。成本是另外24个隔离计时单元，不计作预测性能实验。科学执行器、数据、骨干训练代码没有改动，所有原始目标/掩码/成熟事件检查通过。详见validation.json。

## 设置与全部结果

E0/E1：五专家，关闭路由文本梯度与entropy，投影中间宽96/8。E2/E3：单MLP、无gate/entropy，投影中间宽96/8。所有设置保留64维输出、相同激活/dropout/LayerNorm及专家侧文本反馈。E0是该路线固定的精简五专家基准，不是原full。

Energy/SocialGood、H6、seeds2023/2024/2025；前60%训练、60%–70%骨干选择、70%–80%开发评估。六个旧骨干和兼容形状初值逐张量匹配；改变形状的投影权重仅共享初始化规则，不能宣称参数完全相同。

| dataset | setting | n | parameters | MSE mean ± SD | MAE mean ± SD |
|---|---|---|---|---|---|
| Energy | E0 | 3 | 167088 | 0.016856 ± 0.000617 | 0.098740 ± 0.001606 |
| Energy | E1 | 3 | 93784 | 0.017255 ± 0.001374 | 0.099778 ± 0.003708 |
| Energy | E2 | 3 | 107488 | 0.016636 ± 0.000977 | 0.097731 ± 0.003795 |
| Energy | E3 | 3 | 34184 | 0.016978 ± 0.000652 | 0.099394 ± 0.002340 |
| SocialGood | E0 | 3 | 166832 | 0.188309 ± 0.050573 | 0.247022 ± 0.035284 |
| SocialGood | E1 | 3 | 93528 | 0.188558 ± 0.050827 | 0.246866 ± 0.035743 |
| SocialGood | E2 | 3 | 107232 | 0.190786 ± 0.051439 | 0.248354 ± 0.036144 |
| SocialGood | E3 | 3 | 33928 | 0.190794 ± 0.051539 | 0.248495 ± 0.036193 |

同三seed的简单参考如下，不能用之前五seed均值替代此表：

| dataset | variant | n | mse_mean | mse_sd | mae_mean | mae_sd | parameters |
|---|---|---|---|---|---|---|---|
| Energy | numeric_add | 3 | 0.018144 | 0.002508 | 0.101379 | 0.006121 | 10752 |
| Energy | text_add | 3 | 0.017187 | 0.000774 | 0.099851 | 0.002366 | 95136 |
| SocialGood | numeric_add | 3 | 0.191805 | 0.050082 | 0.249476 | 0.034284 | 10496 |
| SocialGood | text_add | 3 | 0.18995 | 0.050784 | 0.248002 | 0.035241 | 94880 |

## 预先固定门槛

相对E0的平均MSE/MAE退化均≤1%，参数减少≥40%，之后再看缓存预测或更新中位数减少≥20%及另一项是否明显增加。下表是第一步精度/参数门槛；百分比为误差均值之比。

| dataset | setting | mse_change_percent | mae_change_percent | parameter_reduction_percent | accuracy_parameter_gate_pass |
|---|---|---|---|---|---|
| Energy | E1 | 2.369 | 1.051 | 43.871 | False |
| Energy | E2 | -1.305 | -1.022 | 35.67 | False |
| Energy | E3 | 0.724 | 0.663 | 79.541 | True |
| SocialGood | E1 | 0.132 | -0.063 | 43.939 | True |
| SocialGood | E2 | 1.316 | 0.539 | 35.725 | False |
| SocialGood | E3 | 1.32 | 0.596 | 79.663 | False |

这是工程筛选阈值，不是统计非劣效边界或显著性检验。严格执行门槛不等于证明模型在所有任务不可用。不能因为某个退化只略大于1%就事后放宽阈值。

## 隔离投影压缩的效应

差值为候选减控制，负值有利。E1−E0只压缩五专家模型的投影；E3−E2只压缩单MLP模型的投影。这些对照比只拿最小模型与完整模型比较更能区分改动来源。

| dataset | candidate | control | n | delta_mse_mean | delta_mse_sd | delta_mae_mean | delta_mae_sd | mse_improved_seeds | mae_improved_seeds |
|---|---|---|---|---|---|---|---|---|---|
| Energy | E1 | E0 | 3 | 0.000399 | 0.000871 | 0.001037 | 0.002122 | 2 | 1 |
| Energy | E3 | E2 | 3 | 0.000342 | 0.000418 | 0.001663 | 0.001772 | 0 | 1 |
| SocialGood | E1 | E0 | 3 | 0.000249 | 0.000778 | -0.000156 | 0.00081 | 2 | 2 |
| SocialGood | E3 | E2 | 3 | 8e-06 | 0.000296 | 0.000141 | 0.000242 | 2 | 1 |

## 成本

以下所有比值由同一次24单元测量中的E0计算，正值表示减少。两域×四设置×三轮；每轮50预热/200测量，seed2023预热权重隔离副本。正式预测未改变。

| dataset | variant | parameter_reduction_percent | prediction_cached_latency_reduction_percent | update_fixed_record_latency_reduction_percent | prediction_fresh_text_latency_reduction_percent |
|---|---|---|---|---|---|
| Energy | E2 | 35.67 | 33.42 | 44.12 | 11.75 |
| Energy | E3 | 79.54 | 35.02 | 35.35 | 15.52 |
| Energy | E0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Energy | E1 | 43.87 | 2.85 | -22.75 | 0.47 |
| SocialGood | E3 | 79.66 | 35.23 | 58.21 | 10.62 |
| SocialGood | E2 | 35.72 | 36.69 | 56.57 | 10.23 |
| SocialGood | E1 | 43.94 | 3.4 | -0.32 | -3.36 |
| SocialGood | E0 | 0.0 | 0.0 | 0.0 | 0.0 |

完整原始毫秒、P95、轮间范围与缓存/显存见cost报告。参数降低与时延降低是不同结论；小参数模型未必使小批量GPU执行明显加速。缓存预测不含BERT，新文本预测包含tokenizer与完整12层BERT；不能把前者的降幅直接用于整个系统。微基准不包括队列调度、日志和磁盘写入。

## 审稿回复边界

- ECmL F4：现在可以量化投影与专家结构分别贡献的参数/延迟成本；不继续只凭冻结骨干称整个适配器相对骨干轻量。
- ECmL F3、LCo1 P4：既展示可以删减的结构，也展示压缩的精度代价；没有把全部组件宣称为必要。
- ECmL F1、LCo1 P3：所有seed与退化保留。三个seed估计固定开发流上的优化波动，不是三个独立数据集。
- 这次实验没有解决原生GPT4MTS/VoT、真实publication time或独立泛化确认，也没有新增文本语义证据。

## 下一步

本轮E路线在固定门槛处收束。更高优先级转到A路线：利用已导入的源码明确共同任务、原生与改编实现边界，准备可以运行的外部公平对照。D额外延迟仍是条件性研究方向，不能为追求正结果自动追加72次实验。

## 证据与复现

训练提交：354090c19527ba337507f7a1397cc05cdba916da。原始预测及事件位于experiments/review_compression_20260929/runs；复用控制位于experiments/review_sprint_20260929/runs。分析：python -m iadapter.review_compression_analysis；写报告：python -m iadapter.review_compression_write。项目根目录/home/wuwa/projects/research/iadapter，使用/home/wuwa/miniconda3/envs/llm/bin/python。报告生成不启动训练。

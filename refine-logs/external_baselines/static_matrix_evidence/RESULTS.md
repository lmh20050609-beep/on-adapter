# 五域五seed共同OT静态对照

75/75单元完成：72新增、3个Economy/2023 pilot复用。25组均从原CSV核验目标/掩码/尺度/文本ID、选模边界与评分无更新；缺失或失败单元不进入此表。H6；Energy L48，其余L36。前60%训练、60%–70%选模、70%–80%开发评分。

## 全部结果

| dataset | method | n | origins | valid_elements | MSE mean ± SD | MAE mean ± SD |
|---|---|---|---|---|---|---|
| Economy | numeric_static | 5 | 37 | 222 | 0.087523 ± 0.015701 | 0.235855 ± 0.024382 |
| Economy | iadapter_static | 5 | 37 | 222 | 0.105313 ± 0.023761 | 0.264798 ± 0.037070 |
| Economy | gpt4mts_static | 5 | 37 | 222 | 0.061619 ± 0.011277 | 0.198772 ± 0.018977 |
| Energy | numeric_static | 5 | 143 | 858 | 0.013982 ± 0.000689 | 0.090592 ± 0.002622 |
| Energy | iadapter_static | 5 | 143 | 858 | 0.027068 ± 0.014423 | 0.120500 ± 0.029080 |
| Energy | gpt4mts_static | 5 | 143 | 858 | 0.015348 ± 0.001360 | 0.092643 ± 0.005802 |
| Security | numeric_static | 5 | 25 | 150 | 2.750218 ± 0.096165 | 1.218393 ± 0.013255 |
| Security | iadapter_static | 5 | 25 | 150 | 2.718436 ± 0.097310 | 1.215638 ± 0.008398 |
| Security | gpt4mts_static | 5 | 25 | 150 | 2.705181 ± 0.119980 | 1.226716 ± 0.014830 |
| SocialGood | numeric_static | 5 | 85 | 510 | 0.146024 ± 0.032361 | 0.221556 ± 0.021961 |
| SocialGood | iadapter_static | 5 | 85 | 510 | 0.149492 ± 0.033051 | 0.223820 ± 0.021836 |
| SocialGood | gpt4mts_static | 5 | 85 | 510 | 0.097411 ± 0.014844 | 0.196118 ± 0.014814 |
| Climate | numeric_static | 5 | 44 | 264 | 0.999289 ± 0.012638 | 0.777723 ± 0.007251 |
| Climate | iadapter_static | 5 | 44 | 264 | 1.006596 ± 0.013282 | 0.781827 ± 0.007056 |
| Climate | gpt4mts_static | 5 | 44 | 264 | 1.046567 ± 0.056300 | 0.789762 ± 0.032790 |

## iAdapter相对控制

百分比为平均误差之比减1，正值为退化。改善seed数均以5为分母。

| dataset | control | mse_change_percent | mae_change_percent | mse_improved_seeds | mae_improved_seeds |
|---|---|---|---|---|---|
| Climate | numeric_static | 0.731 | 0.528 | 0 | 0 |
| Climate | gpt4mts_static | -3.819 | -1.005 | 4 | 1 |
| Economy | numeric_static | 20.326 | 12.271 | 1 | 1 |
| Economy | gpt4mts_static | 70.91 | 33.217 | 0 | 1 |
| Energy | numeric_static | 93.598 | 33.014 | 1 | 1 |
| Energy | gpt4mts_static | 76.367 | 30.069 | 2 | 0 |
| Security | numeric_static | -1.156 | -0.226 | 4 | 3 |
| Security | gpt4mts_static | 0.49 | -0.903 | 2 | 4 |
| SocialGood | numeric_static | 2.375 | 1.022 | 0 | 0 |
| SocialGood | gpt4mts_static | 53.465 | 14.125 | 0 | 0 |

## 比较边界

- 各域三个方法在相同原点/目标/有效掩码上评分，开发期没有更新。内部两方法共享每seed的单变量骨干，预热后将反馈清零并保持零；GPT4MTS保留作者模型和冻结规则。
- 共同可见性使用end_date代理，真实发布时间没有验证。iAdapter读取当前文本mean/64，GPT4MTS消费历史文本CLS/512序列；文本信息量、表示与训练阶段不相同，所以不是结构单因素试验。
- seed只是同一固定流上的优化重复；内部方法共享骨干，而跨架构没有相同初始参数保证。差值按相同seed标签和评分流对齐，仅作描述性统计。重叠目标不能当独立重复，不给出显著性或独立泛化结论。
- 本表为静态OT任务，不能与原多变量在线表合并，不能由它推出在线机制的因果贡献。VoT没有可核验的生成/原点时间记录，仍未列入；75单元不是原四方法100单元的完成声明。
- 原始预测和全部训练历史仍在各source目录；runs.csv列明每个单元来源与提交号，paired_runs.csv保留全部逐seed差值。计时包含不同缓存命中和共享训练机会，仅记录执行成本，不能替代独立效率测量。

# 五种子稳定性与成本补充

## 完成与范围

新增20/20单元完成，4个新骨干；复用初始实验中五种候选对应的30单元，形成两域×五方法×五seed=50单元。新增32个同状态文本诊断序列，与原48个合并为80个。原始2×2路由/entropy交互仍只有3seed，本报告没有将其虚报为5seed。

两批训练模型/数据/执行器及run_cell代码经AST逐项比较相同；变化仅在队列入口和分析。两批分别从原始CSV重建目标、掩码和边界，全部通过。数据仍为前60%训练、60%–70%骨干选择、70%–80%开发评估；增加种子不恢复数据独立性，不构成新测试集确认。

## 五种子主表

| dataset | variant | n | parameters | MSE mean ± SD | MAE mean ± SD |
|---|---|---|---|---|---|
| Energy | numeric_add | 5 | 10752 | 0.017652 ± 0.002189 | 0.100134 ± 0.005011 |
| Energy | text_add | 5 | 95136 | 0.016758 ± 0.001075 | 0.098705 ± 0.002732 |
| Energy | full | 5 | 167088 | 0.017548 ± 0.002086 | 0.100779 ± 0.006356 |
| Energy | no_decisiveness | 5 | 167088 | 0.016540 ± 0.001113 | 0.097981 ± 0.003031 |
| Energy | single_mlp | 5 | 107488 | 0.016621 ± 0.001134 | 0.097899 ± 0.003157 |
| SocialGood | numeric_add | 5 | 10496 | 0.180540 ± 0.038751 | 0.242942 ± 0.026259 |
| SocialGood | text_add | 5 | 94880 | 0.179552 ± 0.038729 | 0.242151 ± 0.026566 |
| SocialGood | full | 5 | 166832 | 0.180220 ± 0.037464 | 0.245852 ± 0.026401 |
| SocialGood | no_decisiveness | 5 | 166832 | 0.178362 ± 0.038148 | 0.241742 ± 0.025941 |
| SocialGood | single_mlp | 5 | 107232 | 0.179561 ± 0.039552 | 0.242161 ± 0.027228 |

## 候选结论

- Energy / no_decisiveness：相对full平均MSE变化 -5.74%，平均MAE变化 -2.78%；MSE改善 3/5，MAE改善 2/5。
- Energy / single_mlp：相对full平均MSE变化 -5.28%，平均MAE变化 -2.86%；MSE改善 3/5，MAE改善 3/5。
- SocialGood / no_decisiveness：相对full平均MSE变化 -1.03%，平均MAE变化 -1.67%；MSE改善 5/5，MAE改善 5/5。
- SocialGood / single_mlp：相对full平均MSE变化 -0.37%，平均MAE变化 -1.50%；MSE改善 3/5，MAE改善 5/5。

百分比使用原始误差均值之比，不是逐seed百分比均值。1%容差仅是预先固定的开发筛选规则，不是统计非劣效保证。负值有利于候选，所有退化保留在表中。具体paired均值/SD见five_seed_paired_summary.csv。

## 独立在线成本

下表为30组隔离微基准的相对full变化，正值表示减少。参数采用实际计数；计时为同设备三轮中位数的中位数。每轮50次预热、200次测量，seed2023预热权重副本。没有将新增优化种子当作计时轮次。

| dataset | variant | parameter_reduction_percent | prediction_cached_latency_reduction_percent | update_fixed_record_latency_reduction_percent | prediction_fresh_text_latency_reduction_percent |
|---|---|---|---|---|---|
| Energy | single_mlp | 35.67 | 32.97 | 47.27 | 8.57 |
| Energy | no_decisiveness | 0.0 | -4.64 | 4.22 | -2.74 |
| SocialGood | single_mlp | 35.72 | 35.11 | 48.13 | 21.8 |
| SocialGood | no_decisiveness | 0.0 | 3.93 | 15.14 | 4.48 |

无熵正则不减少参数；其微小或混合的时间变化应结合轮间范围解释。single_mlp保留相同文本投影但缩减专家/门控结构，缓存预测与更新的节省不能直接等同于含BERT的新文本端到端节省。完整P95、内存口径和轮间范围见profile_output/review_sprint_cost_20260929/REPORT.md。

## 五种子当前文本内容诊断

差值为控制文本减正确文本；正值才表示正确内容误差更低。每个原点的模型/数值/历史反馈固定，诊断不改变参考轨迹。

| dataset | variant | condition | n | delta_mse_mean | delta_mse_sd | delta_mae_mean | prediction_divergence_mae | eligible_origins | total_origins |
|---|---|---|---|---|---|---|---|---|---|
| Energy | text_add | train_mean | 5 | 1.443e-08 | 2.943e-07 | 1.733e-07 | 4.491e-06 | 143 | 143 |
| Energy | text_add | historical_donor | 5 | 2.286e-08 | 3.114e-07 | 1.052e-07 | 6.478e-06 | 143 | 143 |
| Energy | full | train_mean | 5 | -2.911e-06 | 1.010e-05 | -1.105e-05 | 6.435e-05 | 143 | 143 |
| Energy | full | historical_donor | 5 | -1.247e-06 | 9.473e-06 | -7.210e-06 | 8.694e-05 | 143 | 143 |
| SocialGood | text_add | train_mean | 5 | 6.427e-07 | 4.713e-07 | 1.034e-06 | 1.961e-05 | 85 | 85 |
| SocialGood | text_add | historical_donor | 5 | 3.698e-07 | 1.233e-06 | 2.082e-06 | 2.760e-05 | 85 | 85 |
| SocialGood | full | train_mean | 5 | -3.069e-06 | 1.250e-05 | -7.453e-06 | 6.514e-05 | 85 | 85 |
| SocialGood | full | historical_donor | 5 | -4.378e-06 | 8.603e-06 | -9.166e-06 | 9.210e-05 | 85 | 85 |

这些同状态干预针对当前文本，不是删除全部历史文本或证明原始文本不含预测信息。文本分支是否提供增益与其是否利用具体内容是不同问题；不能用置零造成的变化独立证明语义利用。

## 对review的更新

- ECmL W3/F3、LCo1 W2/P4：候选的组件删减结果现在有5seed，显示是否维持原3seed方向；不宣称所有模块都必要。
- ECmL W4/F4：补充候选的参数、独立预测/更新/新文本成本和内存；正式预测状态未改变。
- LCo1 W4：文本内容敏感性复核扩大到5seed，仍报告控制条件和边界。
- W1/P3：报告全部种子及配对波动，不以窗口数放大统计样本量。
- 原生GPT4MTS/VoT公平比较、真实publication time和独立数据确认仍未由本轮解决。

# 共同OT静态对照：单seed开发试验

Economy，L36/H6，seed2023；训练253行、选择段截至296、开发段截至338。37个预测原点、222个有效OT目标元素（重叠目标不视为独立重复）。三方法均不使用开发标签更新。

| method | seed | origins | valid_elements | mse | mae | epochs | best_epoch | trainable_parameters_before_static_freeze | total_parameters_excluding_BERT |
|---|---|---|---|---|---|---|---|---|---|
| numeric_static | 2023 | 37 | 222 | 0.080086 | 0.220759 | 6 | 1 | 10496 | 37350 |
| iadapter_static | 2023 | 37 | 222 | 0.116353 | 0.284588 | 6 | 1 | 166832 | 193686 |
| gpt4mts_static | 2023 | 37 | 222 | 0.078934 | 0.226644 | 6 | 1 | 1445384 | 82551560 |

内部两个方法共享单变量骨干，GPT4MTS保留作者模型。iAdapter在训练流预热，开发反馈保持零；GPT4MTS为批量监督训练后静态评估。文本可用性截止点一致，实际历史长度和编码方式不同，不能称文本输入完全相同。新表不与原多变量在线结果合并。只有一个seed，不计算SD/显著性，不声称普遍优于任何方法。训练阶段、时间和文本接口完整字段见summary.csv，原始预测可重算。

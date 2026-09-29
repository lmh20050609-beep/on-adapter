# 候选在线成本：独立微基准

硬件：NVIDIA GeForce RTX 4060 Laptop GPU；PyTorch 2.6.0+cu124。两个域、五方法、seed2023预热权重副本；每方法3轮、每轮50次预热/200次测量。统计量为轮内中位数的中位数，括号是各轮中位数范围，单位毫秒。不同轮按固定seed轮换方法顺序，不能将600个调用视为600个独立科研重复。

## 参数与计时

| dataset | variant | trainable_parameters | backbone_parameters | prediction_cached ms (range) | update_fixed_record ms (range) | prediction_fresh_text ms (range) |
|---|---|---|---|---|---|---|
| Energy | full | 167088 | 27238 | 3.212 (3.208–3.253) | 9.282 (8.703–11.024) | 8.918 (8.423–9.681) |
| SocialGood | single_mlp | 107232 | 26854 | 2.050 (1.968–2.175) | 4.856 (4.504–5.556) | 7.755 (7.376–8.114) |
| Energy | single_mlp | 107488 | 27238 | 2.153 (2.134–2.180) | 4.894 (4.679–5.483) | 8.154 (7.673–8.553) |
| Energy | numeric_add | 10752 | 27238 | 1.785 (1.691–1.835) | 3.096 (2.537–3.321) | 1.785 (1.691–1.835) |
| SocialGood | full | 166832 | 26854 | 3.160 (3.075–3.530) | 9.363 (8.614–11.302) | 9.917 (9.085–10.633) |
| Energy | no_decisiveness | 167088 | 27238 | 3.361 (3.226–3.497) | 8.890 (8.640–12.183) | 9.163 (8.995–9.429) |
| SocialGood | text_add | 94880 | 26854 | 1.889 (1.868–1.902) | 3.499 (3.281–3.554) | 7.574 (7.528–8.415) |
| SocialGood | no_decisiveness | 166832 | 26854 | 3.035 (2.940–3.081) | 7.945 (7.814–8.846) | 9.473 (8.945–9.642) |
| SocialGood | numeric_add | 10496 | 26854 | 1.666 (1.665–1.735) | 2.845 (2.799–3.324) | 1.666 (1.665–1.735) |
| Energy | text_add | 95136 | 27238 | 1.912 (1.900–2.034) | 3.509 (3.482–3.536) | 7.697 (7.518–7.978) |

参数与测量使用同一实际模型实现。full和no_decisiveness参数相同，移除正则并不减少模型参数。缓存预测不含BERT；新文本预测包含tokenizer、完整冻结12层BERT、CPU缓存格式转换、投影和数值预测。有限性检查与设备同步纳入计时。

## P95

以下是各轮P95的中位数，不是将所有调用合并后的P95。

| dataset | variant | prediction_cached_p95_ms | update_fixed_record_p95_ms | prediction_fresh_text_p95_ms |
|---|---|---|---|---|
| Energy | full | 6.617 | 12.893 | 17.914 |
| SocialGood | single_mlp | 4.598 | 6.166 | 15.391 |
| Energy | single_mlp | 4.519 | 6.273 | 15.196 |
| Energy | numeric_add | 3.836 | 4.493 | 3.836 |
| SocialGood | full | 6.735 | 12.208 | 18.877 |
| Energy | no_decisiveness | 6.837 | 12.393 | 18.501 |
| SocialGood | text_add | 4.073 | 4.875 | 14.926 |
| SocialGood | no_decisiveness | 6.456 | 10.438 | 17.156 |
| SocialGood | numeric_add | 2.127 | 3.798 | 2.127 |
| Energy | text_add | 4.293 | 4.965 | 14.865 |

## 内存

| dataset | variant | cached_path_peak_MiB | fresh_path_peak_MiB | cpu_embedding_cache_bytes | replay_queue_peak_bytes | gradient_cache_bytes |
|---|---|---|---|---|---|---|
| Energy | full | 19.621 | 437.753 | 3634176 | 19026 | 1408 |
| SocialGood | single_mlp | 18.621 | 437.511 | 2211840 | 4214 | 384 |
| Energy | single_mlp | 18.654 | 437.518 | 3634176 | 19026 | 1408 |
| Energy | numeric_add | 17.521 | 17.427 | 0 | 18662 | 1408 |
| SocialGood | full | 19.608 | 437.747 | 2211840 | 4214 | 384 |
| Energy | no_decisiveness | 19.621 | 437.753 | 3634176 | 19026 | 1408 |
| SocialGood | text_add | 18.473 | 437.463 | 2211840 | 4214 | 384 |
| SocialGood | no_decisiveness | 19.608 | 437.747 | 2211840 | 4214 | 384 |
| SocialGood | numeric_add | 17.499 | 17.411 | 0 | 3878 | 384 |
| Energy | text_add | 18.496 | 437.47 | 3634176 | 19026 | 1408 |

GPU列为本进程PyTorch峰值allocated，非整卡占用或reserved。缓存路径峰值覆盖预测与更新，包括Adam状态；fresh路径在恢复预热权重、移除微基准Adam状态后计量，包含BERT。微基准只放一个活动窗口/文本向量，原在线执行器的全流GPU张量不全部计入这里；原开发运行峰值保存在summary.csv的original_development_peak_allocated_bytes。CPU缓存、队列及梯度字节不代表完整Python进程内存，队列数字不含对象管理开销。

## 相对完整模型

正值表示参数/延迟减少；仅比较同域、同阶段，基准为三轮中位数的中位数。小幅差异若与轮间波动相近，不认定为稳定加速。

| dataset | variant | parameter_reduction_percent | prediction_cached_latency_reduction_percent | update_fixed_record_latency_reduction_percent | prediction_fresh_text_latency_reduction_percent |
|---|---|---|---|---|---|
| Energy | full | 0.0 | 0.0 | 0.0 | 0.0 |
| Energy | single_mlp | 35.67 | 32.97 | 47.27 | 8.57 |
| Energy | numeric_add | 93.57 | 44.41 | 66.65 | 79.98 |
| Energy | no_decisiveness | 0.0 | -4.64 | 4.22 | -2.74 |
| Energy | text_add | 43.06 | 40.48 | 62.2 | 13.69 |
| SocialGood | single_mlp | 35.72 | 35.11 | 48.13 | 21.8 |
| SocialGood | full | 0.0 | 0.0 | 0.0 | 0.0 |
| SocialGood | text_add | 43.13 | 40.22 | 62.63 | 23.63 |
| SocialGood | no_decisiveness | 0.0 | 3.93 | 15.14 | 4.48 |
| SocialGood | numeric_add | 93.71 | 47.26 | 69.61 | 83.2 |

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

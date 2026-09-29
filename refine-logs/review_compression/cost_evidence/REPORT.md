# 候选在线成本：独立微基准

硬件：NVIDIA GeForce RTX 4060 Laptop GPU；PyTorch 2.6.0+cu124。2个域、4种设置、seed2023预热权重副本；每方法3轮、每轮50次预热/200次测量。统计量为轮内中位数的中位数，括号是各轮中位数范围，单位毫秒。不同轮按固定seed轮换方法顺序，不能将重复调用视为独立科研重复。

E0/E1: five experts with routing text feedback and entropy disabled, projection 96/8. E2/E3: single MLP, projection 96/8. Sanity timings are not formal costs.

## 参数与计时

| dataset | variant | trainable_parameters | backbone_parameters | prediction_cached ms (range) | update_fixed_record ms (range) | prediction_fresh_text ms (range) |
|---|---|---|---|---|---|---|
| Energy | E2 | 107488 | 27238 | 2.112 (2.018–2.113) | 4.808 (4.426–5.863) | 8.021 (7.409–8.203) |
| SocialGood | E3 | 33928 | 26854 | 2.004 (1.995–2.108) | 4.275 (4.221–4.728) | 7.806 (7.613–8.378) |
| SocialGood | E2 | 107232 | 26854 | 1.958 (1.951–2.021) | 4.442 (4.229–5.498) | 7.840 (7.590–7.952) |
| SocialGood | E1 | 93528 | 26854 | 2.988 (2.964–3.005) | 10.261 (10.160–10.882) | 9.027 (8.621–9.101) |
| SocialGood | E0 | 166832 | 26854 | 3.093 (2.972–3.208) | 10.228 (8.170–10.405) | 8.734 (8.662–9.432) |
| Energy | E3 | 34184 | 27238 | 2.061 (2.056–2.165) | 5.563 (4.499–5.649) | 7.678 (7.547–7.760) |
| Energy | E0 | 167088 | 27238 | 3.172 (3.129–3.203) | 8.604 (8.552–11.527) | 9.089 (8.721–9.150) |
| Energy | E1 | 93784 | 27238 | 3.081 (3.078–3.201) | 10.561 (9.524–11.006) | 9.046 (8.884–9.442) |

参数与测量使用同一实际模型实现。删除损失正则本身不减少参数；不同结构的实际参数量见表。缓存预测不含BERT；新文本预测包含tokenizer、完整冻结12层BERT、CPU缓存格式转换、投影和数值预测。有限性检查与设备同步纳入计时。

## P95

以下是各轮P95的中位数，不是将所有调用合并后的P95。

| dataset | variant | prediction_cached_p95_ms | update_fixed_record_p95_ms | prediction_fresh_text_p95_ms |
|---|---|---|---|---|
| Energy | E2 | 4.468 | 6.273 | 14.262 |
| SocialGood | E3 | 3.696 | 5.426 | 14.813 |
| SocialGood | E2 | 4.211 | 5.745 | 14.993 |
| SocialGood | E1 | 6.311 | 12.934 | 16.938 |
| SocialGood | E0 | 6.289 | 11.668 | 16.404 |
| Energy | E3 | 4.221 | 6.296 | 14.287 |
| Energy | E0 | 6.652 | 11.555 | 16.402 |
| Energy | E1 | 5.954 | 13.576 | 16.893 |

## 内存

| dataset | variant | cached_path_peak_MiB | fresh_path_peak_MiB | cpu_embedding_cache_bytes | replay_queue_peak_bytes | gradient_cache_bytes |
|---|---|---|---|---|---|---|
| Energy | E2 | 18.654 | 437.518 | 3634176 | 19026 | 1408 |
| SocialGood | E3 | 17.783 | 437.231 | 2211840 | 4214 | 384 |
| SocialGood | E2 | 18.621 | 437.511 | 2211840 | 4214 | 384 |
| SocialGood | E1 | 18.499 | 437.467 | 2211840 | 4214 | 384 |
| SocialGood | E0 | 19.601 | 437.747 | 2211840 | 4214 | 384 |
| Energy | E3 | 17.816 | 437.238 | 3634176 | 19026 | 1408 |
| Energy | E0 | 19.613 | 437.753 | 3634176 | 19026 | 1408 |
| Energy | E1 | 18.572 | 437.474 | 3634176 | 19026 | 1408 |

GPU列为本进程PyTorch峰值allocated，非整卡占用或reserved。缓存路径峰值覆盖预测与更新，包括Adam状态；fresh路径在恢复预热权重、移除微基准Adam状态后计量，包含BERT。微基准只放一个活动窗口/文本向量，原在线执行器的全流GPU张量不全部计入这里；原开发运行峰值保存在summary.csv的original_development_peak_allocated_bytes。CPU缓存、队列及梯度字节不代表完整Python进程内存，队列数字不含对象管理开销。

## 相对参考设置 E0

正值表示参数/延迟减少；仅比较同域、同阶段，基准为本次3轮中位数的中位数。小幅差异若与轮间波动相近，不认定为稳定加速。sanity计时不进入正式成本结论。

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

## 测量边界

固定第一个开发原点，使用真实成熟记录作重复更新微基准。每轮恢复相同预热权重；新文本阶段再次恢复，避免用反复优化后的局部权重测量。更新包含task loss、反馈梯度探针、熵项（若启用）、梯度裁剪、Adam和反馈缓存刷新；不含队列调度、日志和磁盘写入。微基准中的重复更新仅用于计时，不产生新的性能结果。源科学实验文件和queue均未改变。

## 复核入口

config.json记录设置、轮数、计时源码提交与硬件；raw.json保留全部逐次时长。review_cost.py执行隔离测量，review_cost_report.py重算中位数/P95；E-01由review_compression_cost.py提供设置、源目录与模型工厂。计时脚本不修改科学训练状态，本次保留全部原始计时。

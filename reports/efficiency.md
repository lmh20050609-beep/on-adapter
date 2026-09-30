# 实测效率

## 参数

| dataset | method | numeric_backbone_params | trainable_params | adapter_to_backbone_ratio | total_params_including_BERT |
|---|---|---|---|---|---|
| Economy | Numeric adapter | 26854 | 10560 | 0.39 | 37414 |
| Economy | Text addition | 26854 | 94944 | 3.54 | 109604038 |
| Economy | iAdapter | 26854 | 166896 | 6.21 | 109675990 |
| Energy | Numeric adapter | 27238 | 10752 | 0.39 | 37990 |
| Energy | Text addition | 27238 | 95136 | 3.49 | 109604614 |
| Energy | iAdapter | 27238 | 167088 | 6.13 | 109676566 |

完整适配器加文本投影的可训练参数约为这个 minimal 数值骨干的 6 倍。绝对规模较小，不代表参数量相对数值骨干更少；不能以包含冻结 BERT 的总参数作分母掩盖这一点。

## 延迟与显存

| dataset | method | cached_pred_ms | update_ms | fresh_text_e2e_ms | adapter_peak_MiB |
|---|---|---|---|---|---|
| Economy | Numeric adapter | 1.698 | 2.617 | 1.698 | 17.61 |
| Economy | Text addition | 1.88 | 3.21 | 8.626 | 18.58 |
| Economy | iAdapter | 3.239 | 9.163 | 10.81 | 19.72 |
| Energy | Numeric adapter | 1.719 | 2.905 | 1.719 | 17.63 |
| Energy | Text addition | 1.949 | 3.367 | 8.494 | 18.61 |
| Energy | iAdapter | 3.561 | 9.889 | 10.632 | 19.73 |

单位：毫秒，表内为中位数。完整 P95、50 次预热/200 次测量记录见 efficiency.json。固定真实窗口的隔离微基准，不修改正式实验状态；使用同一台共享 RTX 4060 Laptop GPU。Update 包含回放损失、梯度探针和优化器，不含队列管理/日志 I/O。Adapter peak 是本进程 PyTorch 峰值 allocated memory，不是整卡用量，也不等于 CPU 文本缓存。队列数字统计张量与文本 ID 内容，不含 Python 对象管理开销；text_cache_bytes 为 CPU embedding 数组大小，GPU 副本计入峰值显存。新文本端到端预测包括真实 BERT 前向；缺省在线实验使用已缓存的冻结表示。

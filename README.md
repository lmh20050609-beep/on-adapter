# On-Adapter：在统一九域多模态 benchmark 上的诊断、改进与确认

本仓库包含 On-Adapter（在线多模态适配器）当前最新的完整训练流程与结果：

- **统一的九域多模态时间序列 benchmark**：Time-MMD 数据，V1 文本时间错配修复，独立文本编码，冻结评分；
- **7 个方法的对照**：iTransformer、ADAPT-Z、DSOF、Proceed、GPT4MTS、MM-TSFlib、On-Adapter；
- **改进后的方法 On-Adapter-CV**，以及冻结方法后在 3 个训练 seed 上的确认实验。

> 旧版仓库内容（iAdapter 原论文代码与数据补充包）保留在标签 `legacy-iadapter-supplement` 和提交历史中。

完整的实验设计、统计方法、数据来源与结果见 **[docs/EXPERIMENT_DESIGN_AND_RESULTS.md](docs/EXPERIMENT_DESIGN_AND_RESULTS.md)**。

---

## 1. 主要结果（确认实验：seed 2025 / 2026 / 2027，方法事先冻结，对照未调参）

| 方法 | 主统计 S ↓ | CVaR₉₀ 平均名次 ↓ | 平均 MAE | 平均 MSE | 尾部优于骨干的任务 |
|---|---:|---:|---:|---:|---:|
| iTransformer（静态骨干） | **1.000000** | 3.843 | 1.0628 | 11.978 | — |
| **On-Adapter-CV（本文）** | **1.000495** | **3.380** | 1.0716 | 12.100 | **22/36** |
| On-Adapter-TA（消融：不加 CVaR 损失） | 1.000108 | 3.481 | 1.0655 | 12.008 | 21/36 |
| MM-TSFlib | 1.014709 | 4.009 | 1.0690 | 12.089 | 16/36 |
| GPT4MTS | 1.035984 | 4.000 | 1.0879 | 12.205 | 16/36 |
| ADAPT-Z | 1.049515 | 3.731 | 1.1421 | 13.390 | 18/36 |
| DSOF | 1.050956 | 3.972 | 1.2860 | 13.904 | 15/36 |
| On-Adapter（原始） | 1.093644 | — | 1.0974 | 12.210 | 17/36 |
| Proceed | 1.110804 | 5.065 | 1.1752 | 13.486 | 11/36 |

- **主统计** S：先求每格的 CVaR₉₀ 与同格 iTransformer 的比值，3 个 seed 取平均后，再对 36 个域/H 任务等权取几何平均。越低越好，iTransformer 恒为 1。
- **分 seed 的 S**：On-Adapter-CV 为 1.00082 / 1.00118 / 0.99938；On-Adapter-TA 为 1.00192 / 0.99972 / 0.99857。
- On-Adapter-CV 是事先指定的主方法。On-Adapter-TA 是同一配方去掉 CVaR 损失的消融，与主方法在同一冻结计划下运行；它的名次一栏在其自己的 7 方法表中计算。
- 结果文件：[reports/v1conf_20261009/results.json](reports/v1conf_20261009/results.json)；冻结计划：[reports/v1conf_20261009/PLAN.json](reports/v1conf_20261009/PLAN.json)。

**结论：**
- On-Adapter-CV 的 CVaR₉₀ 平均名次在 7 个方法中最好，并全面优于其余 5 个对照（全部在线方法与多模态方法）；
- 主统计与静态骨干基本持平（+0.05%），**未严格优于骨干**；剩余差距集中在 Health_US；
- 相对原始 On-Adapter，主统计从 1.0936 降到 1.0005。
- 消融显示：CVaR 尾部损失改善了平均名次（3.48 → 3.38）和 Environment、Agriculture，但在 Health_US 上更差（1.018 → 1.029），所以主统计没有因此下降（1.0001 → 1.0005）。

---

## 2. Benchmark

| 域 | 频率 | 回看窗口 L | 预测步长 H |
|---|---|---:|---|
| Environment | 日 | 96 | 48 / 96 / 192 / 336 |
| Energy、Health_US | 周 | 24 | 12 / 24 / 36 / 48 |
| Traffic、SocialGood、Security、Economy、Climate、Agriculture | 月 | 8 | 6 / 8 / 10 / 12 |

**数据与预处理：**
- 数据来自 Time-MMD，文本使用 V1 最小修复版：只屏蔽已确认晚于描述时点的条目，共 413 处，占 1.34%；数值列逐字节不变。
- 训练 / 验证 / 测试按时间顺序 60% / 20% / 20%；标准化只用训练段拟合；目标缺失时 mask，不插补。
- 在线反馈只使用满足 `old_origin + H ≤ current_origin` 的成熟标签。
- 位置：[reports/benchmark_text_repair_all9_v1_20261008/](reports/benchmark_text_repair_all9_v1_20261008/)（数据 `data/V1/`、报告）；修复代码与 348 条判定规则在 [benchmark_text_repair_all9_v1/](benchmark_text_repair_all9_v1/)。

**评分：**
- 实现：[benchmark_textrepair_v1/score.py](benchmark_textrepair_v1/score.py) 中的 `cell_metrics`；
- 损失为每个原点在全部已观测目标元素上的 MAE；
- CVaR₉₀ 取最坏 10% 概率质量的均值，边界原点按分数权重计入，并与变分定义交叉核对。

**协议文件：** [reports/benchmark_textrepair_v1_20261008/PROTOCOL.md](reports/benchmark_textrepair_v1_20261008/PROTOCOL.md)；V1 主表：[results.json](reports/benchmark_textrepair_v1_20261008/results.json) 和 [results.csv](reports/benchmark_textrepair_v1_20261008/results.csv)（756 格逐格指标）。

---

## 3. 方法：On-Adapter-CV

在原始 On-Adapter（冻结 iTransformer 编码器 + IG-SMoE 适配器 + 文本 FiLM 专家 + 在线梯度反馈）的基础上，只做了 5 项改动：

1. **关闭 VICReg 文本—数值对齐辅助项**：原训练目标为 4–17，验证 MSE 仅 0.03–3，辅助项主导了训练。
2. **训练预算**：On 的训练轮数上限从 3 改为 15，早停 patience 仍为 3，按验证 MSE 选检查点。
3. **CVaR₉₀ 变分尾部损失**：L = MSE + 0.5·[τ + mean(max(e_i − τ, 0)) / 0.1]。e_i 为逐原点已观测 MAE，所用变量、mask 和尺度与评分一致；τ 为各批 90% 分位数的跨批次指数移动平均。
4. **骨干优先的弃权门控**：默认输出 On 内部的冻结骨干预测；仅当适配器在已成熟原点上的折扣平均 MAE（γ = 0.9）低于骨干时才采用适配器，至少需要 4 个成熟原点，并带滞回。
5. **关闭在线参数更新**：在线反馈和门控决策仍然逐原点进行，只是不再更新参数。原规则（Adam, lr 3e-4）在延迟反馈下会失控。

代码：
- 方法版本包：[method_comparison_v1conf_20261009/](method_comparison_v1conf_20261009/)，与上游的差异记录在 `FORK.json`；
- 门控：[scripts/seed2_round5_ta.py](scripts/seed2_round5_ta.py) 中的 `abstain`；
- 在线复现框架：[scripts/onadapter_harness_20261008.py](scripts/onadapter_harness_20261008.py)；
- 原始 On-Adapter 源码：[original-paper-replay/uniform-l36-source/](original-paper-replay/uniform-l36-source/)。

---

## 4. 复现

**环境：** Python 3.9，PyTorch 2.8（CUDA 12.9），见 [requirements.txt](requirements.txt)。全部命令都在仓库根目录执行。

```bash
pip install -r requirements.txt
bash scripts/fetch_external.sh      # 6 个对照方法的代码，按 SOURCE.json 固定 commit 下载，并用 MANIFEST.sha256 逐文件校验
python scripts/fetch_weights.py     # bert-base-uncased、gpt2 的权重，按固定 revision 下载
```

**路径：** 配置文件中的 `runs_root`、`cache_root`、`control_file`、`report_root`，以及 `scripts/v1conf_dispatch.sh` 里的 `B=` 都是原实验机器上的路径（`E:/FedNoise/...`、`C:/Users/...`），使用前需改成本机路径。

**① V1 benchmark 主表**（训练 3 个读文本方法 × 3 seed，可断点续跑；随后审计与计分）：

```bash
python -m benchmark_textrepair_v1.runner run
python -m benchmark_textrepair_v1.finalize audit
python -m benchmark_textrepair_v1.score
```

不读文本的 4 个对照共 432 格，复用历史同协议格，由 [benchmark_textrepair_v1/legacy_numeric_reuse_manifest.json](benchmark_textrepair_v1/legacy_numeric_reuse_manifest.json) 固定哈希，并通过 [reuse_verification.json](reports/benchmark_textrepair_v1_20261008/reuse_verification.json) 和 [numeric_reuse_equivalence.json](reports/benchmark_textrepair_v1_20261008/numeric_reuse_equivalence.json) 验证兼容；其原始运行目录不在仓库中。

**② On-Adapter-CV 确认实验**（训练 → 测试轨迹 → 计分）：

```bash
python -m benchmark_v1conf.runner run --config configs/method_comparison_v1conf_cv_20261009.json
python scripts/v1conf_20261009.py --list cv | while read t d h s; do python scripts/v1conf_20261009.py --traj $t $d $h $s; done
python scripts/v1conf_20261009.py --score
```

`scripts/v1conf_dispatch.sh` 把上面三步串起来，训练用 GPU 串行，测试轨迹用 CPU 并行。

---

## 5. 仓库结构

```
method_comparison_v1conf_20261009/          最终方法（On-Adapter-CV / TA）
method_comparison_textrepair_v1_20261008/   V1 benchmark 包（7 个方法的实现与桥接）
benchmark_textrepair_v1/                    V1 主实验驱动、审计与冻结评分（score.py）
benchmark_v1conf/                           三 seed 确认实验驱动
benchmark_text_repair_all9_v1/              V1 文本修复的数据构建代码与判定规则
original-paper-replay/uniform-l36-source/   原始 On-Adapter 源码（方法代码依赖）
scripts/                                    在线复现框架、门控、确认实验计分与调度、下载脚本
configs/                                    V1 主实验与确认实验配置
reports/                                    V1 数据、主表与确认结果、冻结计划与哈希、开发记录
docs/                                       完整实验设计与结果文档
external/*/SOURCE.json, external/MANIFEST.sha256, weights/*/SOURCE.json   第三方代码与权重的固定来源
```

---

## 6. 开发过程

方法在单个训练 seed（seed 2）上经过 12 轮迭代：
- 每轮先冻结计划（`PLAN.json` 及其 sha256），只在验证集上选型；
- 测试只计算一次，从第 8 轮起还需满足验证门槛才动用测试。

各轮的假设、选型和失败记录见：
- [reports/seed2_goal_20261009/SUMMARY_rounds1-8.md](reports/seed2_goal_20261009/SUMMARY_rounds1-8.md)
- [reports/seed2_goal_20261009/GOAL_STATUS.md](reports/seed2_goal_20261009/GOAL_STATUS.md)
- `reports/seed2_goal_20261009/round*/`

仓库只保留最终流程的代码，中间版本包不包含在内。

## 7. 限制

- seed 2025–2027 与开发用的 seed 2 处在同一测试时间段，确认实验检验的是对训练随机性的稳健性，不是在全新数据上的泛化。
- On-Adapter 经过 12 轮开发，对照方法用的都是原始设置，调参投入不对称。
- 最终方法关闭了在线参数更新，在线成分只保留反馈和门控。
- 0.05% 的主统计差距在统计上与持平无法区分；不声称显著、普遍最优或 SOTA。

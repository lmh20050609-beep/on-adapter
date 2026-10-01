# N3 fresh-agent code review

**Verdict: PASS — same-family provisional；真实 runtime 尚未检查。**

按 `aris:experiment-bridge` Phase 2.5 审查 `source_patch/iadapter/extension.py`、锁定计划和 `output/iadapter-replan-executed-20261001/iadapter` 依赖快照。没有发现要求在部署前修补的 BLOCKING 实现问题。此结论不是 GPU sanity、完整测试或追加域结果通过的证明。

本次只读取源码与原始 CSV，并用 Windows bundled Python 执行 NumPy/pandas、AST 提取的纯函数及显式轻量替身控制流检查；未导入 Torch/transformers、未调用 WSL、未启动 GPU、未修改研究源码。已阅读现有执行器和候选测试，但没有把既往测试内容当成本轮通过记录。

## 已核实的实现

- **范围与候选固定。** `extension.py:26,65,74` 固定 Agriculture/Health、OT、H6/L36、2023–2027；10 个共享骨干、50 个评价、30 次 warm。原五维候选由 `revision_candidate.py:23` 直接复用，未调用 N1 排序或用扩域标签选候选。`run.py:15` 骨干仅在 train 上反传、用 val 选择 checkpoint；最多 30 epoch、batch16、lr0.001、patience5、d32/2层与锁定计划一致。最终源码 `run.py:2` 有 `Path` 导入，没有未定义 Path 的当前失败路径。
- **数据与真值。** `data.py:60` 按 end_date 稳定排序；数值填补与均值/尺度仅来自训练前缀；`extension.py:28` 只取 OT。检查确认已提取 CSV 与 Downloads 原 ZIP 成员逐字节一致，`load_data` 与独立 `original_reference` 的 x/target/valid/mean/scale/input_missing 全量一致。`extension.py:68,96` 在保存及汇总时重新从原 CSV 验证原点、真值、mask 与 MSE/MAE，评分没有把骨干输出当真值。
- **成熟、warm 与 reset。** `engine.py:40` warm 仅运行 train；`shared_online.py:78` 每次重建 Adam 和成熟队列，`InternalBackend:20` 重建零 hg/ge；先发布预测，再消费 `origin+6` 已成熟的记录，首 7 个预测尚未受在线更新影响。测试末尾 6 个未成熟原点不更新。原点、原始 pooled 输入、hg/ge 快照以及 text IDs 保存在同一 Replay/Record，更新时使用该记录，不用更新时的新文本。
- **缺失目标与 context IDs。** 两域 OT 实际均无缺失。若一个成熟窗口全缺失，已有 Stream/OnlineRunner 仍消费一次并记 `skipped_missing_target`，不反传；`test_shared_online.py` 已覆盖该机制。IDs 在预测时直接从该原点 `data.text_ids[t]` 复制，并检查每个源行不晚于原点，成熟更新沿用同一记录。`save` 未再次独立比对 IDs，但当前调用链没有错配路径，也没有基于不存在的全缺失实际数据制造缺陷。
- **冻结与公平初始化。** 每个 dataset/seed 只取一次骨干，立即冻结并清空梯度；每个评价后检查 named parameters 与起点一致且梯度为空。当前骨干没有训练态滚动统计更新。`review_sprint.py:31` 按模块名固定初始化，`revision_candidate.py:23` 固定 5→96 第一层；raw5/LLM5 用完全相同 topology，运行时逐参数检查初始状态相同。LLM static/online 重载同一 warm state 并重设 RNG；static 的 `external_static.py:48` 用 no_grad/eval，没有优化器或更新调用。随后 online 另建优化器、反馈与队列，源码还检查首 7 个预测相同。
- **文本与 transform。** `prepare.py:29,35` 使用完整冻结 12 层 BERT、masked mean、64 token 截断。`extension.py:32` 用同一 tokenizer 的 offset 截出实际可见字符前缀，传给固定 Qwen；未添加目标数值或标签。`task_text_prior.py:47,55` 保持本机固定 weights、SYSTEM、greedy、max_new_tokens8、singleton，并检查 prompt≤512 token 和生成 logits 有限。只有目标说明改为新域。`text_signal_probe.py:27,37` 与 `extension.py:59` 的 PCA/标准化均只 fit 训练前缀；未来输入扰动检查证实不改变 fit 参数。缺文表示置零，`models.py:100` 又在投影后乘 available，避免投影 bias 产生缺文信号。
- **缓存、出处与失败。** BERT cache 检查当前文本序列及形状；prior 输入保留可见文本、dataset/origin/split/selected IDs/key，candidate lock 保存模型来源、prompt/context、解码设置和代码版本；新域独立 prior cache 避免复用原域结果。生成只接受≤1885 unique 输入；失败写 failed 与 traceback、累计已用时间，预算停止写 stopped_budget；campaign 只接受 completed 生成状态，遇失败停止并保留 failed 队列，不把部分结果汇总为通过。失败/完成/预算停止三个新控制流以明确的无模型替身运行通过。

## 原 ZIP 检查的实际结果

| 数据域 | 总行数 | train_end / val_end | 测试原点 | 测试预测数 | 成熟消费数 | 非空文本 / 训练非空文本 | OT缺失 |
|---|---:|---:|---|---:|---:|---:|---:|
| Agriculture | 496 | 297 / 396 | 395–489 | 95 | 89 | 432 / 233 | 0 |
| Health | 1389 | 833 / 1111 | 1110–1382 | 273 | 267 | 1389 / 833 | 0 |

两域各自 train/val/test 目标均落在对应区间内。Health 的 10 行形成 5 组重复 start/end interval，但 YEAR/WEEK/REGION TYPE/REGION 联合键唯一，稳定排序保留全部记录；H6 是后续 6 条报告。end_date 仍是可用时间代理，本次未验证真实发布时间；没有将其改称真实发布时刻。

## 非阻塞澄清与当前失败路径

1. 锁定文档的“600秒 preprocessing budget”应明确为“新增 Qwen 生成累计 wall time，包含 load/tokenization/失败；BERT/PCA 成本另记”。协调者已明确该含义，与 `extension.py:45` 计时范围一致。预算在每个 singleton 完成后检查，因此它是单条调用边界的停止阈值，不是硬中断计时器；替身案例准确记录了超过阈值后停止，不能声称真实运行已经超过预算。
2. **当前已观测执行阻碍来自环境：** `RUNTIME_STATUS.json` 记录 `WSL/Service/CreateInstance/0x80072747` 和 `HCS_E_CONNECTION_TIMEOUT`，production patch 尚未应用。此审查没有再次调用 WSL，也没有发现 N3 已运行后的 traceback；不应把模拟的 `review load failure` 当作真实模型失败。
3. N3 的真实 tokenizer offset/prompt token 数、模型加载与 forward、同初始权重断言、static/online 首 7 预测、冻结断言、耗时以及完整现有测试仍需在既定环境恢复后完成。按锁定计划先做 Agriculture/2023 sanity，再继续已授权矩阵；本次不开展这些执行。

## Primary trace

- `review_trace_n3/reviewer_checks.py`：审查脚本；仅真实 CSV/纯函数/明确替身，非模型实验。
- `review_trace_n3/reviewer_checks.json`：逐域原数据、split、成熟队列、训练前缀 transform 与生成控制流的本轮输出；status=passed。
- `review_trace_n3/control_flow/*/generation_status.json`、`generation_budget.json`：三个替身用例的原始状态/预算证据。`failed_load` 的 traceback 由审查替身故意产生。
- 原数据：`C:/Users/18452/Downloads/31817_iAdapter_Adaptive_lightw_Code And Data Supplement.zip` 中 `iAdapter/data/Agriculture.csv`、`iAdapter/data/Health.csv`。

最终结论：**PASS，无 BLOCKING；same-family provisional；runtime unverified。**


## 最终补丁的最小范围复核

**补充 verdict：PASS，无新增 BLOCKING；same-family provisional，runtime 仍未检查。**

本轮仅检查新增 `paired_report()`、`analyze()` 配对输出连接及 `generate()` 成本字段，不重复数据/执行器审查。600 秒预算文档现已明确为新增 Qwen 累计生成 wall time（包含加载、tokenization 与失败），先前措辞问题已解决。

`paired_report()` 在 dataset 内按 seed 建索引，明确要求候选与每个对照均覆盖 2023–2027，拒绝重复 dataset/seed/method。它也要求候选与对照、各 seed 的 valid_elements 相同。Windows AST 提取实测用乱序输入构成 2 域×5 seed×5 方法，得到 40 条配对记录与 8 条汇总；所有计算与独立 NumPy 结果一致：candidate−control 有符号差值、五个差值的样本 SD（ddof=1）、严格 delta<0 的改善 seed 数、100×(候选均值/对照均值−1)。正负结果全部保留，SD 没有写成置信区间或以窗口充当随机重复。

7 个错误输入均按预期失败：重复 cell、缺失 seed、缺失对照、错误 seed 集合、单 cell 目标计数不同、整个对照目标计数不同、各 seed 目标计数变化。`analyze()` 从固定 10 job×5 方法逐项读取 completed 状态，重新依原 CSV 计算得分，再传给配对函数并写 `paired_runs.csv` / `paired_summary.csv`；因此配对函数自身不枚举缺失整个 dataset 的情况，不会使当前分析入口漏掉计划域。

新增 `total_input_tokens` / `total_output_tokens` 精确引用 `task_text_prior.classify_batch()` 已返回的 `input_tokens` / `output_tokens` 字段。明确的无模型替身检查了“一个已缓存输入＋一个新 singleton”路径：token 总计正确包含两条缓存记录，生成 batch 仍为 1，状态/预算写入成功，候选及执行器没有改动。此替身中的 token 数、显存值均为测试数据，不是实测模型成本。

成本字段口径：token 总数是整个 cache 的累计计数；`seconds` 与 `peak_allocated_bytes` 对应本次调用，累计时间以 `generation_budget.json` 为准。`reset_peak_memory_stats()` 在 `load_model()` 后，故峰值包含常驻权重和生成阶段分配，不包含模型加载瞬态峰值。这是非阻塞的报告口径说明；真实 CUDA 峰值与耗时尚未测量。

新增 primary trace：`review_trace_n3/addendum_checks.py` 与 `addendum_checks.json`，status=passed，未导入 Torch/transformers，未调用 WSL/GPU。原 `reviewer_checks.json` 保留为前次补丁的检查记录；本轮 trace 覆盖最后新增字段。

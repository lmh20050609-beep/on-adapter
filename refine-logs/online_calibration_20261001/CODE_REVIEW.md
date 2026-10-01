# 新实现独立代码审查

日期：2026-10-01。审查类型：experiment-bridge 要求的新鲜二级代理审查，same-family provisional。请求的审查配置为 gpt-6-astra / xhigh。本结论是实现正确性审查，不是性能证明或独立确认性研究结论。

## 结论

**BLOCKING：未发现阻止最小 sanity 运行的实际计算错误。** 当前组合器、数据切分、真实目标评分、旧预热复用和分支状态重置逻辑正确。可以在同一已提交源码版本下先运行 Energy/2023 sanity；实际 GPU 运行、耗时与收益仍须由执行方取得证据。

有三项非阻断意见。第一项是计划文字与实现执行顺序的实际差异；第二项是已有来源锁在后处理入口中的遗漏；第三项是成本报告必须区分的计时范围。它们不构成“已经使用测试集选参”或“当前数值结果错误”的证据。

本审查没有修改任何实验源码，也没有运行神经网络推理、训练或 GPU 实验。阅读了全部计划、决策账本、三个提交审查的实现/测试文件，以及相关数据、骨干、预热、文本变换、在线驱动、指标与审计接口。额外只做了 CPU 上的只读数据/JSON/NPZ 来源核验。执行方报告的八项纯组合器测试通过；本审查阅读了这些测试，没有把它们伪称为自己重新运行的结果。

## 已核实的正确实现

1. **真实目标与掩码。** `frozen_run` 和 `OnlineRunner` 保存的都是原数据的未来窗口目标。每个分支经 `audit_arrays` 独立读取原 CSV、重建 OT 标准化目标与缺失掩码，并重新评分。混合器对两条账本的原点、目标、掩码要求完全相同；未把模型输出当作标签。原 CSV 标准化统计仅来自前 60%。

2. **60/10/5/5/20 边界。** `region_origins` 使用 `[lo-1, hi-H)`，因此每个窗口的第一个目标至少为 lo、最后一个目标至多为 hi−1。校准使用截止 `calibration_end−1` 已完整成熟的固定分支窗口，这与 ASSUMPTIONS 中显式记录的阶段初始化约定一致；不是把逐 h 可用性偷偷引入实现。目标边界实测如下：

   | 数据 | 训练截止（不含） | 校准原点 | 选择原点 | 评估原点 |
   |---|---:|---:|---:|---:|
   | Energy，1479 行 | 887 | 1034–1102，69 个 | 1108–1176，69 个 | 1182–1472，291 个 |
   | SocialGood，900 行 | 540 | 629–668，40 个 | 674–713，40 个 | 719–893，175 个 |

   旧骨干通过前 60% 的训练窗口拟合、60–70% 的窗口选择。读取的源实现保证每个训练/选择窗口的目标终点早于对应右边界。

3. **六组旧来源确实一致。** 对 Energy/SocialGood × 2023/2024/2025，读取源文件确认 numeric、LLM5、raw-BERT-PCA5 和 full 的旧运行均完成，预热文件存在，四个分支各自的源记录指向该组相同的 OT 骨干 checkpoint。每组的预热更新最后成熟目标索引为 Energy 880、SocialGood 533，均严格早于训练截止 887/540。这里没有误用旧在线结束权重，也没有把 full 换成五维简单融合权重。full 在新评估片段以自己的旧 full 预热重新运行，满足独立强参考的要求。

4. **后 20% 固定输入和训练前缀变换。** 现有完整 BERT 缓存已经分别覆盖 (1479,768) 和 (900,768)，并由 `load_data` 核对全长文本；本轮扩展的是固定 Qwen 标签缓存。旧 Qwen 缓存实际有 1538 个键，全部 valid。prepare 复用旧缓存键，仅为缺失的固定可见文本生成新条目，原缓存文件不被覆盖；生成函数不接收数值目标。可见规则是原 tokenizer 的 64 token 截断。六个 transforms 的旧配置均指向对应同一骨干，拟合原点满足 `max(origin)+6<train_end`；PCA center 也与真实训练前缀可用文本的均值相符。原生成 transforms 的源实现仅用训练前缀拟合 PCA 和 LLM 标签标准化；新 `features` 只应用旧变换，不重新拟合。原 cache_key 中的 domain、固定目标描述、system prompt 和可见文本参与键构造，未发现该实现的实际缓存身份错误。

5. **校准分支冻结且阶段重置。** 校准只运行 static。每个 phase/mode 都恢复各自的 warm state，重新创建 backend，因此反馈缓存、优化器和成熟队列从空状态开始；静态分支不消费目标更新。骨干权重相等断言及静态 displacement=0 检查有效。旧 static 校准前缀通过 `np.testing.assert_allclose(atol=1e-6, rtol=1e-6)` 核对；已修复的 `audit.same` 参数问题不存在。

6. **LS 与 Hedge 的初始化信息范围一致。** 每个表示/seed/域的 LS 和所有 eta 的 Hedge 都使用相同 static 校准账本、相同完整成熟 cutoff 和相同可用文本/有效标签掩码。LS 的先验为裁剪的 Acal/Bcal，lambda 为 kappa·mean(d²)+epsilon；新片段的 A/B 为零。per-horizon 沿 channel 求和，shared 进一步沿 horizon 求和并广播共享 alpha，单位匹配。Hedge 的两个端点与五个固定混合候选分别使用真实候选平方误差，从同一校准集合按成熟顺序 warm start；初始化/更新后按账本约定先 floor 再归一化。没有把 LS 的全部校准 A/B 又额外带入评估。

7. **先预测后更新、原发预测和 once-only 消费。** `replay` 先读取 alpha 并发出当前混合预测，再执行当前时钟衰减及 `origin+H<=t` 的成熟更新。更新使用账本内原发出的 numeric/text 预测，不用当前参数回算过去预测。缺失目标先被掩码隔离；无文本时输出 numeric 并且不添加价值统计。LS 时钟不会因无文本而停止。每个成熟 origin 只消费一次，末尾不会 flush 未成熟记录，初始 gate 被深拷贝，因此验证运行状态不会带入评估。

8. **全局选参与三个预定场景。** `select` 明确遍历两个域、三个 seed、两个表示和全部三个场景；每个算法族/setting 均有 36 个条件，按同条件 numeric MSE 比值平均选出一个全局设置，平局用 setting index 决定。该函数只读取 calibration/selection 账本，没有读取 assessment 指标；`evaluate_mix` 使用锁定设置、不找测试最优路径。没有引入 >2% 门槛，旧其他结果也没有混入新评估。

9. **内容探针范围准确。** donor 来自训练前缀，并保留当前预测时的 feature、反馈状态、数值窗口和适配器参数；zero-current-text 同样只改变当前文本。探针在 eval/no_grad 下运行，训练更新仍使用原始 context 的原文本。这里比较的是同一既定轨迹上的当前内容敏感性，不能当成替代训练/在线政策或文本因果效应；计划已经明确这一点。

10. **队列和失败处理适合本地串行任务。** ledgers 的 queue 记录计划和源码版本，拒绝继续未处理的 failed/running 组，completed 组不重跑；组目录与 branch 目录使用 `exist_ok=False`，不会悄悄覆盖旧失败结果。JSON 使用临时文件替换写入。实际本地合作操作者场景不需要再增加锁服务、哈希方案或恢复框架。

## NON-BLOCKING 意见与窄修正

### N1：评估分支账本和分数在规则锁定之前生成

位置：`source_patch/calibration_pilot.py:176` 的 phase 循环包含 assessment；`select` 在整组 queue 全部 completed 之后才启动。

影响：这与 EXPERIMENT_PLAN 第 5 步“全部验证账本齐全后选参数并锁定，然后全部评估”的字面顺序不同。当前代码中，分支预测独立于组合规则且 `select` 只读 selection，**因此没有发现算法使用 assessment 做选择的路径**。不过现在不能声称“首次生成或查看评估分数发生在参数锁定后”。

具体修正：优先把执行顺序改成先完成六组 calibration/selection、运行 select 锁定设置，再生成 assessment 分支与混合结果。仅需分开这两个阶段及其完成判据，不需要调度框架。若明确保留预先计算固定分支账本，执行前应准确记录这一偏离，并在锁定之前不读取或据此修改任何 assessment 结果；报告应说明实际顺序。

### N2：select / evaluate_mix 没有延续已有源码锁检查

位置：`source_patch/calibration_pilot.py` 的 `select`、`evaluate_mix` 入口。ledgers 会核对 `queue_state.code == revision()`；select 仅核对 jobs completed，evaluate_mix 仅核对 locked.status。

影响：同一已提交源码版本连续执行时结果正确，当前没有观察到版本混用。但是修改源码后直接运行这两个入口，不会触发已有的 queue/source 不变约束；locked.json 会记录新版本，即使输入账本来自旧版本。

具体修正：在创建输出目录前，以现有 revision/queue/locked 字段做简单相等检查：当前源码必须 clean；select 要求当前版本等于 queue 版本；evaluate_mix 要求当前版本等于 locked 和 queue 版本。沿用现有来源字段即可，不要增加新的指纹或迁移机制。已有输入生成来源也应按一次完整提交版本顺序执行；本审查没有发现其当前实际不一致。

### N3：报告分支成本时要剔除内容探针和审计/保存开销

位置：`ContentBackend.predict`、分支 `prediction_seconds`、`probe_seconds` 和 `metrics.seconds`。

影响：ContentBackend 在一次 predict 调用内额外运行两个探针，因此 LLM5/raw5 的 prediction_seconds 包含探针，numeric/full 则没有。metrics.seconds 还包括审计、压缩和保存等工作。直接横向比较这些原始字段会夸大简单文本分支的预测成本。

具体处理：不影响本轮预测或上线 sanity。后处理使用已经保存的 probe_seconds 分离探针开销，分别报告原发预测、真实在线更新、探针诊断和整体实验耗时；双分支成本相加，参数规模也按实际两条路径相加。不要因为 alpha=0 声称节省了已经执行的文本路径。现有输出足够开展这项确定性整理，无需重跑模型。

## 验证边界

本轮不替执行方证明实际 GPU sanity 成功、600 秒生成预算足够、最终收益或统计显著性。SocialGood 评估段实际有 33 个缺失目标 cell（跨重叠预测窗口计数），因此报告必须保留当前掩码，而不是将这些窗口删除后与另一分支改用不同原点。已见数据上的规则选择仍为探索性；小于 1% 或随 seed 波动的真实改善应原样报告，不应套用旧 >2% 淘汰条件。

**审查建议：可以继续最小 sanity；对 N1/N2 做小范围执行顺序/来源一致性修正，对 N3 在确定性汇总中处理。不需要扩大模型、搜索范围或增加安全基础设施。**


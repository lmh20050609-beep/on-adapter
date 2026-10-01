# VoT 接入核验与剩余条件

本文件核验 [作者仓库](https://github.com/decisionintelligence/VoT) 的 `f769621da9efde3e82a1975f478efdec83dcbd10`。下载的源文件是只读材料，尚未运行 native VoT，尚未产生新的基线分数。

## 已有代码能做什么

`models/PatchTST_clip.py` 和 `exp/exp_long_term_forecasting_clip.py` 提供内生文本对齐与频率融合的实现；发布脚本实际调用 `run_clip.py + PatchTST_clip`。脚本当前激活的是 SocialGood、seed=2025，其余域列在注释中；不能直接运行整份多并发/多超参数脚本作为本机公平比较。仓库也提供预生成的 event-driven 数值预测，但它们不是原点级生成日志。完整方法还包含历史示例检索和事件推理；这一结构在 [VoT 原论文](https://arxiv.org/html/2603.15452v1) 中说明。因此，只有模型代码和 benchmark 名称仍不能验证完整复现。

## 当前源文件中的具体差异

| 检查项 | 源文件证据 | 对接入的影响 |
|---|---|---|
| 模型入口 | scripts/vot.sh:70/95 为 PatchTST_clip/run_clip.py；iTransformer_clip.py:96/98 调用未定义的 forecast_clip/forecast_2 | 使用作者实际PatchTST入口；不补造iTransformer多模态路径后称原生实现 |
| 先验路径 | data_loader.py:250 读取 stat_predictions_ecnu_rag；发布目录是 stat_predictions_rag | 原路径不可直接运行；应在独立副本显式修正并保留差异 |
| SocialGood数据范围 | 作者CSV892行是原投稿900行的前缀；原尾8行OT缺失。先验893条与作者L8窗口885条不一致 | 保存行号/日期映射及缺失政策；不能假定跨文件同ID天然对应 |
| 先验元数据 | JSON 每项只有 Prediction，没有原点、选中文本、检索例子或生成时刻 | 可以检查长度、有限值和目标对齐；无法据此证明按真实可用时间生成 |
| 先验长度 | Agriculture 为 12；Energy 为 48；Health 大多数 48，另有少量更长 | 截取前 6 值不等于原生 H=6 提示生成；若使用必须标注截断 |
| 编码器版本 | language_model/download.py 下载 Qwen3-Embedding-0.6B；论文实现说明提到 Qwen2-Embedding 0.6B | 不能自动下载后把结果称为同版本复现；先确定作者使用的版本 |
| 数值文本编码器 | 官方脚本为 GPT2，exp 文件查找 language_model/openai-community/gpt2 | 可复用本地 GPT2 权重，需明确路径映射并禁止隐式联网回退 |
| 数据划分 | data_loader.py:72 等使用 70/10/20 | 与当前修订主实验 60/20/20 不同，表格不能直接混合 |
| 流程 | 官方脚本提供三个训练阶段 | 只跑数值或融合末阶段不等于完整 VoT |

作者发布的 Agriculture 先验有 489 条，ID 连续 0–488，数量与 496 行/L=8 的滑窗总数相符。这是对齐线索，不是已验证的原点映射。Health 有 1354 条，Energy 有 1444 条。此前完整记录见上一交付包 `refine-logs/external_baselines/vot_prior_audit.json`，新源文件见 `tmp/vot_source_audit_20261001/`。

## 可执行的比较顺序

1. WSL 恢复后，先建立独立的 VoT 运行副本，保留作者原文件。使用脚本实际入口 PatchTST_clip，复用已有 GPT2，仅做依赖/形状/一次训练更新的 smoke；smoke 不计入性能主表。
2. 用原 CSV 校验 ID→历史窗口→未来真实 OT 目标的映射，尤其是排序和边界；先验只能作为输入，绝不能充当 ground truth。任何长度变换单独记入配置。
3. 若使用作者先验，结果命名为“VoT native model with released priors (availability provenance unverified)”，单独作为来源受限的补充表。保留强静态结果，不用一个较差的在线扩展替代它。
4. 可信主表需要原点级先验来源：当时可用文本、检索样本只来自训练段、生成提示/模型、输出与解析结果。若作者记录不可得，本机重新生成时保存这些字段，任何替换推理器都明确命名为本地 VoT 变体。
5. 先固定一个域/seed 完成端到端检查，再扩展到事先锁定的矩阵。使用与 iAdapter 相同的目标、评分原点、原始掩码；完整系统比较允许不同骨干，但不能称为同插入位置的适配器消融。延迟在线扩展另设行，与作者静态 VoT 同时报告。

本地 0.5B 枚举分类器输出的五类先验不能直接替代 VoT 的多步数值预测。新生成路线需要先写明推理器和生成提示，再进行少量质量/内存/耗时检查；不把“有代码”当成所有条件已满足。当前未调用 DeepSeek 或其他付费 API，未安装额外环境。

补充可执行来源检查见 `VOT_DISPATCH_AUDIT.json`：只抽取原始 forward，iTransformer 的 forecast=0/2 均触发相应缺失方法；PatchTST 则定义了三个被调用的方法。该检查没有构造模型、没有Torch/GPU，不是端到端smoke。GitHub API 后续下载触发限流，已使用同一固定提交的 raw 源文件成功取回；未改动作者代码。

## 本轮仍缺的材料

- 原 Table 2 的运行日志、位置级统计和配置来源，仍不能由新 benchmark 结果恢复。
- 可信 VoT 原点级推理/检索生成记录，或重新生成且保留差异的本地记录。
- 真实文本 publication-time 元数据；当前只有 end_date 代理。
- 可访问的 WSL 实验环境。当前失败发生在发行版命令/文件共享层，尚未到模型执行。

这些缺口限制“原始完整复现”的称谓，不抹除已完成 GPT4MTS 静态/在线比较、五域修订矩阵和局部文本收益。

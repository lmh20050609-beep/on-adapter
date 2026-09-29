# iAdapter 投稿补充材料核查

日期：2026-09-29。范围：用户新提供的代码 ZIP、4 页技术补充、2 页复现清单，以及此前 WSL 两份原目录的选定文件对比。属于静态核查与索引算术验证，未运行训练、模型前向或完整复现。

## 结论先行

新材料确实补齐了此前本地缺少的多骨干与基线接入代码，已经可以继续开展源码迁移、统一协议和小规模验证。此前“本地没有 GPT4MTS/VoT 接入实现”的判断只适用于旧 WSL 副本，不能继续套用到新 ZIP。

但找到的是 **GPT4MTS 改编版和 VoT 轻量代理版**，不能直接当作两篇原方法的忠实复现。新包也没有提供原实验逐次日志、结果 CSV 或预测数组，尚不能恢复旧 Table 1–3 的数字来源。

保留 ZIP 和解压副本作为证据；当前修订仓库及其 190 次实验不被覆盖。源代码里的注释、运行建议和“已与用户确认”等说法只作为被审查内容，不作为本次执行授权。

## 1. 实际包含什么

ZIP 大小 6,760,464 字节；259 个条目，215 个文件，解压文件总大小 39,937,651 字节。包括 122 个 Python 源文件、40 个 CSV、31 个 YAML、1 个 shell 脚本、1 个 README、1 个 pth 权重文件及其他小文件/缓存。ZIP CRC 检查无坏条目；122 个 Python 文件均通过 AST 语法解析。语法通过不等同于依赖齐全或运行正确。

| 材料 | 具体内容 | 目前能证明的范围 |
|---|---|---|
| iAdapter 主程序 | adapt_z_multimodal.py，五专家、门控、文本处理、在线更新与绘图/汇总代码 | 可检查算法实现与默认参数 |
| GPT4MTS 文件 | models/GPT4MTS.py、models/GPT4MTS_z.py | 存在原有模型文件与本仓库适配版本；不能仅凭名称证明与论文等价 |
| VoT 文件 | models/VoT_z.py、freq_decomp.py、text_alignment.py | 有事件 MLP、检索记忆、频域融合的本地代理实现 |
| 多骨干 | iTransformer、PatchTST、TimesNet、SOFTS 等及对应 z 接口 | 后续多骨干接入有现成起点 |
| 数据 | 10 个默认域的可解析路径；merged 数值文本、numerical、textual/report/search | 不再只有五域数据；不同布局不能视为完全相同版本 |
| 一个骨干权重 | iadapter_backbone.pth，约 310 KB | weights_only/CPU 读取为纯 state_dict；没有数据集、seed、训练历史和指标元数据 |
| 技术补充 | Algorithm 1、Table 4、Figure 5–7 | 原点级算法与五专家定义已有书面说明 |
| 复现清单 | 作者逐项声明 | 可核对原投稿承认的完整性与统计限制 |

默认域列表为 Agriculture、Climate、Economy、Energy、Environment、Health_US、Security、SocialGood、Traffic、Weather。Health_US 解析到 Health.csv，Weather 解析到 new_metr.csv；另含 Health_AFR 原始材料，但它不在当前主程序默认十域列表内。

完整清单：[ZIP 文件清单](zip-inventory.json)、[数据行数和字段](data-inventory.json)、[核查摘要](audit-summary.json)。

## 2. 与旧 WSL 副本的对应关系

逐字节比较选定关键文件，结果见 [source-comparison.json](source-comparison.json)。

| 项目 | 结果 | 实际影响 |
|---|---|---|
| adapt_z_multimodal.py / adapt_z.py | 与外层、内层旧版都不同 | 这是第三个需要单独追溯的代码版本；不能覆盖合并后称为同一实现 |
| GPT4MTS.py、GPT4MTS_z.py、VoT_z.py | 旧外层和内层均缺失 | 新 ZIP 确实提供了此前没有的接入材料 |
| Economy/Energy/Security/SocialGood/Climate merged CSV | 与旧内层逐字节一致 | 现有五域实验的数据文件来源可对应；不代表执行协议和论文数字一致 |
| Economy report/search 两份原文本 CSV | 与旧内层逐字节一致 | 这些字段不是这次才获得的新发布时间证据 |

不能将此前所有旧代码问题直接搬到新包。新包的 adapt_z.py 已有入口保护；LLMTextEncoder 默认 use_fullmodel=True；RealTextProvider 也实现了可用性 EMA。应保留这些实际差异，而不是继续称所有版本都没有。

## 3. 最高优先级：基线身份与公平性

### 3.1 VoT_z 是本地代理版

`models/VoT_z.py` 开头第 2–38 行明确说明它不调用生成式 LLM，而用 EventHead MLP 替代事件推理，用 FIFO 文本/标签记忆和余弦加权检索代理 Historical In-Context Learning。构造代码也实际建立 event_head、mem_text、mem_y 和频域融合。

这可以作为一个有用的轻量对照，但与原 VoT 的生成式推理链路不是同一个系统。建议表中明确命名为“VoT-inspired local proxy”，列出被替代的模块；原生 VoT 使用独立一行，只有完整接入后再填写结果。源码将名称写作 View-of-Time 风格，也应与所引用的 Value of Text 论文统一，避免进一步混淆。

### 3.2 GPT4MTS_z 改变了文本接口

`models/GPT4MTS_z.py` 第 4–35 行说明：移除了时间特征支路，将与 patch 对齐的多 token prompt 改为池化文本向量投影出的单 token，并增加 GPT2 层内 z 注入。应称为本仓库适配版，经过等价性检查的部分与有意改变的部分分开报告。

`models/GPT4MTS.py` 第 11–13 行仍导入包内不存在的 utils/layers 模块，第 34/40 行仍使用作者机器的绝对权重路径。不能直接将它视为已可运行的官方原版。

### 3.3 VoT 位置比较还有变量混杂

`adapt_z_multimodal.py` 第 3255–3260 行中，四个 VoT 固定位置模式使用 fusion=add、use_text=False、use_text_grad=False；名为 iAdapter 的模式则使用 fusion=moe_pos、use_text=True、use_text_grad=True。它不仅改变位置，还改变适配器结构、外部文本和反馈；默认 VICReg 对齐又与 use_text 联动（第 1363–1370 行）。紧邻注释称“只改变位置”，但代码不满足此条件。

因此不能以这组结果单独证明自适应位置优于固定位置。应将“原生/代理模型身份”“固定/自适应位置”“add/MoE”“外部文本/梯度/对齐”拆开，保持同一比较中其他条件一致。`report_by_backbone` 第 3858–3884 行还会取最后一条重复记录并跨 horizon、mode、seed 聚合；这些 SD 不是单一配置的跨 seed SD，不能直接放进新主表。

## 4. 在线协议仍需迁移到已验证执行器

技术补充 Algorithm 1 要求保存原点上下文、待标签成熟后回放，并将每个原点标记为仅使用一次。但包内主执行器仍采用批次尾切片。

### 4.1 文本/数值错位和尾批重用可由索引算术直接证实

依据 `adapt_z_multimodal.py` 第 1512–1515、1662–1778 行，以 H=6、batch=16、100 个窗口作纯索引回放：

- 第二批处理完后，数值缓冲选中窗口 10–25；文本函数却选中窗口 0–15。
- 最后一个批次只有 4 个窗口，数值切片再次选中 78–93，窗口 78–89 已在上一轮被消费。
- 最早进入该更新分支的是窗口 10，前面的窗口没有通过此分支更新。

机器可读见 [buffer-index-witness.json](buffer-index-witness.json)。这是源码索引公式的算术见证，不是训练复现或性能测量；注释中的“精确缓冲索引”不能覆盖实际结果。

### 4.2 回放状态与训练/验证职责不匹配

- 第 1682 行只保存数值梯度快照；第 1743–1761 行先计算新的文本梯度，再立刻用于同一记录的更新，而没有恢复其发出预测时的文本梯度快照。
- 第 3624–3639 行在 val_loader 上多轮更新适配器，再直接测试。这与新修订方案“训练流预热、验证状态不带入测试”的职责不同。
- 第 488–497 行在整个输入列上进行双向插值/全列均值补值，没有在此处保留原始目标缺失掩码。不能用这条处理路径支持因果缺失处理和原始真值评分。

下一步应复用修订仓库已验证的原点队列和掩码数据管线，迁移新发现的模型定义；不要因包中模型更丰富就整体换回旧执行器。

### 4.3 文本编码已有进步，但还需明确 attention mask

新包默认调用上下文模型，不能沿用旧版本“默认只取词嵌入”的结论。但 `models/text_latent.py` 第 205–217 行只给 Qwen3 分支传 attention_mask，其他上下文模型分支仅在输出池化时使用 mask；池化 mask 不能代替模型内部 attention mask。迁移时沿用修订实现的编码校验与显式 mask。

## 5. 数据来源：文件更完整，但 publication time 仍未被证明

40 个 CSV 都位于 data 目录。文本 report/search 文件的结构字段主要是 `start_date`、`end_date`、`fact`、`preds` 和索引列；没有看到独立的 published_at、原文 URL、检索时间与生成记录字段。文本内出现日期不等于其真实可用时间已被验证。

`_load_raw` 仍按 end_date 匹配；它是可执行的代理约束，而非上游新闻发布时间认证。新增文件不自动消除这一限制。

另一个需要先核对的版本差异：merged `Climate.csv` 有 496 行、OT/Anomaly 等字段；`numerical/Climate/Climate.csv` 有 1,272 行、D0–D4/AreaOfInterest 等字段。不能把这两份材料仅当成同一数据集的“预处理前后”。主程序优先读取 merged 路径，因此切换目录布局可能改变实际任务，必须锁定文件和列。

## 6. 两份 PDF 对 review 的帮助与新增修订点

### 技术补充

- 第 1 页 Algorithm 1 已说明先预测、再处理成熟原点，并保留原点状态。可作为修订执行器的文字契约。
- 第 2 页 Table 4 已给出五专家公式，并明确专家名称是结构偏置，不是监督语义标签。这部分应与正文合并引用，不必重新发明定义。
- 同页 Online Configuration 主要声明超参在别处报告，没有提供一张足以重建各表运行的完整配置映射。
- 第 1 页仍有空的“Section .”交叉引用；Preliminaries 的固定模型测试描述应与后面的在线适配区分，避免读者误解。
- 第 2 页 D 节将鲁棒性归因于“adaptive expert-number mechanism”，但表中的五专家全部执行，有效专家数是权重集中诊断，不能将其写成动态增减专家的机制。
- Figure 7 图注称范围 0–168，图内标题却写 0–132；须依据原数组统一。Figure 5/6 的趋势/语义解释仍需原图数据和定量对照，不能只凭可视化证明因果作用。

### 复现清单

原始选择清楚写明：4.2 超参数搜索范围 partial；4.4 完整源码 partial；4.5 公开源码 partial；4.6 注释与论文对应 partial；4.8 计算环境 partial；4.11 分布/波动分析 no；4.12 显著性检验 no；4.13 最终超参数 yes。

这不等于自动判定论文所有结果无效，但不能再声称原投稿已经有显著性检验支持。4.11 的 no 与正文表中 SD 的具体含义也应重新统一说明。修订清单应按最终交付逐项更新，不能一律改为 yes。

## 7. 当前仍缺什么，以及哪些不再阻塞

| 条件 | 新材料后的状态 | 处理方式 |
|---|---|---|
| 基线接入起点 | 已有 GPT4MTS_z / VoT_z | 可以立即开展静态等价性核对与迁移，不必从零编写代理模型 |
| 原生外部基线身份 | 尚未验证；VoT 明确代理 | 保留代理对照并准确命名；正式原方法比较另用公开实现 |
| 五域与更多域数据 | 已有；五域 merged 与旧副本一致 | 锁定路径/字段，优先现有五域，再条件性扩展 |
| 模型权重与环境 | 无 requirements/environment/pyproject；未打包 GPT2/BERT tokenizer/权重 | 可以重建依赖、复用已核验的本地 BERT；检查 GPT2 来源；不必索要所有历史 checkpoint |
| 原表数字与运行配置 | 未发现结果 CSV、log、npy/npz、运行 JSON；仅一个无元数据 state_dict | 旧表继续未追溯；新公平结果可以重跑，不必等旧日志 |
| 真实文本发布时间 | 仍仅见时间区间字段 | 寻找上游说明；无法取得时保留代理假设 |
| 论文 LaTeX | 包内无 tex/bib | 继续提供替换文本；最终排版整合需要作者源稿 |
| 协议可运行性 | 语法通过，未做 import/模型运行验证；已发现具体错位 | 先迁移到原点队列并检查，再做单配置试跑 |

## 8. 对五版方案的具体调整

1. A 证据优先路线：不再写“没有本地基线代码”。先建立原生/代理/适配版本差异表，固定任务和位置，再开展新公平比较。包里的代理 VoT 本身不需要 DeepSeek API，原生 VoT 的依赖另算。
2. B 精简路线：仍在已审计 revision 上做；新包的 PosGate、VICReg、可用性 EMA 是额外因素，不能默认混入同一消融。
3. C 文本价值路线：新增的 raw report/search 可用于明确渠道、覆盖和文本选择，但 Climate 等布局差异必须先解决。
4. D 延迟反馈路线：以补充材料 Algorithm 1 为契约，不延用已证实存在错位的批切片代码。
5. E 成本路线：对移植后的实际运行配置重新测量；不能沿用旧 190 次实验的数字描述 GPT2/VoT 新骨干。

最直接的工程顺序：冻结新 ZIP → 记录模型身份和数据映射 → 单独迁移 GPT4MTS_z → 对齐原点队列/目标掩码 → 单数据集单 seed 试跑 → 迁移并准确命名 VoT 代理 → 再决定原生基线和更多数据域的预算。

本次产物仅为检查报告、解压副本、PDF 文本/页面和证据 JSON。没有改写原附件，没有修改 WSL 修订模型，没有启动训练；因此不能把本报告称作已经完成复现。

# 本轮结果到主张判断

日期：2026-10-01。review_independence: same-family；acceptance_status: provisional；overall_claim_supported: partial。

本轮已有可以交付的局部进展：在已暴露的 Energy 和 SocialGood 开发流上，相同五维输入接口、相同融合结构及逐张量相同初始参数的 LLM5 简单候选，在线平均 MSE 分别比 raw BERT-PCA5 低 3.1356% 和 1.5431%。SocialGood 的小幅改善同样是有效的探索性进展，不因低于原 2% 工程阈值而归零。它没有同时击败两域数值及最强系统参考，也没有使原反馈刷新主张成立。

本审查只读既有结果与源码，未训练、未改源码。原 B1/B2/B3 的严格门槛及 B3 groundedness 停止结果保持原状；E003 属于用户修订后的探索支线，不将观察结果后选择的候选包装为预先确认。

## 判断及适用范围

| 主张 | claim_supported | confidence | 判断 |
|---|---|---|---|
| C1：相同带反馈预热权重下，部署反馈刷新在参数更新之外有增量预测价值 | no | high | U1F1 对 U1F0 两域平均 MSE 均变差；否定的是本轮平均增量表述，不能推断所有场景永远无效。 |
| 原 C2：受限表示在固定信息预算下为简单融合提供可推广、足以支撑原方法性能定位的内容增量 | partial | medium | 有局部内容与表示线索，但跨域结果冲突、强控制未全面胜出，原双域门槛未通过。 |
| C2-narrow：相同五维接口/结构/初始参数的 LLM5 简单融合，在这两条开发流的三 seed 均值上比 raw BERT-PCA5 简单融合误差更低 | yes | high（限定描述性事实） | 两域 MSE 与 MAE 均值均改善，逐 seed 方向完整报告；不等于稳定性、显著性或未见数据泛化确认。 |

### C1

**what_results_support：** 在同一个带反馈训练流预热起点上，U1F1 相对 U1F0 的平均 MSE 在 Energy 增加 1.5766%（1/3 seed 改善）、SocialGood 增加 0.4679%（0/3 改善）；MAE 分别增加 0.6595% 和 0.1504%。因此没有本轮平均意义上的部署反馈刷新增量。此结论来自差值方向，不是机械套用 2% 阈值。

完整联合在线系统相对自身静态仍有真实进展：Energy 平均 MSE 下降 28.7064%（3/3），SocialGood 下降 2.5329%（2/3）。这说明该起点的联合部署变化有收益；不能将收益单独归给 F，也不能省略数值强控制仍优于 full 的事实。

**what_results_dont_support：** 反馈缓存独立性能贡献、每个组件必要、普遍优于数值控制、整个训练到部署过程去除反馈的效果。四格都从已经使用反馈的预热状态出发，并未检验完全无反馈训练路线。

**suggested_claim_revision：** “在两条开发流中，联合在线部署优于对应静态 full；但在相同反馈预热起点上，部署期间刷新历史反馈未显示超过参数更新本身的平均增量。”

**missing_evidence：** 跨新数据的反馈增量证据，以及将训练期反馈与部署期反馈分别隔离的证据。

**next_experiments_needed：** 当前交付不需要追加实验。只有未来决定继续把反馈增量作为性能主张时，才另立锁定协议检验上述缺口；不能用再挑 seed 或延迟网格改写当前失败结论。

### 原 C2

**what_results_support：** B1 原 768 维简单文本在线在 SocialGood 对 numeric_online 的平均 MSE 降低 0.8542%（2/3），对自身静态降低约 2.7503%（3/3）；Energy 对 numeric_online 则增加 2.2090%。B2 PCA16 内容读出在 Energy 对 metadata P1 降低 4.4255%（3/3），历史 donor 平均更差，是局部内容相关线索；SocialGood 增加 6.6942%（0/3）。Energy 的 P2 MSE 0.0188262 仍高于裸骨干 0.0139274，不能把读出内部改善直接称为全系统竞争力。

B3 LLM5 在 Energy 对 metadata P1 平均 MSE 降低 1.1149%（3/3），donor 更差，满足用户修订后的局部继续探索条件。它同时比 raw PCA5 差 1.4361%，SocialGood 对 P1 差 5.4696%。规则候选虽有 Energy 内部改善，但 donor 平均更好，不能按本轮规则优先当作内容线索。由此选择 LLM5 进入 E003 有可追踪的探索依据。

**what_results_dont_support：** 原双域稳定内容增益门槛、文本语义已正确、任务压力被因果识别、一般性超越强基线、原五专家 full 获得了 E003 收益。B3 原 30 条预检的两处目标/时间语义错配仍成立，0 个格式 invalid 不等于 groundedness 正确。当前 probe 改动的是既定训练与在线历史下的当前文本，不能证明全部文本历史的因果贡献。

**suggested_claim_revision：** “受限文本的预测价值依赖表示与接入方式。本轮发现局部内容相关线索，并在固定五维简单融合中观察到两域相对 raw 表示的小幅平均改善；对数值强控制和在线适配自身的收益具有域差异。”

**missing_evidence：** E003 同融合结构中的 metadata/内容控制、足够强且一致的内容干预效应、未用于开发的新数据确认、明确真实发布时间、LLM 历史预训练信息风险识别。当前 n=3 是同一固定流上的优化随机 seed，不能视为三份独立时间数据。

**next_experiments_needed：** 先交付本轮。若未来验证候选，锁定这一个候选的表示、模型 revision、prompt、预算、结构、学习率和评分协议，再按预先顺序完成追加域及全部 seed；必要时加入该相同融合结构的 metadata 控制。原 B5 仍为未触发，未来另立固定验证计划；不凭最好 seed 扩域，不按新域结果选择策略，不自动跑满预算。

### C2-narrow：E003 两域均值改善

**what_results_support：**

| 比较：LLM5 online 减去控制 | Energy MSE 变化 | Energy MAE 变化 | MSE 改善 seed | SocialGood MSE 变化 | SocialGood MAE 变化 | MSE 改善 seed |
|---|---:|---:|---:|---:|---:|---:|
| raw PCA5 online | −3.1356% | −1.5271% | 2/3 | −1.5431% | −0.7951% | 3/3 |
| numeric online | +0.9915% | +0.3913% | 1/3 | −2.3358% | −1.5915% | 2/3 |
| 自身 LLM5 static | +0.6977% | +0.3836% | 1/3 | −1.6167% | −0.6891% | 2/3 |

变化率按“各配置三 seed 均值之比减一”计算，不是逐 seed 百分比的平均；负值表示误差降低。两种表示在训练前逐张量同初始参数，每种表示各自训练一次 warm，然后从其自身相同 warm 权重分叉 static/online；两种表示的 warm 权重不要求相同。

LLM online 减 raw online 的 MSE 配对差，Energy 为 −0.000460858，sample SD 0.000550600；SocialGood 为 −0.002048015，sample SD 0.001844604。Energy seed 2025 MSE 反向 +0.000001936，另外两 seed 改善；两域的 MAE 都是 3/3 seed 改善。这些数据支持“小幅且存在波动的局部进展”，不支持显著性用语。

LLM 当前文本 donor 减正确文本的平均 MSE：Energy +0.000001305（两 seed donor 反而更好），SocialGood +0.000172575（三 seed donor 均更差）。后者支持当前内容敏感性线索，前者效应很弱且方向不一致；它们不能单独解释全部表示收益。代码中 donor 只替换当前输入，沿用同一参数、特征及反馈快照，并返回原正确预测/回放上下文，未改变之后训练或历史反馈。

**what_results_dont_support：** LLM5 在 Energy 优于 numeric、在线过程在 Energy 改善其自身静态、跨所有 seed 稳定改善、原 full/MoE 有效、LLM 标签语义准确、总体 SOTA。Energy 裸骨干 MSE 0.0139274 仍低于 LLM5 online 0.0142365；SocialGood static GPT4MTS 0.0880900 仍低于 0.1306719。GPT4MTS 的文本历史和编码方式不同，是应展示的系统参考，不是同容量表示消融。

**suggested_claim_revision：** “在 Energy 和 SocialGood 两条已暴露开发流（OT、H=6、三优化 seed）上，保持五维接口、简单融合结构及初始参数一致，以固定小模型五类趋势表示替换 raw BERT-PCA5，在线平均 MSE 分别降低 3.14% 和 1.54%，平均 MAE 降低 1.53% 和 0.80%。该探索结果表明表示与接入方式有可继续研究的局部收益；对数值控制和自身静态的改善并不跨域一致。”

**missing_evidence：** 如果仅报告上述有限样本描述性结果，没有阻止本轮交付的缺失。若要改称稳定、泛化、显著或内容机制已证实，则需要新的独立证据；再加同一流优化 seed 不能替代未见数据。

**next_experiments_needed：** 本轮无需追加。未来锁定候选验证与本轮探索分表，保持所有原失败域及 seed；不把未来计划写成已获得的支持。

## 来源核对与质量边界

已读取探索修订、E003 固定计划、研究契约与原实验计划。确定性 precheck 的 12 项 verified 仅表示值存在，不能当作语义通过。审查另以 CPU 只读重算了 E003 的 24 正式预测和 12 donor 序列：共 36 条均与原 CSV 目标/掩码、报告 MSE/MAE 相符；Energy 143 原点/858 目标元素，SocialGood 85 原点/510 元素。6 对表示初始权重逐张量一致，12 份 static 最终权重与自身 warm 一致，12 条在线序列的成熟原点只消费一次，成熟时刻、原文本 ID、回放原点与首 7 个更新前预测核对通过。

本次未发现项目中的 EXPERIMENT_AUDIT.json，因此 integrity_status=unavailable；已有 B1/B2/E003 验证报告和本次有限只读核对提供上述具体证据，不能替代全面完整性审计，也不因此阻断本轮探索结果交付。所有语义结论仍是 same-family/provisional。

共同限制：60% 训练、60–70% 选择、70–80% 开发；这些开发域已多轮暴露，候选准入属于观察结果后的探索。没有 p 值或独立新数据确认；end_date 是可用性代理，不是核实的 publication time；固定小模型可能包含历史预训练信息，这一风险尚未识别。原 190 矩阵与五域外部系统证据保持独立，不混合胜率。

## 成本事实与可用表述

18 轮成本已结束，每域/方法 3 次计时重复，使用 seed 2023 的 warmed 模型，每轮新建隔离副本。每轮 50 warm-up、200 次缓存输入预测及 200 次固定成熟记录更新，更新包含反馈计算和刷新；同步已计入，队列、日志、输入加载、文本编码/生成未计入。不是部署吞吐，也不是 18 个模型随机 seed。

| 域 | 方法 | 预测 median / P95 (ms) | 更新 median / P95 (ms) |
|---|---|---:|---:|
| Energy | raw PCA5 | 9.421 / 10.816 | 12.963 / 15.999 |
| Energy | LLM5 | 3.700 / 6.458 | 7.629 / 10.445 |
| Energy | numeric | 4.735 / 10.381 | 6.418 / 8.475 |
| SocialGood | raw PCA5 | 3.769 / 6.314 | 13.117 / 15.560 |
| SocialGood | LLM5 | 3.331 / 9.318 | 9.824 / 15.389 |
| SocialGood | numeric | 8.084 / 9.476 | 10.686 / 13.651 |

表中每列是三个独立计时轮次对应统计量的中位数，P95 列不是汇集全部 600 次调用后的 P95；完整 18 轮保存在 profile_output/e003_20261001/rounds.csv。共享 RTX 4060 Laptop GPU 波动明显，同结构 raw/LLM 的时延差不能推断 LLM 表示本质更快。

raw PCA5 与 LLM5 可训练参数均为 21,632，numeric 为 10,496；原 768 维 text_add 为 94,880，原 full OT 为 166,832。小投影减少可训练参数是直接事实，但不等于端到端零开销：Qwen 权重文件 988,097,824 bytes（约 0.99 GB），全量生成 1,538 个 unique 表示耗时 95.927 秒、invalid=0，生成峰值 allocated 1,016,280,064 bytes；含预检及失败诊断的累计预算记账 114.590/3600 秒。原 groundedness 停止保留，FP16 batch8 的非有限失败已保留诊断，batch1 有限后完成固定生成，没有因开发误差搜索 prompt。

## 当前路由

交付本轮完整结果、局部进展和未满足的原主张；不将 partial 改为 broad yes，不因原阈值未通过抹掉 E003。原 B4/B5 未触发，E003 完成。未来追加域属于锁定候选后的单独验证，不是现在选最好 seed 后自动扩域。当前没有需要新训练、更多网格或外围审批才能完成的交付缺口。

主要证据：experiments/replan_20260930/final_review_packet/{facts,precheck}.json；b1/reports、b2/reports、b3/reports；exploratory_fusion/reports/{runs,paired_runs,paired_summary,summary,reference_summary,validation}；其原预测文件；iadapter/revision_candidate.py 与 shared_online.py；profile_output/e003_20261001/{rounds,summary}.csv 与 validation.json。

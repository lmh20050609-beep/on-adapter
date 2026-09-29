# iAdapter 五路线试验跟踪表

## 当前状态（覆盖下方历史 PLANNED 表）

2026-09-29，证据见 [五 seed 决策与索引](../../refine-logs/review_sprint/REVIEW_UPDATE_FIVE_SEED.md)。原 190 次结果不变。

| 项目 | 实际状态 | 证据与剩余范围 |
|---|---|---|
| M0-01/02/03 | 完成 | 两次端到端 sanity；同状态诊断轨迹/RNG 不变检查与代码审查通过 |
| B-01 | 完成（三 seed） | 42 次开发评估包含完整 2×2；交互未扩到五 seed |
| B-02 | 完成，候选扩展到五 seed | 新增 20 次，复用其中五方法的原 30 次，合并 50 单元；保留 Energy 新 seed 反转 |
| C-01 | 完成，门槛未满足 | 80 个同状态文本诊断序列；没有稳定正确内容优势 |
| C-02/03 | 未触发 | 依照 C1 门槛不追加渠道变体/接入网络 |
| E 成本测量 | 完成 | 两域×五方法×三轮=30 组；缓存/更新/新文本分开计量 |
| E-01 投影压缩 | 完成，候选未通过两域门槛 | 新跑12次、复用12次；投影参数80160→6856。仅投影压缩Energy MSE+2.37%；压缩+单MLP SocialGood MSE+1.32%。追加24组同轮成本完成；不扩宽度/seed搜索。见 `refine-logs/review_compression/REVIEW_FINDINGS.md` |
| D-01 | 条件性待执行 | 未新增延迟或年龄机制，不声称已有鲁棒性证据 |
| A-01/02/03/04 | 部分静态审计，运行未完成 | 已区分 GPT4MTS 改编和 VoT 代理；原生公平比较仍缺 |
| F-01 | 尚未完成 | 五 seed 仍是开发数据，不能替代独立数据确认 |
| W-01 | 逐点回复草稿已更新 | 最新 `REVIEW_RESPONSES_FIVE_SEED.md`；未集成整篇 LaTeX |

## 最初计划记录

日期：2026-09-29。全部新增项为 **PLANNED**，本次未启动实验。已有 190 次结果仅作为假设来源与历史证据，不作为新路线的独立确认。

| ID | 阶段 | 目的 | 方法/变体 | 数据与划分 | 主指标 | 优先级 | 状态 | 启动或停止条件 |
|---|---|---|---|---|---|---|---|---|
| M0-01 | Sanity | 固定新边界、目标与原点 | numeric_add | Economy/H6/2023；开发协议 | 原点/掩码/指标一致 | MUST | PLANNED | 通过后运行 M0-02 |
| M0-02 | Sanity | 骨干冻结、文本缓存、相同范围 | full | 同 M0-01 | 冻结与未来不变性 | MUST | PLANNED | 不修改正式旧结果 |
| M0-03 | Sanity | 文本反事实不污染状态 | full：正确/训练均值 | 同一原点当时状态 | 后续预测与参数一致 | MUST | PLANNED | 通过才进入 C1 |
| B-01 | 结构 | 路由文本梯度×entropy 交互 | full/route-off/entropy-off/both-off | Energy+SocialGood；H6；3 seed；开发流 | paired MSE/MAE、种子波动 | MUST | PLANNED | 不根据旧 test 选超参 |
| B-02 | 简洁性 | 单 MLP 是否足够 | single-MLP、numeric_add、text_add | 同 B-01 | 精度/参数/时间 | MUST | PLANNED | B 总计 42 个单元 |
| C-01 | 内容诊断 | 文本内容与分支结构分离 | full/text_add×正确/均值/donor/零 | 同 B；48 个只读预测序列 | 预测变化与 paired error | MUST | PLANNED | donor 仅来自历史；无测试末权重回放 |
| C-02 | 来源诊断 | fact 与 preds | text_add：fact-only/preds-only | 两域×3 seed；最多 12 次 | 覆盖、截断、配对误差 | CONDITIONAL | PLANNED | C1 有可用信号才执行 |
| C-03 | 接入策略 | 成熟反馈加权是否优于固定权重 | numeric/text 两路输出融合 | 开发期固定学习率/常数权重 | 相对数值基线退化与收益 | OPTIONAL | PLANNED | 不宣称安全保证；不纳入首轮预算 |
| E-01 | 效率 | 压缩投影×单专家 | 原/8维投影×五/单专家 | 两域×3 seed；≤24 次 | 精度容差、参数、延迟P95 | NEXT | PLANNED | B 控制完全一致才复用 |
| D-01 | 反馈 | 固定H隔离额外延迟 | full/无反馈/常数减幅/年龄衰减 | H6；d=0/6/12；两域×3seed；≤72次 | 延迟退化、paired error | CONDITIONAL | PLANNED | 仍以延迟机制为主贡献时执行 |
| A-01 | 外部协议 | 对齐共同预测任务 | GPT4MTS/VoT/iAdapter | 共同 OT 单变量任务；另表 | 输入/目标/信息一致 | MUST | PLANNED | 保留原生结构，静态在线分开 |
| A-02 | 外部接入 | GPT4MTS 最小运行 | 原生 GPT4MTS + iAdapter静态对照 | Economy/H6/1seed；2 单元 | 对齐预测与评分 | MUST | PLANNED | 权重/代码版本确定 |
| A-03 | VoT来源 | 核验事件生成与历史检索 | 原生 VoT 或明确 VoT-local | 时间来源可追溯 | 信息泄漏、生成成本 | MUST | PLANNED | 缺链路先记录，不读来源不明prior |
| A-04 | 外部主表 | 重新建立公平比较 | 数值适配静态/iAdapter静态/GPT4MTS/VoT | 五域×5seed×4=100单元 | MSE/MAE/SD/成本 | CONDITIONAL | PLANNED | A1–A3 通过；不按旧适配时间承诺工期 |
| F-01 | 独立确认 | 检验锁定候选是否泛化 | 至多两个候选+强对照 | 新时段或未用于设计的数据；5seed | 预先声明主效应及失败域 | MUST FOR STRONG CLAIMS | PLANNED | 数据未取得则保留探索性标签 |
| W-01 | 写作 | 选择一条主线并逐点回复 | A+B/C/D/E之一 | 所有成功与失败记录 | 主张与证据对应 | MUST | PLANNED | 不将五条路线都堆成主贡献 |

最小本地 B–E 未去重预算：≤150 次开发流评估 +48 个只读诊断序列；不含 M0、可选 C3、外部模型与独立确认。试跑计时后才能填 GPU 小时。

结果记录至少包含：完整配置、代码/数据版本、开发/确认标签、所有尝试、失败状态、原点与文本 ID、成熟事件、原始预测、掩码、初始化共享方式、误差与成本。不用最优运行覆盖同配置的失败记录。

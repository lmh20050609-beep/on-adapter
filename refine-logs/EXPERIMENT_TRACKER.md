# iAdapter 下一轮计划跟踪表

2026-10-01。已完成本轮必做及用户修订探索。M0/B1/B2已完成，30新评价+30复用控制与18读出任务通过审计；B1两项两域门槛未通过，B2仅Energy内容门槛通过，因此B3触发。B3原质量停止保留；用户修订后1538生成、18读出任务和24新简单融合评价完成，62测试通过。历史结果保持原状态。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R000 | M0 | 输入/来源审计 | 两域文本覆盖、可见token、固定OT来源 | 60/10/10 | ID/token、缺文类型、覆盖 | MUST | DONE | 不训练；因果问题先修复 |
| R001 | M1 | 第一单元 | Energy/2023 frozen_backbone | OT H6开发 | MSE/MAE/原点mask | MUST | DONE | 现有骨干，无adapter |
| R002 | M1 | 简单文本静态 | Energy/2023 text_add_static | 同上 | MSE/MAE | MUST | DONE | 新warm，现有骨干/缓存 |
| R003 | M1 | 简单文本在线 | Energy/2023 text_add_online | 同上 | MSE/MAE/更新事件 | MUST | DONE | 与R002同warm起点 |
| R004 | M2 | 反馈增量 | 两域三seed U1F0/U0F1 | 60/10/10 OT H6 | 配对误差/参数/缓存 | MUST | DONE | 12新评价；另12因子控制复用 |
| R005 | M2 | 剩余强简单控制 | 两域三seed裸骨干/text_add两mode | 同上 | 误差与失败域 | MUST | DONE | 含R001–3共18新评价；6warm |
| R006 | M3 | 可用性与内容 | P0/P1/P2 Ridge选择 | 同上 | MSE/MAE/配对方向 | MUST | DONE | 18选择任务、至多54lambda fits |
| R007 | M3 | 正确内容干预 | P2历史donor，metadata不改 | 同上 | 相对正确content误差 | MUST | DONE | 6序列，无另拟合 |
| R008 | M4 | 小模型资源与格式 | Qwen0.5B训练文本30条 | 只用训练文本 | 格式/领域/生成成本 | CONDITIONAL | DONE_FORMAT_STRICT_QUALITY_STOP | 30条格式全有效，2条明确目标误判；原始质量报告保留 |
| R009 | M4 | 新表示检验 | BERT-PCA5/规则/LLM | 同上开发 | 三表示读出误差 | CONDITIONAL | NOT_TRIGGERED | 18选择任务、至多54fits；生成≤1543 |
| R010 | M4 | 新内容干预 | LLM表示历史donor | 同上 | 内容效果 | CONDITIONAL | NOT_TRIGGERED | 6序列，无另拟合 |
| R011 | M5 | 部署验证 | 同结构raw/candidate×static/online | 两域三seed OT H6 | 全部控制误差/位移 | CONDITIONAL | NOT_TRIGGERED | 至多24评价、12warm；raw路线通过可复用B1 |
| R012 | M5 | 原点内容作用 | 两表示同状态donor | 同上 | 干预误差/预测变化 | CONDITIONAL | NOT_TRIGGERED | ≤12序列；不替代真实对照 |
| R013 | M5 | 锁定候选成本 | candidate/raw/numeric，各三轮 | 隔离副本 | 参数/med/P95/生成/缓存 | CONDITIONAL | NOT_TRIGGERED | 18轮，50warm/200measure |
| R014 | M6 | 新域来源审计 | Agriculture/Health身份/OT/文本/时间 | 尚未评价新域 | schema/provenance | CONDITIONAL | NOT_TRIGGERED | CSV存在不代表身份核验通过 |
| R015 | M6 | 追加域验证 | 骨干/数值/raw/candidate静态/在线 | 两新域60/20/20，五seed | MSE/MAE/失败域 | CONDITIONAL | NOT_TRIGGERED | ≤50评价、10骨干、30warm；raw=candidate则40 |
| R016 | M7 | 修订与主张判断 | 更新正文/回复/证据缺口 | 所有已触发块 | claim/evidence对应 | MUST | DONE | 正向小收益和失败域均交付；主张核对完成 |

默认新评价30（含R001–R005，不能把R001–3再加一次），6warm；B2 18选择任务；6干预。条件性上限104评价/48warm/10骨干/36读出选择任务/24干预/1543生成/18计时轮，类别不混计。

## 用户要求后的探索工作

| E ID | 工作 | 状态 | 边界 |
|---|---|---|---|
| E001 | 同固定小模型全量表示生成 | DONE_EXPLORATORY | 1538unique；累计114.60秒，原stopped_groundedness保留 |
| E002 | 规则/raw/LLM与metadata及历史donor读出 | DONE_EXPLORATORY | 18选择任务/54配置、36评分；EnergyLLM对metadata−1.11% |

| E003 | 同5维简单融合raw/LLM×static/online | DONE | 24新评价/12warm/12donor；MSE两域对raw−3.14%/−1.54% |
| E004 | 锁定LLM/raw/numeric候选成本 | DONE | 18隔离轮；生成成本另列 |

原R009/R010/R011–R015按原严格门槛未触发；新增E001–E004是用户修订探索，不将旧门槛改写为passed。追加域、近期文本优先为后续锁定验证方向。本轮合计54新评价、30复用、18warm、36读出选择任务、108lambda配置、24donor序列/30条件、1538unique、18cost、零新骨干。

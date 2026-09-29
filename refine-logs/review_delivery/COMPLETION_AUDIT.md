# 交付验收：本轮证据包与未解决事项分开核验

验收依据：用户批准的“先修实现、再建立五数据集小闭环”方案、后续A–E有限路线及当前目标“借助代码尝试回答review、优化实验”。该方案明确允许负结果，要求缺来源的旧表撤下，缺LaTeX时交付替换文本；不能把实验完成理解为论文已获支持或所有review已解决。下表覆盖具体前置条件、矩阵、检查和交付物。文件指纹与自动断言见 completion_evidence.json。

## 原批准方案逐项证据

| 要求 | 核验范围与证据 | 判定 |
|---|---|---|
| 唯一工作目录、保护历史材料 | 当前/home/wuwa/projects/research仅iadapter；原目录依用户后续指示删除；docs/BACKUP.md、PATH_MIGRATION.json与备份校验 | 完成；后续明确指令替代最初保留目录要求 |
| 模型/执行/汇总分离，导入不训练 | legacy_core定义提取；models.py、engine.py、run.py和analyze.py分离；CLI有入口保护 | 完成；投稿原包另存，不导入其顶层训练 |
| 逐原点先预测后成熟更新 | engine.Replay/MatureQueue/Stream；新增shared_online；保存原点文本/反馈上下文，使用原预测评分 | 完成；51测试及190+75原CSV审计支持 |
| 原点对应、无漏用/重复、无末尾flush | 原190与在线75更新日志；H=1/6/8/12/20、chunk=1/8/16、长度51回归 | 完成；覆盖H大于/等于/小于chunk和末尾不足批次 |
| 未来文本/标签不影响当前预测，未来输入不影响历史填充 | test_future_context、test_protocol、test_shared_online的具体前缀不变性检查 | 完成于已测执行路径；不扩张为真实文本来源无泄漏 |
| 冻结完整BERT、attention mask、池化、缓存/投影 | prepare.py实际调用12层BertModel，五域编码记录及单条直接编码一致性；engine重用冻结表示并重算投影 | 完成；记录验证为每域前两条直接编码抽查，不称全部文本双算 |
| 无文本数值方法不消费文本；无效文本投影置零 | uses_text路径和NaN embedding数值对照测试；no_text投影测试 | 完成 |
| 因果数据、训练前缀标准化、缺失目标掩码 | data.py及独立原CSV审计；固定原单位零用于开头缺失；input_missing保存 | 完成；end_date仅代理 |
| 训练/验证职责，训练流预热，评价时缓存队列重置 | run.fit_backbone选择checkpoint不训练验证；warm Adapter训练流一遍；每次Stream新优化器/缓存/队列 | 完成；无验证状态带入 |
| 骨干不变、适配器可更新 | run.py逐张量冻结断言；test_protocol和shared_online；原运行完成状态 | 完成；审计不冒称重新训练权重 |
| 统一variant/run_dir、完整配置、原预测/掩码、ID、门控和耗时 | config.py/run.py保存结构；190份原run、75在线run、源提交及状态 | 完成；旧190历史梯度tensor未全部落盘，内存快照与回放有代码/测试支持；新增在线已保存tensor |
| B主矩阵75 | runs.csv精确键集与config.matrix比对；每个五域H6三方法五seed | 完成 |
| C两域八消融80、Security路由5 | 同精确190键集；component_verdicts与ablation_table；消融路径不变性测试 | 完成；不将全部必要性作为通过条件 |
| D两域H12共30 | 精确键集、35骨干配置；同原CSV审计 | 完成；不代表任意horizon |
| 机制与分组分析 | change_scores仅历史窗口；验证阈值独立重算；state_paired_analysis；Energy/SocialGood完整流图标记文本/成熟 | 完成描述性分析；无概念漂移真值或因果主张 |
| 效率三方法两域 | efficiency.json六组50预热/200测量；med/P95、参数、编码、显存、CPU/队列/反馈口径 | 完成；孤立副本，非完整部署吞吐 |
| 失败/缺件/冲突不能进入汇总 | test_artifact_audit篡改目标/掩码/原点/状态与缺件拒绝；审计先于分析 | 完成；不取首行或最好seed |
| 原指标可重算、原结果不改 | 本次campaign --audit-completed 190/190通过且不改文件时间；原始CSV目标对照 | 完成；日志见original190-final-readonly-audit.txt |
| 正文替换、逐点回复、表图、缺项清单 | REVIEW_RESPONSE、MANUSCRIPT_REPLACEMENTS、REVIEW_DECISIONS与映射表；原reports表/图/附录 | 完成本轮交付；整篇稿件未整合或提交 |

## 后续路线与停止条件

| 路线/明确要求 | 当前证据 | 判定与边界 |
|---|---|---|
| B因子/单MLP及额外seed | 42初始+20新增；combined_validation，三seed因子、五seed候选 | 完成；新增Energy反转保留 |
| C同状态文本诊断 | 80序列，历史donor、均值/零控制，reference轨迹/RNG不变测试 | C1完成但门槛失败；C2/C3不触发 |
| E压缩与成本 | 12新增+12控制，六组合精度/参数门槛与24计时 | 完成有限检验；没有双域通过候选，不扩搜索 |
| A作者模型及共同任务 | pinned GPT源码/权重、真实模型smoke、75静态单元与25原CSV组 | 完成GPT部分；文本消费与训练差异披露 |
| A2共同延迟执行器 | 75在线与75静态同起点；25组核验，前7预测相同；未来不变/缺失标签/冻结检查 | 完成；GPT为明确新在线扩展 |
| 保留强静态控制 | all_static_online_runs包含150唯一单元；all_controls表覆盖全部五个对照 | 完成；不是只选退化GPT在线比较 |
| 独立结论核对 | fresh gpt-6-astra ultra；原CSV/150预测重算；partial/high | 完成临时同家族复核；无完整性审计认证 |
| D额外延迟网格 | 条件为继续以延迟鲁棒性作主要贡献；本轮不保留此主张 | 未触发；不可标成已验证鲁棒性 |
| F独立确认 | 尚无未用于设计的新数据/时段 | 未完成，仅限制更强泛化主张；不以更多同流seed替代 |

## 未完成、不能被本轮验收消除的缺口

原Table1–3源记录、VoT生成与原点可用性链路、真实publication-time、独立新数据确认、整篇LaTeX整合/编译均未完成。原A四方法100单元没有完成，实际只报告三方法75静态和75在线。本文不把这些缺项标为通过；用户批准的有限证据包允许撤销相应声明并明确缺口。若以后要求恢复原广泛结果或完整投稿稿件，这些仍是待办。

## 最终处置

本轮代码、预定有限实验、协议检查、实际结果/失败、成本、机制图、正文替换与逐点回复均有对应可读证据，允许结束本轮尝试回答review的工作。没有用缩小已批准矩阵或删除失败结果来满足验收。科学主张判断仍为partial；不宣布论文审稿问题全部解决或创新有效性已获证明。失败分支按原规则停止，不继续无目的训练。

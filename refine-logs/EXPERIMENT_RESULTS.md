# iAdapter 修订计划执行报告

2026-10-01。按顺序完成原计划必做块，并根据用户“允许小幅、波动提升，继续大胆探索”的要求完成固定小模型表示与简单融合探索。达到本轮明确范围后收束；追加域是下一阶段，不假称已完成。

## 最值得保留的进展

**新的5维LLM表示简单融合，相对同结构5维BERT-PCA在线融合，两域的平均MSE和MAE均降低。** Energy MSE降低3.14%、MAE降低1.53%，MSE在2/3seed改善；SocialGood分别降低1.54%与0.80%，MSE在3/3seed改善。这是可写入论文的限定开发结果，不因低于某个工程门槛被抹掉。

相对数值在线适配，SocialGood平均MSE降低2.34%、MAE降低1.59%，MSE在2/3seed改善；Energy平均MSE高0.99%、MAE高0.39%，只有1/3seed改善。Energy的候选静态版比在线版更好；SocialGood在线版相对自身静态降低1.62% MSE。因而表示收益与在线更新收益分别报告。

原完整iAdapter在线相对自身静态平均MSE降低28.71% Energy、2.53% SocialGood，体现联合适配的条件性作用。进一步隔离显示，部署时反馈刷新并未带来超过参数更新本身的平均增益。这限制一个机制解释，但不取消前面的适配改善。

## 完成顺序与实际数量

| 工作 | 实际完成 | 结果/分支 |
|---|---|---|
| M0输入审计 | 1903原点 | 缺文、字符裁剪、可见token与来源ID核对 |
| B1同任务强控制及U×F | 30新评价＋30精确复用，6warm | 新增裸骨干/简单文本/独立反馈策略，原CSV评分通过 |
| B2 metadata/content | 18读出选择任务、54lambda配置、6donor | Energy有局部内容增益，SocialGood退化 |
| B3原严格预检 | 30训练文本 | 格式全合格；两条明确语义误判，原stopped_groundedness保留 |
| E001–E002固定表示探索 | 1538unique生成、18选择任务/54配置，6联合donor序列 | 用户修订准入；模型/prompt/内容预算未变，Energy LLM对metadata MSE−1.11%，3/3seed |
| E003简单接入 | 24新评价、12warm、12同状态donor | 同5维结构、同初始化；两域均值优于raw在线 |
| E004成本 | 18隔离计时轮 | 每轮50warm/200measurement，生成成本单列 |
| M7表图/正文/回复/主张核对 | 已交付 | 62测试通过；fresh独立结果核对完成 |
| 原B4/B5严格分支 | 未触发 | 不改写旧门槛；E003是显式探索，追加域未执行 |

共54次新神经适配评价、18warm、36读出选择任务、108lambda候选配置、零新骨干、18成本轮。每lambda按6个masked输出求解，不能将输出求解当独立模型重复。B3两个donor条件共享6条源原点序列；累计24条donor序列/30个条件评分。所有原始预测、warm权重、失败日志与可见文本来源保留在WSL独立实验目录。

## 协议与证据范围

Energy/SocialGood，OT单变量，H6，seeds2023/24/25，L48/36。前60%训练、60–70%选择、70–80%开发；此前已暴露，不能称独立未见测试。骨干复用相同seed的minimal ITransformerZ d32/2层/batch16 checkpoint。原点先预测，再消费一次成熟目标；回放原文本与反馈上下文；骨干冻结、原缺失目标mask，末尾不flush。

BERT与LLM读取同一M0可见字符串；LLM额外仅固定目标描述。PCA、标准化、Ridge只fit训练前缀。新5维raw/LLM首层与所有同形状参数初始完全一致，每份warm状态分叉static/online，探针不改变正式原预测/参数/历史反馈。原190主表、五域外部比较和本轮开发表分开，不拼胜率。

## 新候选主表

表1：三seed平均值及sample SD，全部完整结果如下；donor为同状态探针，不是独立训练。

| dataset | representation | mode | n | mse_mean | mse_sd | mae_mean | mae_sd |
|---|---|---|---|---|---|---|---|
| Energy | LLM5 | donor_same_state | 3 | 0.014238 | 0.000167 | 0.090381 | 0.00022 |
| Energy | LLM5 | online | 3 | 0.014237 | 0.000164 | 0.090377 | 0.000205 |
| Energy | LLM5 | static | 3 | 0.014138 | 0.000327 | 0.090032 | 0.000785 |
| Energy | raw_BERT_PCA5 | donor_same_state | 3 | 0.014685 | 0.000665 | 0.091735 | 0.001659 |
| Energy | raw_BERT_PCA5 | online | 3 | 0.014697 | 0.000694 | 0.091779 | 0.001747 |
| Energy | raw_BERT_PCA5 | static | 3 | 0.014912 | 0.001698 | 0.093366 | 0.006586 |
| SocialGood | LLM5 | donor_same_state | 3 | 0.130844 | 0.01516 | 0.208336 | 0.012311 |
| SocialGood | LLM5 | online | 3 | 0.130672 | 0.015281 | 0.20818 | 0.012376 |
| SocialGood | LLM5 | static | 3 | 0.132819 | 0.017523 | 0.209625 | 0.014407 |
| SocialGood | raw_BERT_PCA5 | donor_same_state | 3 | 0.132773 | 0.014102 | 0.209882 | 0.01163 |
| SocialGood | raw_BERT_PCA5 | online | 3 | 0.13272 | 0.014202 | 0.209849 | 0.011744 |
| SocialGood | raw_BERT_PCA5 | static | 3 | 0.133779 | 0.016167 | 0.210152 | 0.013364 |

表2：LLM在线相对各控制的均值变化（负数表示误差降低）。

| dataset | control | mse_change_percent | mae_change_percent |
|---|---|---|---|
| Energy | raw_online | -3.1356 | -1.5271 |
| Energy | numeric_online | 0.9915 | 0.3913 |
| Energy | LLM_static | 0.6977 | 0.3836 |
| SocialGood | raw_online | -1.5431 | -0.7951 |
| SocialGood | numeric_online | -2.3358 | -1.5915 |
| SocialGood | LLM_static | -1.6167 | -0.6891 |

全部seed及配对SD：[E003原始行](replan_execution/evidence/E003/runs.csv)、[配对差值](replan_execution/evidence/E003/paired_runs.csv)、[配对汇总](replan_execution/evidence/E003/paired_summary.csv)。强裸骨干、数值static/online、full在线及static GPT见[参考表](replan_execution/evidence/E003/reference_summary.csv)。SocialGood static GPT MSE0.08809仍优于新候选0.13067；Energy裸骨干0.01393仍优于0.01424。GPT是文本历史/架构不同的系统参考，不能称完全等容量干预。

## 文本与机制得到的启发

SocialGood简单文本原768在线对数值平均MSE−0.85%是小幅进展。Energy BERT-PCA16内容读出对metadata−4.43%且3/3seed改善，是局部可读出信号；SocialGood却+6.69%。固定LLM5读出Energy对metadata−1.11%、3/3seed改善而仍不胜rawPCA5，经过简单接入后两域均值才胜同容量raw。读出与实际在线接入不是同一个结果，不能互相替代。

新候选当前donor平均误差增量很小：Energy约1.3e−6，且两个seed的donor反而更好；SocialGood约0.000173，三个seed donor都更差。它们提供有限的当前内容敏感性分析，不证明因果语义或完全解释训练历史中的表示收益。两个LLM语义误判仍保留，实际改善说明有噪声特征可以有预测价值。

Energy开发139/148最新选中记录没有内容token进入BERT，SocialGood为16/90。下一种最具体改进是相同64token下近期文本优先。训练缺文变化发生在已提供原文层面，尚无上游原因证明；不以未来文本补齐训练。详见[输入审计](replan_execution/INPUT_AUDIT.md)与[后续方向](replan_execution/PROGRESS_AND_NEXT_ROUTES.md)。

## 成本

新的raw/LLM适配器均21632可训练参数，数值适配10496；相对同OT原768投影text_add的94880参数减少77.20%。这支持在线可训练部分更小，不意味着冻结LLM总模型更小。Qwen公开权重约0.99GB；1538unique输出的全量生成95.93秒，包含预检、失败及诊断的累计预算墙钟约114.59秒。生成与raw BERT编码/PCA的来源分开记录。

表3：三轮中位数；各轮先50预热，再测200次。P95列是各轮P95的中位数，峰值列为三轮allocated最大值。

| dataset | method | rounds | parameters | backbone_parameters | prediction_median_ms | prediction_p95_ms | update_median_ms | update_p95_ms | peak_allocated_bytes_max | text_array_cache_bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| Energy | LLM5 | 3 | 21632 | 27238 | 3.7 | 6.458 | 7.629 | 10.445 | 18520576 | 23660 |
| Energy | numeric_online | 3 | 10496 | 27238 | 4.735 | 10.381 | 6.418 | 8.475 | 18356224 | 0 |
| Energy | raw_BERT_PCA5 | 3 | 21632 | 27238 | 9.421 | 10.816 | 12.963 | 15.999 | 18520576 | 23660 |
| SocialGood | LLM5 | 3 | 21632 | 26854 | 3.331 | 9.318 | 9.824 | 15.389 | 18507776 | 14400 |
| SocialGood | numeric_online | 3 | 10496 | 26854 | 8.084 | 9.476 | 10.686 | 13.651 | 18352640 | 0 |
| SocialGood | raw_BERT_PCA5 | 3 | 21632 | 26854 | 3.769 | 6.314 | 13.117 | 15.56 | 18507776 | 14400 |

[全部18轮](replan_execution/evidence/cost/rounds.csv)保留。本机共享RTX4060 Laptop计时波动较大，raw/LLM在线结构完全相同，不从观测时延差推断LLM表示本质更快。缓存预测和固定记录更新包含同步、投影/骨干/梯度/Adam/反馈刷新，排除输入加载、队列与日志，不能视为完整部署吞吐；CPU缓存只报数组payload。没有改动正式结果状态，见[仪器记录](replan_execution/COST_INSTRUMENTATION.md)。

## 论文与review已如何修改

新主张是“限定开发任务下，固定小模型的低维表示改善了同容量简单融合”，不把结果归给原五专家full，也不宣称普遍SOTA。小幅、局部、波动结果正面呈现，全seed与退化同时给出。反馈性能贡献降级，参数适配本身的作用保留。原2%两域筛选结果及原质量停止保留，用户修订探索单列。

[完整论文替换文本](review_delivery/MANUSCRIPT_REPLACEMENTS.md)、[逐点review回复](review_delivery/REVIEW_RESPONSE.md)、[主张判断](replan_execution/CLAIMS_FROM_RESULTS.md)及图源已更新。新窄主张获得same-family/provisional评审支持，原广泛C2仅partial、部署反馈增量C1=no。

真正publication-time、可信原生VoT链路、原Tables1–3匹配来源、未用于设计的新时段和完整LaTeX仍未补齐。它们限制更广泛的论文主张，不阻止记录这轮实际改善。下一阶段先锁定一个表示/接入配置，再验证追加域；本轮没有按最好seed继续扩域或开大网格。


新接入配对图：[可编辑图源与数据](replan_execution/figures/plot_candidate.py)、[PNG预览](replan_execution/figures/E003_progress.png)、[PDF](replan_execution/figures/E003_progress.pdf)。

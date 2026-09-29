# Review证据当前入口

2026-09-29。此前`REVIEW_COVERAGE.md`是190次在线矩阵阶段的历史验收，包含当时尚未获得补充材料的判断。现已导入投稿代码、取得官方基线源码并完成新增研究；当前状态如下。

| 工作 | 当前证据 | 结论边界 |
|---|---|---|
| 原批准在线修订矩阵 | 190/190评估、35个骨干；reports目录；本次只读原CSV复核再次通过 | 五域H6、两域H12；不是原十域结果恢复 |
| 组件与文本诊断 | 42个初始开发单元+20个新seed单元；80个同状态文本诊断序列 | 五候选五seed；完整2×2仍三seed；内容有效性门槛未满足 |
| 组件成本 | 30组三轮隔离测量 | 单MLP比full省参数/时间，但对简单文本控制的额外收益有限 |
| 投影压缩 | 12新增+12复用单元、24组成本；refine-logs/review_compression | 没有候选同时通过两域预定精度/参数门槛，停止该分支扩展 |
| 官方GPT4MTS接入 | 固定作者提交、预训练权重、batch2等价/batch1修复/真实目标梯度检查 | 作者模型移植可运行；不等于原benchmark完整复现 |
| 新共同OT静态表 | 五域×五seed×三方法=75/75；其中72新增、3复用pilot；25组原CSV审计通过 | static iAdapter MSE最低0/5、MAE最低1/5；文本消费和训练阶段不同 |
| VoT | 官方源码与10份prior已审计，投稿包代理身份已区分 | 生成材料、原点与可用时间来源缺失，未用于排名 |
| 方法、图和回复 | reports/method_appendix.md、机制图；各refine-logs阶段的中英文回复草稿 | 无整篇LaTeX源，尚未整合或提交完整论文 |

阅读顺序：

1. [静态五域五seed结果](../refine-logs/external_baselines/STATIC_MATRIX_FINDINGS.md)与[回复更新](../refine-logs/external_baselines/STATIC_MATRIX_REVIEW_RESPONSE.md)。
2. [组件五seed决策](../refine-logs/review_sprint/REVIEW_UPDATE_FIVE_SEED.md)。
3. [压缩路线](../refine-logs/review_compression/REVIEW_FINDINGS.md)。
4. [基线来源与缺口](../refine-logs/external_baselines/INTEGRATION_REPORT.md)。
5. [原190矩阵交付说明](HANDOFF.md)，用于原在线结果与方法段落。

以上各表的任务、数据划分和是否在线更新不同，不能合成一个总体胜率。当前不支持普遍SOTA、全部组件必要、语义专家分工、真实publication-time无泄漏或显著优越性。end_date始终是代理；旧Table1–3来源问题仍未恢复。

下一项优先工作是共同延迟协议下的在线扩展定义与验证，保留同一初始权重的静态对照。GPT4MTS-online须准确称为新声明的扩展，不能冒称原方法已自带在线协议。原生VoT缺口不会用来源不明的prior填补。

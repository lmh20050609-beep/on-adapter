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
| 共同协议在线扩展 | 五域×五seed×三方法=75/75；same-start静态对照保留；online_evidence | 固定策略扩展，不是原生GPT4MTS在线复现；无独立确认 |
| VoT | 官方源码与10份prior已审计，投稿包代理身份已区分 | 生成材料、原点与可用时间来源缺失，未用于排名 |
| 方法、图和回复 | reports/method_appendix.md、机制图；各refine-logs阶段的中英文回复草稿 | 综合回复和替换文本已整理于review_delivery；无整篇LaTeX源，未整合或提交完整论文 |

阅读顺序：

1. [静态五域五seed结果](../refine-logs/external_baselines/STATIC_MATRIX_FINDINGS.md)与[回复更新](../refine-logs/external_baselines/STATIC_MATRIX_REVIEW_RESPONSE.md)。
2. [组件五seed决策](../refine-logs/review_sprint/REVIEW_UPDATE_FIVE_SEED.md)。
3. [压缩路线](../refine-logs/review_compression/REVIEW_FINDINGS.md)。
4. [基线来源与缺口](../refine-logs/external_baselines/INTEGRATION_REPORT.md)。
5. [原190矩阵交付说明](HANDOFF.md)，用于原在线结果与方法段落。

以上各表的任务、数据划分和是否在线更新不同，不能合成一个总体胜率。当前不支持普遍SOTA、全部组件必要、语义专家分工、真实publication-time无泄漏或显著优越性。end_date始终是代理；旧Table1–3来源问题仍未恢复。

共同延迟协议在线扩展已完成75/75；与静态起始权重一致，25组原CSV和成熟更新核验通过。iAdapter在线相对数值在线平均MSE仅2/5域略低，平均MAE五域均更高。固定GPT4MTS在线扩展25/25单元两指标劣于自身静态，必须保留强静态对照。该扩展不冒称作者原生在线算法。

当前综合入口：[本轮修订结论](../refine-logs/review_delivery/REVIEW_DECISIONS.md)、[逐点回复](../refine-logs/review_delivery/REVIEW_RESPONSE.md)、[正文替换](../refine-logs/review_delivery/MANUSCRIPT_REPLACEMENTS.md)、[结论核对](../refine-logs/review_delivery/CLAIMS_FROM_RESULTS.md)。原生VoT、真实publication time和原Table1–3来源缺口仍未解决。


## 2026-10-01 updates

Same-target feedback factorial, metadata/content readout and five-dimensional simple-fusion exploration are now complete. New restricted positive mean comparisons are retained alongside stronger references and failures. Full evidence and limits: ../refine-logs/EXPERIMENT_RESULTS.md. Native VoT, original table reconstruction, true publication-time and complete LaTeX remain unresolved.

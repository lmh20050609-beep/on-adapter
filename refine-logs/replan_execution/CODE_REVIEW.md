# Experiment Bridge 最终修复复核：M0–B2

日期：2026-09-30。审查性质：fresh-agent、same-family / provisional。
仓库：`/home/wuwa/projects/research/iadapter`。

## 部署判定

**PASS：当前没有 BLOCKING 问题，可以在计划规定的源提交与干净工作区检查后，按 M0 → Energy/2023 首3单元 sanity → 完成B1 → B2 的既定次序部署。**

这项判定只表示代码审查通过；不代表首3单元GPU sanity或新实验已经完成，也不启动B3–B5。完整初审范围、核心算法结论与原始证据见同目录 `reviewer_response_initial.md`。本次只复核修复涉及的M0统计与B2失败处理，没有扩展实验或重复全量检查。

## BLOCKING

**无。** 两项初审阻断均已修复并核验：

1. **U/F入口参数修复已通过。** `feedback_factorial.full_start` 只将 dataset/seed 交给旧 loader，保留 full 原 warm checkpoint。真实checkpoint入口回归和六组起始参数核验已在初审修复阶段通过；本次没有更改该路径。
2. **M0实际token/字符审计已补齐。** `replan_audit.reconstruct(details=True)` 保存字符裁剪偏移和裁剪前长度；默认返回仍为 `(text, spans)`。逐行输出 fact/preds/label-or-separator/special 四类token计数，四类总和与实际编码token数相同；缺文本行实际编码数为0，空串单独tokenize的2个special token另存于tokenizer_token_count。汇总分开报告1200字符截断与64-token截断，补可见字符、裁剪前后长度、文本年龄摘要；fact/preds比例分母明确为可见内容token总数，排除标签/分隔与CLS/SEP，不把日期锚点偷偷删除。

## NON-BLOCKING

**无未解决项。** 最后发现的空文本special-token计数口径也已修复：token_count、pre_truncation_tokens、visible_token_counts.special在缺文本行全部为0；tokenizer_token_count单列保留空串tokenize的2。对1903行实际数据再次核验，360个缺文本行均符合该规则，各split的special_tokens恰等于2×informative行数，原字符截断及fact/preds内容token计数完全不变。证据见 `reviewer_empty_text_check.json`。

## B2失败记录修复

`text_signal_probe.run_group` 现在先拒绝已有输出目录，再调用 `_run_group`；实际异常发生且当前输出已创建时，保存 `status=failed` 与 traceback，并继续抛出异常。控制测试在模型加载前故意抛错，确认保存准确错误原因；随后第二次调用得到 `FileExistsError`，既有status文件字节保持不变。没有运行骨干前向、PCA、Ridge或GPU实验。

## 必要CPU复核结果

新脚本 `reviewer_final_checks.py` 的所有断言通过，详见 `reviewer_final_checks.json`：

- 实际Energy/SocialGood前80%共 **1903行** 的重建默认接口与details接口一致；`裁剪偏移 + 裁剪后长度 = 裁剪前长度`。
- 输出文本、visible IDs、metadata的原语义保持不变；四类token计数总和正确。
- 新M0汇总与初审独立本地tokenizer计数完全一致：

| 数据域 | 切分 | 字符截断行 | token截断行 | fact内容token | preds内容token |
|---|---|---:|---:|---:|---:|
| Energy | train | 38 | 764 | 28615 | 20656 |
| Energy | selection | 2 | 148 | 3729 | 4854 |
| Energy | development | 16 | 148 | 4045 | 4571 |
| SocialGood | train | 179 | 180 | 3321 | 7580 |
| SocialGood | selection | 90 | 90 | 1971 | 3462 |
| SocialGood | development | 90 | 90 | 1910 | 3527 |

- B2失败状态、错误原因、拒绝覆盖测试通过。
- 初审修复后的 **10项CPU单元检查** 已通过，证据为 `reviewer_tests.txt`；此次未改动其覆盖的U/F数学、默认运行器、原CSV评分或PCA/Ridge逻辑，故不重复无关检查。

## 保留的科学边界

初审已确认30个旧控制的原CSV目标/mask/指标及现有manifest指纹、六组真实起始参数，且四种U/F策略的未来标签/embedding扰动不改变先前预测与状态。默认OnlineRunner与旧Stream精确一致；B2只在训练前缀拟合、仅在选择段选lambda、P3保持P2状态及当前metadata，最终评分针对原CSV目标。

本轮仍是70–80%已暴露开发段上的探索性诊断；end_date仅为可用性代理，U/F只隔离部署阶段，B3–B5尚未触发。没有新的显著性、独立泛化、真实publication-time或语义因果结论。

所有新复核输出仅写到当前trace：`reviewer_m0_recheck/` 及最终空文本修复复核的 `reviewer_m0_empty_recheck/` 为CPU文本审计副本；`reviewer_simulated_failure/` 为故意触发的错误处理测试，均不属于正式实验结果。未更改源代码、数据、cache、checkpoint或正式results目录。

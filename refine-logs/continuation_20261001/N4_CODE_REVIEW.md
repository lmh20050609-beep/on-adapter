# N4 原生 VoT 接入代码审查：最终结果

**Verdict: PASS（静态代码与真实数据映射审查）。** 首轮唯一阻断 B1 已修复。本结论为 **fresh-agent、same-family provisional**；**runtime 尚未检查**，不能表述为 native VoT smoke 已通过。两轮均未导入 Torch，未调用 WSL/GPU，未加载模型或执行更新。

## B1 修复核验

当前 `source_patch/iadapter/native_vot_smoke.py:35–44` 验证作者 CSV 是原数据的相同日期/OT 前缀，并验证被省略尾部没有有效 OT；两个 origin 的完整目标必须同时落在原训练段和共同前缀内。它没有删除原数据尾行，也没有更换标准化器。

本轮直接抽取并执行修复后的八条实际 AST 语句，输入为原 ZIP 的 `iAdapter/data/SocialGood.csv` 和固定作者提交的 SocialGood CSV，全部通过：

- 原数据 900 行、作者 892 行，差异仅是 2024-05 至 12 月的 8 个缺失 OT 尾行；原数据保留这些行。
- 原训练段仍为 540 行，mean/scale 与首轮实际证据完全一致。
- local/author origin 均为 7、8，对应发布 loader 的 window start ID 0、1；真实目标 shape 为 `[2,6,1]`，12/12 目标有效，并与原独立参考定义一致。
- `validation` 已记录 original_rows、author_rows、original_train_end 和明确的前缀/缺失尾行处理规则；`prior_provenance_verified=False` 保留。

`native_vot_smoke.py` 的实际 diff 仅涉及上述映射检查和输出元数据。`native_vot.py` 以及 `data.py/audit.py/config.py/util.py` 与首轮审查快照一致。修改后的完整脚本编译检查通过。锁定文件和接入报告已经披露 893 条 prior 与作者 L8 窗口数 885 的差异。

## 仍然成立的正确性结论

首轮核对的作者 PatchTST 三分支、六个原 helper 的依赖闭包、固定配置、192-token 统计文本、六层本地 GPT2 冻结路径均未修改。数值/AFF 损失继续使用原 CSV 真实目标和原始有效掩码，prior 只作为 AFF 输入且明确 12→6 截断。

batch1 三分支形状检查、batch2 对齐更新、数值 head/attention/decomp_w 的梯度与更新检查、对齐阶段排除骨干参数及其不变性检查、异常写 failed 与 traceback 的路径均保留。没有新增影响科学含义的改动，也没有新增 hash、兼容层或 flags。

## 结论边界

发布 prior 的生成原点、文本可用时间和检索来源仍未验证；893 这一数量关系不提供来源证明。此项是锁定 smoke 的明确限制，不妨碍使用已声明来源限制的两个 prior 做输入接入检查，但不允许据此发布正式基线分数或完整复现结论。

真实 Torch/transformers 导入、`reformer_pytorch/einops` 环境依赖、GPT2 权重加载、CUDA 内存、实际形状/有限值、梯度与更新均待原环境执行。代码含这些断言不代表运行已经通过。无需再扩展本轮静态审查；下一项必要证据是原环境的真实 smoke 结果。

## 保存的证据

- 首轮完整发现与源码快照：`review_trace_vot/N4_CODE_REVIEW_20261001_round1.md`、`001-response.md`、`primary_static_evidence.json`。
- 本轮准确 diff 和源码快照：`review_trace_vot/round2/change.diff`、`round2/reviewed_native_vot_smoke.py`。
- 本轮实际数据核验：`review_trace_vot/round2/primary_recheck_evidence.json`、`primary_round2_trace.txt`。
- 最小范围重审脚本：`review_trace_vot/recheck_round2.py`。未重跑首轮完整脚本，未覆盖首轮证据。

本轮只写审查产物，没有修改实现、原始数据或运行状态。

# E003 fresh-agent code review

日期：2026-10-01。结论：**PASS，可提交并运行预定 Energy/2023 两表示、四评价单元 sanity。**

评审性质：**same-family / provisional**，独立二级代码审查；不是外部模型独立确认。未启动 GPU 训练，未修改实验源码。本结论仅批准实现进入 sanity，不预测科学结果。

## BLOCKING

无。未发现真实的维度错误、旧 loader 参数不兼容、训练/选择泄漏、donor 状态污染、目标错配或队列覆盖问题。

## NONBLOCKING

最终版本无未解决问题。初审发现强参考报表遗漏：固定计划要求裸骨干与 static GPT，初版 audit 仅导出新序列及 numeric 配对差值。执行者已在 `iadapter/revision_candidate.py:102-104,113` 补齐 `references.csv`、`reference_summary.csv` 与结果正文；只复用 B1 已有成绩，不新增训练。用真实 B1 表核验得到 30 行、10 组、每组 3 seed，包含 frozen_backbone、numeric_static、numeric_online、gpt4mts_static、iadapter_online。问题关闭。

## 关键审查证据

- **选择依据与探索范围正确。** 已读取固定计划、探索修订和原共同协议，以及 B3 的 runs/gates/summary。Energy LLM5 对 metadata 平均 MSE −1.114865%，3/3 seed，donor − correct 为 +0.000563840；其对 raw 为 +1.436067%，SocialGood 对 metadata 为 +5.469566%，均须保留。规则 Energy 对 metadata −2.578801%，但 donor 更好，不能按本修订称作内容线索。小于 2% 不是本次用户已授权探索的阻断条件；原门槛仍未通过。
- **配置、旧 loader 与切分正确。** `revision_candidate.py:19-21,50-60` 只枚举两域、三 seed、两表示；`feedback_factorial.py:29-39` 明确按 dataset/seed 建 config 并核对 seed-specific OT/H6、L48/36、d32、2 层、batch16 骨干，额外 representation 字段不会传入不接受它的旧接口。实际 CPU 加载六组均通过：Energy 887/1035/1183、143 个开发原点；SocialGood 540/630/720、85 个开发原点。
- **五维与初始化正确。** `revision_candidate.py:23-36` 仅替换首层为 5→96，沿用原 96→64、ReLU/dropout0.1/LayerNorm；模块名种子重建首层。同配置两表示的全部参数逐 tensor 相等。`models.py:90-107` 与 `legacy_core.py:121-158` 保留简单 text_add 的数值反馈、关闭文本反馈，不接专家或 entropy；缺文本最终表示为零。
- **标准化和预算正确。** `task_text_prior.py:122-157` 的 B3 变换来自训练前缀；PCA 实现 `text_signal_probe.py:27-39` 只对 informative training rows 拟合 basis，均值/scale 只使用训练行。E003 读取既有变换，无重新拟合。实际核验六组 PCA center、投影 train mean/scale，以及 fit_origins 的最后目标早于 train_end。LLM 沿用同一 M0 visible_text 缓存，`prior_arrays` 检查 dataset 与字符串；没有读入额外文本、改 prompt 或重复模型推理。M0 selected_ids 与实际 data.text_ids 全部吻合。
- **同 warm 分叉与在线回放正确。** `revision_candidate.py:56-78` 每组训练流 warm 一次，再恢复同一 warm 进行 static/online；新 backend 和 OnlineRunner 创建零反馈、空队列、全新 Adam。`shared_online.py:78-117` 先预测再消费已成熟原点，H6、batch1、末尾不 flush、原上下文回放。前 7 个开发预测与 static 对照，冻结骨干逐 state tensor 比较，static 参数不变而 online 必须有位移。
- **donor 不污染主序列。** `revision_candidate.py:38-48` donor 只取 informative 训练原点，沿用当前 feature/hg/ge/offset 与当前可用性；eval/no_grad 的 AddFusionAdapter 无状态变动或 dropout 随机消耗。独立运行 CPU unittest，原预测、全部更新事件与最终参数对有/无 donor 探针完全一致；另核对 donor 在训练段且早于当前原点。
- **真实评分与持久队列正确。** `revision_candidate.py:73-76,83-113` 逐单元重新对齐原 CSV OT 目标/有效 mask，并重新计算 MSE/MAE；审计消费序列、原文 ID、同初始化以及 24 新评价+12 donor 序列。`revision_candidate.py:115-126` 首两组恰为 Energy/2023 raw、LLM；12 组串行队列跳过完成项，失败/运行中项不被自动覆盖，输出目录拒绝覆盖。结果展示所有三 seed，配对 SD 采用样本标准差。

## 验证

1. 独立执行 `/home/wuwa/miniconda3/envs/llm/bin/python -m unittest discover -s tests -p test_revision_candidate.py -v`：2/2 通过。
2. `reviewer_cpu_checks.json`：两域×三 seed 实际数据/骨干加载、变换来源、原点/原目标、缺文本置零、同初始参数、两表示首原点前向全部通过。全部前向在 CPU，未进行新的 warm-up 或模型选择。
3. 强参考报表修复：真实 B1 数据选择 30 行，10 组各 3 seed，通过。

后续只需按既定顺序提交源码后运行四格 sanity；sanity 完成后依据实际目标/mask、更新和输出再放行剩余组。本次审查不把局部探索包装成稳定、显著或独立确认。

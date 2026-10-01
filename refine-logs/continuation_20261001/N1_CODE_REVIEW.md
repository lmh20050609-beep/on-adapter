# N1 同字符近期优先代码审查

结论：**PASS（代码准入；same-family / provisional）**。BLOCKING：无。可在恢复 WSL、应用并提交本次补丁后，进入真实 `prepare` 和 Energy/2023 单组 sanity；本审查不代表模型运行已就绪，也不替代 sanity。

审查日期：2026-10-01。审查对象为 `source_patch/iadapter/text_order.py`、`source_patch/iadapter/recent_text.py` 及其测试；依赖代码来自 Windows 交付快照 `output/iadapter-replan-executed-20261001`，其 `DELIVERY_MANIFEST.json` 声明来源提交 `c62791c9e2700de322ac2a3739ac6a2fd9a31630`。本次仅用 Windows Python、AST 提取的纯函数和原始 CSV/归档产物；未调用 WSL、Torch、BERT 或 GPU，未修改实验源码。

## NONBLOCKING

- `N1_IMPLEMENTATION_STATUS.md:7` 的命令参数 `--limit1` 应为 `--limit 1`。`recent_text.py:117` 只定义 `--limit`，照抄现有文档会被 argparse 拒绝。这是执行说明的小错误，不影响模块逻辑。

## 已核对的正确实现

1. **字符预算与来源块保持。** `text_order.py:5–18` 仅接收原 `reconstruct` 已完成 1200 字符尾裁剪后的文本及 span。两条保留记录完整交换，只有一条或空输入保持；不重新抽取长正文。对 1,903 行真实输入逐行核验字符 Counter、UTF-8 字节 Counter、每个 body span 的内容、来源 ID 不晚于原点、组顺序倒置，以及组内 fact/preds 顺序。全部通过，含 415 行原有字符裁剪、5,309 个 body span。Energy 1,183 行中 1,148 行交换；SocialGood 720 行中 340 行交换。重建文本、选中 ID、裁剪长度及旧可见前缀均与归档 M0 一致。
2. **编码及缺失文本协议相同。** `recent_text.py:28,35–53,75–76` 核对旧选择文本、复用 frozen 12 层 BERT 和 `prepare.contextual_pool` 的 mask mean pooling、64 token cap，仍输出 768 维；空文本保持零向量。未加入原选择之外的输入。真正的 token 可见性、新编码数值和 batch/singleton 一致性尚未运行，不能从纯文本检查宣称编码成功或近期文本一定改善。
3. **旧 B1 配置与预测可复用。** 从 `raw_records_20261001.tar.gz` 读取全部 12 个真实 text_add static/online 配置，它们与现行 `Config`/`pilot_config` 完全一致，源代码均与 B1 队列记录的 `28b3944cdaa5dfaa76f14d8bbc70e4161b025921` 一致。两域均为 OT、H6、L48/36、d32、2 层、batch16，分界和训练期归一化统计与原 CSV 一致。`recent_text.py:72` 的严格 config/data 校验没有发现 legacy 字段不兼容；`feedback_factorial.load_bone:29–39` 继续校验并加载对应 dataset/seed 骨干。
4. **初始化与 warm 路径对齐。** `recent_text.py:72,76–83` 复用 name-seeded `initialize_adapter`，比较 current fresh initialization 与 recent initialization，沿用 seed+11000 的 train Stream；近期 static/online 都从同一 warm state 开始。`variant=text_add` 保留数值反馈、关闭文本反馈，未更改网络宽度。current 诊断从旧 static 保存的 warm adaptive state 开始，seed+20000 与 B1 一致，并在 `:73` 对整条旧 online 预测作原有容差校验。
5. **donor probe 不改原预测路径。** 复用 `revision_candidate.DonorBackend:38–48`：先保存原预测/context/gate，再在 eval/no_grad 下替换训练期文本 embedding；该路径没有 optimizer、反馈更新或训练态 dropout。训练 donor 小于 train_end，且逐原点检查 donor<t。当前预测/context 随后仍送入原 OnlineRunner。实际数值上的“不扰动”仍须由上述整条 current replay 比较通过来确认。
6. **目标、mask 和成熟消费正确。** `recent_text.py:55–60,79–83,93–97` 对原 CSV 独立构造 OT 目标与 valid mask，原点为 70%–80% 开发段、H6。独立检查全部 12 条旧保存序列：Energy 每条 143 原点、SocialGood 每条 85 原点，target/mask 精确一致，MSE/MAE 重算与保存值相同；六条旧 online 的事件顺序、origin+6 成熟时刻及原始 text_ids 也通过。新执行复用已经审查的 OnlineRunner，先预测后消费成熟记录，所以前 7 次 static/online 预测应一致；代码显式检查此项，并检查冻结骨干参数及 static/online 位移。
7. **计数与结果范围没有扩张。** `recent_text.py:89–115` 将 12 个新近期评估、12 个旧 current 复用、6 次 current 诊断重放、6 次 warm 和 12 条 donor 序列分开报告；旧数值/裸骨干/static GPT/LLM5 参考保留。总计新增 18 次评估执行及 6 次 warm；donor 是同状态附加前向序列。分析明确标记为 post-results exploratory study，未用 2% 门槛否定已授权的小幅或波动进展。

## 验证证据与边界

- 新增三项标准库 unittest：3/3 通过；两个新模块语法编译通过。
- 独立实数据与归档检查：`review_trace/reviewer_checks.py`；完整数值结果：`review_trace/reviewer_checks.json`。脚本首次读取错误的 M0 快照子路径而失败，改为读取归档中的实际 `m0/*_token_provenance.json` 后通过；该修正仅涉及审查脚本。
- `recent_text.py:107` 锁定 campaign 队列的代码 revision 和计划；prepare 也记录 revision。现有程序未在 campaign 中比较 cache 的 revision，因此本结论适用于已声明的“同一已提交源码 prepare → sanity → 余下队列”流程；若 prepare 后为修复而改源码，应重新生成对应缓存，不把旧缓存继续当作新代码产物。当前未发现已生成缓存或实际混用。
- 仍待真实 WSL 验证：依赖导入、BERT 实际编码、current donor 重放与旧预测一致、近期 static/online 前 7 次一致、真实模型冻结和 online 参数位移。Windows 检查没有冒充这些证据。

准入后的首个命令序列为 `python -m iadapter.recent_text --stage prepare`，随后 `python -m iadapter.recent_text --stage campaign --limit 1`；该单组真实 sanity 通过后，再继续其余五组。

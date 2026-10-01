# 窄范围修改复核

日期：2026-10-01。fresh-agent 同一审查代理 follow-up；same-family provisional。范围仅为上一轮 N1/N2 的阶段拆分、源码锁和队列生命周期，以及 N3 的成本统计处理说明。

**结论：修改正确，未发现实际阻断错误或 assessment 状态/配置回归，可以继续已计划的最小 sanity。** 没有启动训练、神经推理、GPU 实验或其他新增实验，没有修改源码。

读取并比较了审查副本 `source_patch/calibration_pilot.py` 与 WSL canonical `/home/wuwa/projects/research/iadapter/iadapter/calibration_pilot.py`：两者逐字符相同，Python AST 语法解析通过。

## 具体核验

- **先选择后评估已落实。** 非 assessment 的 `ledger_group` 仅遍历 calibration、selection；新的 CLI assessment 入口通过 `ledgers(..., assessment=True)` 才生成 assessment 预测和评分。该入口要求已有同源码的 locked.json、首轮 queue 同源码且全部完成，因此正常调用链不会在锁定前生成评估账本。full 分支仍只在 assessment 运行。
- **两种完成状态不互相覆盖。** 首轮组状态保留在 `groups/<job>/status.json`，其 completed 现在准确表示 calibration/selection 完成；assessment 单独写入 `groups/<job>/assessment/status.json`。创建 assessment 父目录后，在其下创建各分支目录是合法的，不会与 `exist_ok=False` 冲突。assessment 失败只将自己的阶段状态和 assessment queue 标为 failed，不会重写首轮成功状态。
- **assessment 配置复核可稳定通过。** 两轮生成的 metadata 都由同一组 job、相同完整数据、旧 source checks、相同 Config 和 clean revision 组成，没有计时或新阶段路径等必然变化字段。先前保存 config 经过 JSON 往返后仍是相同的列表/字典/数值类型；不存在因为 assessment 本身就导致配置比较失败的问题。新阶段仍恢复各分支 warm state，backend/优化器/成熟队列重新创建，分支状态不会由首轮带入。
- **两个队列各自推进。** 首轮使用 queue_state.json；assessment 使用 assessment_queue_state.json。每个队列均检查 code/planned，在写 running 后运行单组，成功改 completed、失败改 failed；完成组跳过，已有 failed/running 组拒绝自动重跑。--limit 限制的是本次新完成组数，续跑不会重新创建已完成组目录。assessment 缺少第一阶段依赖时，在新队列写入前就拒绝执行。
- **来源锁已补齐。** select 检查首轮 queue 版本与当前 clean revision 相同；evaluate_mix 检查 locked 版本与当前 clean revision 相同，并检查 assessment queue 同版本且全部完成。assessment 又检查首轮 queue 和 lock 与自身版本相同。因此沿合法调用链，首轮、规则锁、assessment、混合评估使用同一源码版本。locked.json 是 select 完成后唯一原子写入的成功产物，当前没有另外产生“未锁定但含相同 code”的路径。
- **成本意见的处理方式合理。** 独立正式微基准使用不带 ContentBackend 的分支；描述性账本计时剔除已有 probe_seconds，可以避免把两个探针算入正式预测路径。该微基准不在此次更新文件中，也未在本次审查执行，实际数值由执行方另外取得；无需为此重跑本次代码审查。

## 唯一非阻断残留

`select` 第 266 行仍先创建 selection 目录，第 267–268 行才验证来源和首轮完成状态。如果误在第一轮完成前调用 select，或以错误源码版本调用，检查会正确拒绝，但留下空 selection 目录；修正调用条件后重试会先遇到 `FileExistsError`。这不影响按当前正确执行顺序运行，也不是评估数据泄漏。

窄修正是把 `out.mkdir(exist_ok=False)` 移到这两个前置检查之后，保持真正开始选参后仍不覆盖已有输出。不需要新恢复机制，也不需要为这一行的移动另开完整审查。

**复核范围内无 BLOCKING 项；此前关于数值实现与证据边界的结论保持不变。**


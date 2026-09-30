# B3 探索准入增量代码复核

结论：**PASS**。BLOCKING：无。NONBLOCKING：无新增代码缺陷。允许在提交本次源码并使受跟踪工作区干净后，串行运行正式 `generate_exploratory`，完成后运行 `readouts`。本复核未启动 GPU、未加载模型、未改动实验产物或源码。

审查性质：`aris:experiment-bridge` secondary code review，same-family / provisional。范围为 `iadapter/task_text_prior.py` 和 `tests/test_task_text_prior.py` 的当前增量，以及 `EXPLORATION_AMENDMENT.md`、现有 preflight / quality / budget / cache 产物。基线提交为 `7f5081d11a495a441f37ce076af9feb4983adaef`；审查时增量尚未提交，因此运行代码提交号应由随后生成的准入记录写入，不在本报告中虚构。

## 审核依据与具体位置

1. **严格入口保持拒绝。** `iadapter/task_text_prior.py:85` 至 `:96` 的 `generation_admission` 首先检查格式和干净源码。默认分支仍要求 `quality.status == passed` 且 preflight / generation 代码版本相同。实际 `quality_review.json` 为 `stopped_groundedness`；用该真实文件在模拟干净的新提交下调用默认入口，得到 `AuditError: Groundedness/domain check did not pass`。新探索入口没有把原质量结论改写成通过。
2. **探索准入明确且复用约束成立。** `iadapter/task_text_prior.py:92` 至 `:95` 对固定 prompt、两个 domain context、五类别枚举、greedy、max_new_tokens=8、repetition_penalty=1、finite logits、frozen parameters 和模型来源逐项检查。实际 preflight 与当前常量及 `weights/qwen2.5-0.5b-instruct/SOURCE.json` 一致。`load_model`、`classify_batch`、`records` 没有出现在本次变更中；当前 `:37`、`:47`、`:54` 对应的实现仍只使用 M0 可见文本、同一本地 Qwen 0.5B、同一 wrapper 和推理配置。本次无需重跑 preflight 或新增来源哈希方案。
3. **原停止结论与新旧代码会分别留存。** `iadapter/task_text_prior.py:102` 写入独立 `generation_admission.json`，明确 `original_quality_status`、`original_preflight_code`、`generation_code` 及 imperfect-prior prediction-utility 用途；`:111` 的完成记录也携带准入身份与原质量状态。`generate_exploratory` 在 `:178` 显式转入探索路径，CLI 在 `:194` 单列该 stage。
4. **缓存和计算上限未放宽。** `iadapter/task_text_prior.py:101` 至 `:110` 保留最多 1543 个唯一输入、跳过已有 key 和累计 3600 秒限制。对真实 M0 和 cache 的只读核对得到 1538 个唯一输入、30 个训练缓存、1508 个待生成输入；30 个 cache key 全部属于当前池，domain、可见字符串、key、split 和 valid 字段一致。`budget.json` 仍是 7.646915407000051 / 3600 秒。未重置预算，也未重复调用模型。
5. **真实目标评分与选择逻辑未改变。** 本次 diff 未触及 `readout_group` / `analyze` 的拟合及评分主体。`iadapter/task_text_prior.py:134` 至 `:147` 使用真实数据目标、训练拟合和 selection 选 lambda；`:155` 至 `:165` 从原 CSV 重建目标并核对保存预测指标；donor 仍来自训练历史。LLM 标签没有被当作真值。`:174` 的原 2% 门槛保留，探索修订文档另行允许局部、小幅、带波动的方向性证据。
6. **异常类型修正正确。** `tests/test_task_text_prior.py:19` 的新增测试使用真实 `AuditError`，与 `iadapter/audit.py:18` 的 `require` 一致；验证原质量失败在严格入口被拒绝、探索入口接纳同一不完美先验，以及已测试的 prompt / token cap / finite logits / model source / format 变更被拒绝。

## 实际验证

在 `CUDA_VISIBLE_DEVICES=''` 下运行：

```text
/home/wuwa/miniconda3/envs/llm/bin/python -m unittest discover -s tests -p test_task_text_prior.py -v
Ran 4 tests in 0.002s — OK
```

另以真实 preflight / quality / cache 文件做 CPU 只读检查，仅 mock `revision` 来验证待提交代码的准入分支：

```text
real_proof_contract True True True True
unique_inputs 1538 cached 30 pending 1508 cache_subset True
strict_rejection Groundedness/domain check did not pass
exploratory_acceptance user_revised_exploratory_imperfect_prior
dirty_rejection Commit source before generation
```

## 解释边界

原 30 条训练预检的 26 UNKNOWN、4 DECREASE、0 格式无效和两条明确目标错配仍是事实；本次 PASS 只认可用户授权的探索性预测价值检验可以继续，不认可 LLM 语义已正确。后续任何小幅改善都可以作为进一步研究线索，但需按修订报告对 raw / metadata 的均值、逐 seed 方向、配对波动、MAE 与 donor；不得把它自动称为显著或稳定优势。现阶段无需新增提示词、模型、文本或重审机制。

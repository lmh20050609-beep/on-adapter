# 修订证据包交付说明

本轮已完成 190/190 次预定适配评估、35 个共享骨干配置以及六组效率测量。
实验源代码提交为 `f68027fc8dc6a091cfc2d80d6896f67660f54651`；后续提交只增加
分析、表图、写作、检查和交付材料，不改变这些已运行的训练代码。

## 阅读顺序

1. `reports/analysis-report.md`：实际结果与结论边界。
2. `reports/main_table.md`、`summary.csv`：均值和跨 seed SD。
3. `reports/component_verdicts.csv`：组件消融配对差值。
4. `reports/efficiency.md`：包含完整 BERT 的参数与成本口径。
5. `reports/manuscript_replacements.md`、`reviewer_responses.md`：可替换段落与逐点回复草稿。
6. `reports/validation.json`、`stats-appendix.md`：全量结果检查与统计单位。

`runs/` 保存所有逐原点预测、有效目标掩码、文本 ID、更新事件、完整配置和
适配器权重；`checkpoints/` 保存共享骨干；`cache/` 保存冻结 BERT 表示和编码记录。
未删除任何旧 WSL 源码、旧 CSV、原论文或历史图。

## 复核与重生成

工作目录 `/home/wuwa/projects/research/iadapter`，解释器
`/home/wuwa/miniconda3/envs/llm/bin/python`。

```bash
python -m unittest discover -s tests -v
python -m iadapter.campaign --status
python -m iadapter.campaign --audit-completed
python -m iadapter.analyze
python -m iadapter.write_evidence
```

这里的 python 指上述环境。效率脚本 `python -m iadapter.efficiency` 会重新测量
并覆盖 reports 下的效率报告，应只在确实需要重测时执行。它不修改正式实验。

campaign 是按实验提交固定的已完成队列。默认命令现会对已完成队列执行只读审计后退出，
不因后续分析提交改变而报错，也不重新训练或修改完成时间。--status 只读完成标记；
--audit-completed 对照原始 CSV 验证全部单元。未完成队列仍禁止混合代码版本。
analyze 在通过原始 CSV 审计前不生成新表图。详见 original_csv_audit.json。
当前共 32 项协议与审计检查通过，包括篡改目标、掩码、原点、反馈和状态分组的拒绝测试，
配对 seed/目标数核验、历史路径内容校验及本地 BERT 无压缩包复用；最新输出见 reports/cleanup-tests.txt。
旧项目目录和旧名称链接已按用户要求删除，运行数据与权重均使用当前项目内部路径。
历史运行记录保留原始来源字符串，通过 docs/PATH_MIGRATION.json 校验解析；删除后的
190 次结果审计仍通过，见 reports/cleanup-after-delete-audit.txt。
完整备份与恢复方式见 docs/BACKUP.md；逐项审稿覆盖见 docs/REVIEW_COVERAGE.md。
要单独复核一个配置，使用 `python -m iadapter.run` 并给一个新的 `--run_dir`；
不要把复核结果混入原 190 次结果。预处理和模型配置见 docs/PROTOCOL.md 与
docs/DECISIONS.md。

## 补齐的可直接使用材料

- reports/method_appendix.md：五专家、门控、描述符、正则与原点级伪代码；明确潜在维度操作的语义边界。
- reports/main_pairwise_analysis.md：三种主方法的全部配对比较，包含文本加法对数值加法。
- reports/state_paired_analysis.md：历史状态变化组内的配对差值和标准差，阈值通过原始 CSV 审计。
- reports/failure_cases.md：每域最不利 seed 的明确事后案例，保留在主表中，不推断路由因果。
- 机制图蓝色刻度为代理新文本行，紫色刻度为队列目标成熟；二者均与预测原点对齐。

这些材料由已有预测生成，没有增加训练或选择新的最佳模型。简单文本加法在 H6 的 0/5 个域降低平均 MSE、1/5 个域降低平均 MAE，不能假定文本本身稳定有益。

## 论文结论必须保留的限制

- H6 三方法主比较中，完整模型仅 2/5 个域平均 MSE 最低、1/5 个域平均 MAE 最低。
- 路由文本梯度删除后，Energy、Security、SocialGood 三域平均 MSE 都更低。
- 25 次 H6 完整模型运行中 22 次出现单专家平均权重超过 99%；不能据此宣称动态专家分工。
- 可训练适配器加文本投影约为 minimal 数值骨干参数的六倍；小绝对规模不等于相对骨干参数高效。
- 原 Table 2、真实 publication time、原十域多骨干结果仍未恢复验证；不主张 SOTA 或显著普遍改善。
- 当前没有论文 LaTeX 正文源码，交付的是修订文本、表图片段与证据包，不是编译后的完整投稿稿件。

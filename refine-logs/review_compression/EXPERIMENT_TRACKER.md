# E-01 跟踪与完成状态

2026-09-29。

| 项目 | 状态 | 证据 |
|---|---|---|
| 独立代码审查 | 通过；same-family provisional | CODE_REVIEW.md |
| 协议与压缩检查 | 43/43通过 | tests.txt |
| Economy 两次sanity | 2/2完成，原CSV核验通过 | experiments/review_compression_20260929_sanity/reports/validation.json |
| 正式窄投影运行 | 12/12完成 | accuracy_evidence/validation.json；原queue_state.json |
| 原宽投影控制 | 12/12复用并核验 | accuracy_evidence/runs.csv 的 reused/source/commit |
| 原骨干 | 六个复用；未新增训练 | 保存的 backbone_checkpoint 和原始元数据 |
| 精度/参数门槛 | 无候选同时通过两域 | accuracy_evidence/decisions.csv |
| 隔离成本sanity | 2/2完成，不计正式表 | profile_output/review_compression_20260929_sanity |
| 四设置同轮成本 | 24/24完成，科学原文件不变 | cost_evidence/status.json、REPORT.md |
| 写作 | 中英文修订解释已形成 | REVIEW_FINDINGS.md、REVIEW_RESPONSE_DRAFT.md |
| 五seed扩展/新宽度搜索 | 不触发 | 预定精度/参数门槛未满足 |

记录路径相对 /home/wuwa/projects/research/iadapter；未在当前导出中的原始预测、权重和计时仍保留在WSL。此前五seed组件实验和原190次结果不覆盖。

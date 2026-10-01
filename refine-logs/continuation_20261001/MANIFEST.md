# 当前交付清单

| 文件/目录 | 实际内容与边界 |
|---|---|
| source_patch/iadapter/{text_order,recent_text,extension}.py | 可应用的新源代码；尚未部署，尚无新GPU结果 |
| source_patch/iadapter/{native_vot,native_vot_smoke}.py | 作者PatchTST定义/六个数学helper接入；原CSV真实目标三阶段smoke，待GPU执行 |
| source_patch/tests/test_text_order.py | 标准库三项保留字符/正文测试，通过 |
| N1_CODE_REVIEW.md、review_trace/ | 独立N1源码及真实文本审查，PASS same-family provisional |
| N3_CODE_REVIEW.md、review_trace_n3/ | 扩域代码、原ZIP、因果变换及最终配对统计审查，PASS same-family provisional |
| N4_CODE_REVIEW.md、review_trace_vot/ | 两轮源码审查；作者892行/原900行尾缺失差异修复；最终PASS same-family provisional，无Torch/GPU |
| N1_IMPLEMENTATION_STATUS.md、N3_CANDIDATE_LOCK.md | 已锁定设置和实际实施状态 |
| N4_IMPLEMENTATION_LOCK.md | 三阶段接入检查的源实现、实际超参数、真实数据、prior范围及界限 |
| EXTENSION_SCHEMA_AUDIT.{md,json} | 追加域官方数值身份核验、重复日期处理 |
| VOT_SOURCE_AND_INTEGRATION.md、VOT_DISPATCH_AUDIT.json | 发布路径、实际入口、先验与版本差异；无基线分数 |
| MANUSCRIPT_INTEGRATED.md | 基于既有真实数据的摘要/方法/实验/讨论整合；N1/N3明确TODO |
| REVIEW_RESPONSE_INTEGRATED.md | 原review九点覆盖的回复草稿，未提交 |
| INTEGRATION_E003_TABLE.csv、INTEGRATION_E003_PAIRED_TABLE.csv | 从已验证原报告生成的展示表，不是新评价 |
| INTEGRATION_VALIDATION.json、build_manuscript.py | 展示来源与可再生成脚本 |
| CONTINUATION_TRACKER.md、DEPLOYMENT_MANIFEST.md | 剩余执行顺序、精确复制项和sanity要求 |
| RUNTIME_STATUS.json | WSL实际故障与未执行/未停止进程的记录 |

本目录不取代唯一WSL仓库，不删除原数据，本提交同步GitHub，WSL工作区部署与GPU执行仍待完成。下一个必需步骤是恢复WSL，在原环境完成sanity和真实队列。原Table2、publication-time及nativeVoT可信来源缺口保留；已有局部改善和原始不利结果同时保留。

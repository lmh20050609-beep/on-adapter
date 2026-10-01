# Agriculture / Health 扩域数据核验

核验日期：2026-10-01。数据来自用户提供的原投稿 Code And Data Supplement.zip，官方来源为 [Time-MMD](https://github.com/AdityaLab/Time-MMD/tree/00281e2d86058286d5548b15a7670e8eda57ef62)。本文件是数据核验，不是扩域预测结果；当前 WSL 尚不可执行，因此部署前仍要对照实际工作目录中的 CSV。

| 域 | 投稿行数 | 官方行数 | 正确观测键下的匹配 | OT 最大绝对差 | 实际目标 |
|---|---:|---:|---:|---:|---|
| Agriculture | 496 | 532 | 496；官方另有 36 条早期记录 | 0 | 美国 retail broiler composite，月度 |
| Health | 1389 | 1389 | 1389，一对一 | 0 | 美国 National unweighted ILI 就诊百分比 |

Agriculture 投稿数据是官方数据的日期子集，范围为 1983-01 至 2024-04。官方多出的 36 条位于 1980–1982 年，不能为了扩大样本悄悄加入本次实验。目标身份与 [Time-MMD 原论文](https://arxiv.org/html/2406.08627v1) 的数据说明一致。

Health 的区域字段为 `REGION TYPE=National, REGION=X`。`OT` 与 `100 × ILITOTAL / TOTAL PATIENTS` 的最大差约为 4.995e-6，符合原 CSV 的舍入精度。它表示门诊就诊中流感样疾病的比例，不能改称确诊流感率或人口感染率。

Health 有 5 组同一日期区间、不同 YEAR/WEEK 的报告：1997W53/1998W1、2003W53/2004W1、2008W53/2009W1、2014W53/2015W1、2020W53/2021W1。原投稿和官方数据都包含这些观测。只按 start_date/end_date 合并会产生多对多连接，造成 1399 行及假的 2.13282 最大差；该连接已判定无效。以 YEAR、WEEK、REGION TYPE、REGION 为键后，1389 条观测一一匹配且 OT 完全相同。

本轮保留不同报告，不去重、不重写日期，也不替它们推断新的文本发布时间。H=6 表示后续 6 条报告，在这些日期重复处不声称恰好 6 个自然周。排序保留同日期报告的原始顺序。文本可用时间继续使用 end_date 代理，真实 publication time 尚未验证。

N3 固定 60/20/20、OT、L=36、H=6。Agriculture 的训练/验证边界为 297/396，测试 95 个原点、89 次可成熟更新；Health 为 833/1111，测试 273 个原点、267 次可成熟更新。最后 6 个原点不在评分结束后补更新。两个域的原始 OT 都无缺失；接口仍保留原始目标掩码。

机器记录：`EXTENSION_SCHEMA_AUDIT.json`。Health 的顶层匹配数和差值已使用正确一对一观测键；无效 date-only join 留在 `invalid_date_only_join_diagnostic` 中，不进入身份结论。原文件在 `tmp/timemmd_source_audit_20261001/`，未改动原 ZIP。

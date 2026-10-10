# 九域最小文本修复（V1）数据与来源报告

2026-10-08。目标：在保持数值数据、划分、目标、mask 与反馈规则不变的前提下，把已确认的文本时间错配最小修复扩展到全部 9 个基准域，并冻结完整实验协议。

**V1 只屏蔽已确认晚于描述时点的独立条目**；来源分级为 C/D 的条目一律保留并披露，筛查未命中不等于已认证。本报告不读取任何模型成绩。

## 1. 模型实际读取的文本字段

| 方法 | 读 `fact` | 输入路径 |
|---|---|---|
| On-Adapter | 是 | `online.ResidualOnline` 经 `data.prepared_text.pooled(origin)`，可视行 (origin-1, origin) 拼接，64 token mean pooling |
| GPT4MTS | 是 | `multimodal.GPTForecast` 经 `text.gpt(origin, lookback)`，512 token cls |
| MM-TSFlib | 是 | `multimodal.MMD` 经 `text.tokens(origin)`，GPT2 6 层，256 token |
| iTransformer / ADAPT-Z / DSOF / Proceed | 否 | 代码路径不访问 `data.texts` / `prepared_text` |

`preds`、`Final_Output`、`Final_Search_*` 任何方法都不读。

## 2. 来源分级与屏蔽

规则文件 `source_evidence/decision_rules_all9.json`，SHA256 `85152a0e584b9bb5717895596c26b903f497e3a1285b0c02c78f51021f948689`；继承 4 域冻结规则 56 条（SHA256 `3d0109e3dd46e5f2a312dc59e8823d9f6d556437e7b06e6246531e5a11813eda`），新增 5 域独立复核 292 条。

| 域 | 独立条目 | A | B（屏蔽） | C（保留披露） | D（未核验保留） | 屏蔽条目占该域 | 被改动行数 |
|---|---|---|---|---|---|---|---|
| Environment | 15014 | 0 | 238 | 126 | 14650 | 1.585% | 238 |
| Energy | 1478 | 1 | 0 | 0 | 1477 | 0.000% | 0 |
| Health_US | 1387 | 3 | 10 | 7 | 1367 | 0.721% | 10 |
| Traffic | 2287 | 0 | 40 | 44 | 2203 | 1.749% | 38 |
| SocialGood | 4513 | 0 | 103 | 20 | 4390 | 2.282% | 92 |
| Security | 1288 | 35 | 4 | 0 | 1249 | 0.311% | 4 |
| Economy | 1840 | 3 | 7 | 0 | 1830 | 0.380% | 7 |
| Climate | 1866 | 0 | 9 | 5 | 1852 | 0.482% | 9 |
| Agriculture | 1271 | 2 | 2 | 2 | 1265 | 0.157% | 2 |
| **合计** | 30944 | 44 | 413 | 204 | 30283 | 1.335% | 400 |

屏蔽的是条目出现次数；同一原始文本在多行重复出现时按行分别屏蔽。未删除任何数值行、预测原点或评分目标。

## 3. 各域各划分覆盖率

| 域 | 划分 | 条目 | A | B | C | D | 屏蔽 | 条目保留率 |
|---|---|---|---|---|---|---|---|---|
| Environment | train | 8914 | 0 | 126 | 63 | 8725 | 126 | 0.9859 |
| Environment | val | 3050 | 0 | 91 | 28 | 2931 | 91 | 0.9702 |
| Environment | test | 3050 | 0 | 21 | 35 | 2994 | 21 | 0.9931 |
| Energy | train | 886 | 0 | 0 | 0 | 886 | 0 | 1.0000 |
| Energy | val | 296 | 0 | 0 | 0 | 296 | 0 | 1.0000 |
| Energy | test | 296 | 1 | 0 | 0 | 295 | 0 | 1.0000 |
| Health_US | train | 831 | 2 | 8 | 4 | 817 | 8 | 0.9904 |
| Health_US | val | 278 | 0 | 1 | 0 | 277 | 1 | 0.9964 |
| Health_US | test | 278 | 1 | 1 | 3 | 273 | 1 | 0.9964 |
| Traffic | train | 1360 | 0 | 23 | 17 | 1320 | 23 | 0.9831 |
| Traffic | val | 461 | 0 | 10 | 13 | 438 | 10 | 0.9783 |
| Traffic | test | 466 | 0 | 7 | 14 | 445 | 7 | 0.9850 |
| SocialGood | train | 1459 | 0 | 44 | 8 | 1407 | 44 | 0.9698 |
| SocialGood | val | 1556 | 0 | 38 | 5 | 1513 | 38 | 0.9756 |
| SocialGood | test | 1498 | 0 | 21 | 7 | 1470 | 21 | 0.9860 |
| Security | train | 774 | 22 | 3 | 0 | 749 | 3 | 0.9961 |
| Security | val | 256 | 10 | 0 | 0 | 246 | 0 | 1.0000 |
| Security | test | 258 | 3 | 1 | 0 | 254 | 1 | 0.9961 |
| Economy | train | 1101 | 0 | 0 | 0 | 1101 | 0 | 1.0000 |
| Economy | val | 369 | 0 | 2 | 0 | 367 | 2 | 0.9946 |
| Economy | test | 370 | 3 | 5 | 0 | 362 | 5 | 0.9865 |
| Climate | train | 1001 | 0 | 3 | 2 | 996 | 3 | 0.9970 |
| Climate | val | 430 | 0 | 4 | 2 | 424 | 4 | 0.9907 |
| Climate | test | 435 | 0 | 2 | 1 | 432 | 2 | 0.9954 |
| Agriculture | train | 437 | 0 | 0 | 1 | 436 | 0 | 1.0000 |
| Agriculture | val | 401 | 1 | 1 | 0 | 399 | 1 | 0.9975 |
| Agriculture | test | 433 | 1 | 1 | 1 | 430 | 1 | 0.9977 |

## 4. 实际 token 变化

按每种方法冻结的 tokenizer、token 上限与文本映射规则，比较 V0 与 V1 下每行实际模型输入。

| 域 | 方法 | 含屏蔽条目的行 | 其中屏蔽条目仍落在 token 窗口内 | 行 token 变化 | test 原点 | test 有 token 变化的原点 |
|---|---|---|---|---|---|---|
| Environment | On-Adapter | 238 | 0 | 261 | 3003 | 23 |
| Environment | GPT4MTS | 238 | 0 | 231 | 3003 | 21 |
| Environment | MM-TSFlib | 238 | 224 | 262 | 3003 | 23 |
| Energy | On-Adapter | 0 | 0 | 0 | 285 | 0 |
| Energy | GPT4MTS | 0 | 0 | 0 | 285 | 0 |
| Energy | MM-TSFlib | 0 | 0 | 0 | 285 | 0 |
| Health_US | On-Adapter | 10 | 0 | 17 | 267 | 2 |
| Health_US | GPT4MTS | 10 | 0 | 10 | 267 | 1 |
| Health_US | MM-TSFlib | 10 | 10 | 20 | 267 | 2 |
| Traffic | On-Adapter | 38 | 0 | 27 | 102 | 4 |
| Traffic | GPT4MTS | 38 | 0 | 38 | 102 | 8 |
| Traffic | MM-TSFlib | 38 | 32 | 67 | 102 | 15 |
| SocialGood | On-Adapter | 92 | 0 | 51 | 172 | 10 |
| SocialGood | GPT4MTS | 92 | 0 | 53 | 172 | 10 |
| SocialGood | MM-TSFlib | 92 | 27 | 99 | 172 | 19 |
| Security | On-Adapter | 4 | 0 | 3 | 55 | 1 |
| Security | GPT4MTS | 4 | 0 | 4 | 55 | 1 |
| Security | MM-TSFlib | 4 | 3 | 7 | 55 | 2 |
| Economy | On-Adapter | 7 | 0 | 1 | 80 | 1 |
| Economy | GPT4MTS | 7 | 0 | 7 | 80 | 6 |
| Economy | MM-TSFlib | 7 | 5 | 13 | 80 | 12 |
| Climate | On-Adapter | 9 | 0 | 4 | 95 | 1 |
| Climate | GPT4MTS | 9 | 0 | 9 | 95 | 2 |
| Climate | MM-TSFlib | 9 | 6 | 16 | 95 | 4 |
| Agriculture | On-Adapter | 2 | 0 | 2 | 95 | 1 |
| Agriculture | GPT4MTS | 2 | 0 | 2 | 95 | 1 |
| Agriculture | MM-TSFlib | 2 | 2 | 4 | 95 | 2 |

On-Adapter 的 64-token 窗口在多数域并未包含被屏蔽的条目本身，token 变化主要来自 `clean_fact` 尾部 1200 字符窗口因删除而前移；MM-TSFlib 的 256-token 窗口则会保留其中相当一部分。

## 5. 身份与复现

| 域 | V1 文件 SHA256 | 非 fact 列全等 |
|---|---|---|
| Environment | `8e49bd5a186f2a63f9547e7b6342d4f0…` | True |
| Energy | `d32f6dca6818cf8cb415c15af77a9d8e…` | True |
| Health_US | `d2a9e10eb61b3817221f7d8ca176275a…` | True |
| Traffic | `d1255c65f764f3ec07ef92129162347c…` | True |
| SocialGood | `9d31f381f459e0058a6808df1314756a…` | True |
| Security | `11df63ad31ef0a0d13448abbb25958d9…` | True |
| Economy | `1c30e12372d248e875691b45041180fb…` | True |
| Climate | `2b95bbc4cb16c02872422192bc9086a5…` | True |
| Agriculture | `eed9766319b022c240efcc1b028b6993…` | True |

Agriculture/Economy/Security 三域 V1 与三域先导冻结文件逐字节一致：True。

## 6. 必须保留的限制

- 来源复核只覆盖“正文中出现晚于描述时点年份”的候选条目；未命中不代表无问题，也未认证为 A。
- C/D 条目全部保留在实际数据中，仅作披露；V1 不等于“所有条目均已认证”。
- 数值仍使用原 `end_date` 可用性代理与回顾性目标 mask 原点网格；不证明真实发布时间与历史修订版本意义上的完整因果性。
- 来源决定不读取模型成绩，不因结果调整屏蔽规则。
- 本阶段不扩展到其他 seed 或 SocialGood 主表；SocialGood 的 V1 已包含在本 9 域数据中，但不单列额外先导。

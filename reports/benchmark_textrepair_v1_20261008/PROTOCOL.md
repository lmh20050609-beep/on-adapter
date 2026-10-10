# 冻结协议：九域 V1 最小文本修复 benchmark

冻结于 2026-10-08，在任何本轮结果产生之前。冻结内容为数据身份、代码身份、
执行矩阵、复用规则与统计定义。**本次冻结发生在旧测试集已被反复观察之后，
不能追溯称为独立确认或未见结果的预注册。**

## 1. 数据

- 数值数据、划分、目标、mask、反馈规则与原 benchmark 完全一致；
  V1 只改动 `fact` 文本列。
- 文本版本：**仅 V1**（只屏蔽已确认晚于描述时点的独立条目；C/D 保留并披露）。
  V0/V2 已有的三域诊断保留，不作为主表数据版本。
- 域与窗口（不扫描 L36，不删域）：

| 域 | 周期 | L | H |
|---|---|---|---|
| Environment | daily | 96 | 48/96/192/336 |
| Energy | weekly | 24 | 12/24/36/48 |
| Health_US | weekly | 24 | 12/24/36/48 |
| Traffic / SocialGood / Security / Economy / Climate / Agriculture | monthly | 8 | 6/8/10/12 |

- seeds：2025、2026、2027。规模 9×4×3×7 = 756。

## 2. 文本读取路径（只读盘点）

| 方法 | 是否读 `fact` | 输入路径 |
|---|---|---|
| On-Adapter | 是 | `online.py` 中 `self.encoder = data.prepared_text`，`pooled(origin)`；行 i 的映射文本为可视行 (i-1, i) 拼接，64 token mean pooling |
| GPT4MTS | 是 | `multimodal.GPTForecast`，`text.gpt(origin, lookback)` 取 origin 之前的历史，512 token cls |
| MM-TSFlib | 是 | `multimodal.MMD`，`text.tokens(origin)`，GPT2 6 层，256 token |
| iTransformer / ADAPT-Z / DSOF / Proceed | 否 | 代码路径不访问 `data.texts` / `prepared_text` |

`preds` 及 `Final_*` 字段任何方法都不读。

## 3. 编码

沿用已封存的独立编码 v2：每个非空完整文本单独一次前向，实际模型
batch size 恒为 1；权重、层数、token 上限、pooling、FP32 编码 / FP16 缓存
不变。缓存身份包含 `data_sha256` 与 `text_version`，与历史缓存目录隔离。

## 4. 训练与评估

- 逐 run 生效配置以 `resolve_settings` 的 resolved 结果为准，并由每个
  run 的 `config.json` 落盘；不使用 campaign 模板代替。
  规则：rows<500 → d_model=32, dropout=0.2, batch=8, epochs=50；
  On-Adapter/ADAPT-Z → epochs=3；GPT4MTS → epochs=10, batch=4；
  MM-TSFlib → batch=8。patience=3，学习率 0.0003（GPT4MTS 0.001）。
- 训练段拟合预处理（均值/标准差只用训练段），验证段选模（严格验证 MSE），
  测试不调参。
- 在线方法反馈规则：`old_origin + H <= current_origin`（与原数值协议一致）。
- 不降低任何方法的训练预算，不遗漏失败配置。

## 5. 复用规则

- 只复用**文本无关**的 4 个方法（iTransformer、ADAPT-Z、DSOF、Proceed）
  在历史 756 组中的完成单元，共 432 格，逐一由
  `benchmark_textrepair_v1/legacy_numeric_reuse_manifest.json` 固定
  result / config / best.pt / predictions 哈希，并通过
  `reuse_verification.json` 与 `numeric_reuse_equivalence.json` 证明兼容。
- 3 个读文本方法共 324 格全部在本包内从相同定义的初始状态重新训练。
  不沿用任何旧文本版本成绩。

## 6. 评分

- 主空间：全部选定数值变量的已观测目标元素，训练集均值/标准差标准化空间。
- MAE / MSE 按有效目标元素评分。
- CVaR_q：每个原点已观测元素的 MAE 分布，均匀原点分布的最坏 q 概率质量，
  边界原点取分数权重；与变分定义独立核对。
- **主统计**：先在每个域/H 内对 seed 求平均风险，再除以同一域/H/seed 的
  iTransformer 共同骨干 CVaR90，得到比值；对 36 个域/H 任务取等权几何平均。
- 并列报告：平均排名、MAE、MSE、CVaR80/95、逐域/H 结果。
- 预先声明的探索性分组：月频六域几何平均、全部 7 种频率组合、频率等权敏感性。
- 零值导致几何平均不可定义时明确报告，不事后加 epsilon。

## 7. 必须保留的限制

- 三 seed 之外未扩展；SocialGood 未追加配对先导作为主实验前提。
- 来源决定只覆盖筛查命中项；未命中不等于已认证 A；C/D 保留但未核验。
- 数值仍使用原 `end_date` 可用性代理与回顾性目标 mask 原点网格，不证明真实
  发布时间与历史修订版本意义上的完整因果性。
- 本协议冻结晚于旧测试集观察，不构成独立确认。

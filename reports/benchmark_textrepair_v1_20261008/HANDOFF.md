# 交接文档：On-Adapter 九域 V1 文本修复完整评测

2026-10-08。本文档用于把当前工作转交给 Claude Code CLI（或其它会话）继续。

## 0. 一句话状态

**全部完成。** 任务 1（9 域 V1 数据与来源分级）、任务 2（协议冻结）、任务 3（51 项路径检查）
均已完成；756 格完整评测（324 新训练 + 432 复用）**324/324 训练完成、0 失败**；
收尾流程（audit / tables / compare / score / manifest / build_report）已全部执行。
最终报告见同目录 [REPORT.md](REPORT.md)。

### 0.1 最终结论摘要

主统计（域/H 等权跨任务几何平均相对 iTransformer 骨干 CVaR90，越低越好）：

| 方法 | V1 | 历史(V0+旧编码) | 相对差 |
|---|---|---|---|
| iTransformer | 1.0000 | 1.0000 | +0.00% |
| MM-TSFlib | 1.0147 | 1.0155 | -0.08% |
| GPT4MTS | 1.0360 | 1.0293 | +0.65% |
| ADAPT-Z | 1.0495 | 1.0495 | +0.00% |
| DSOF | 1.0510 | 1.0510 | +0.00% |
| On-Adapter | 1.0936 | 1.0930 | +0.06% |
| Proceed | 1.1108 | 1.1108 | +0.00% |

- **最大相对差 0.65%**：文本修复与编码修复都没有实质改变方法之间的比较结构，排序不变。
- **On-Adapter 不是第一**（平均排名 4.17）；在 Energy 明显劣化（H24 比值 2.53），
  在 SocialGood（0.86–0.94）与 Traffic（0.96–0.99）明显占优。
- 探索性月频六域几何平均：On-Adapter **0.9786**（七方法最低），与历史记录的 0.9787 一致。
- 已披露的补救检查发现：Time-MMD 文本的时间错配是**系统性**的（如 2001 年条目写"截至 2024 年 3 月"），
  不是 13 条孤立问题。但**修复后结果几乎不变**，说明这些错配不是旧比较结构的主要来源，
  也不能反过来证明其余文本无问题。

### 0.2 收尾状态

| 收尾步骤 | 结果 |
|---|---|
| `finalize audit` | 324/324 完成，0 失败，执行锁一致 |
| `finalize tables` | 324 个文本格 → 27 张冻结编码表 |
| `finalize compare` | 10,846 个历史工件**全部未改动、无新增** |
| `score` | 756 格全部合成，独立 Float64 复算通过 |
| `finalize manifest` | 3,280 个工件全部哈希校验通过 |
| `build_report` | [REPORT.md](REPORT.md) |

**执行锁曾需重冻结一次**：`data_cache.py` 在首次冻结后被修过一次（补 `text_version` 属性，
修复缓存加载路径的 `AttributeError`）。已核实**全部 324 格的 `source_fingerprint` 完全一致且等于当前代码**，
随后重新冻结使锁与实际运行代码一致；复用清单与矩阵哈希未变。该修复不涉及任何科学参数。

## 1. 环境与路径

| 项 | 值 |
|---|---|
| 项目根 | `C:\Users\18452\Documents\ChatGPT\FedNoise` |
| baseline 目录 | `C:\Users\18452\Documents\ChatGPT\FedNoise\iadapter-baselines` |
| 解释器 | `iadapter-baselines\.venv\Scripts\python.exe`（Windows Python，非 WSL python3） |
| 运行方式 | 在 `iadapter-baselines` 目录下用 `.\venv\Scripts\python.exe -m <模块>` |
| GPU | RTX 4060 Laptop 8GB，torch 2.8.0+cu129，串行执行 |
| 训练输出盘 | `E:\FedNoise\iadapter-baselines\runs\` |
| 缓存盘 | `E:\FedNoise\` |

**注意**：配置文件里的路径必须用 Windows 形式（`E:/...` 或 `C:\...`）。用 WSL 的 `/mnt/c/...` 写进配置会让原生 Windows Python 找不到文件。

### 1.1 WSL 用法（Claude Code CLI 在 WSL 内运行时）

CLI 在 WSL 里，但项目用的是 **Windows 原生 Python**。跨系统调用有一条硬性要求：

**必须先 `cd` 到 Windows 项目目录，再调用 venv 的 python.exe。**

```bash
# ✅ 正确
cd /mnt/c/Users/18452/Documents/ChatGPT/FedNoise/iadapter-baselines
./.venv/Scripts/python.exe -m benchmark_textrepair_v1.runner plan

# ❌ 错误：从 WSL 的 cwd（如 /home/wuwa）用绝对路径调用
/mnt/c/Users/18452/Documents/ChatGPT/FedNoise/iadapter-baselines/.venv/Scripts/python.exe -c "import benchmark_textrepair_v1"
# → ModuleNotFoundError
```

原因（已实测）：Windows Python 的 `sys.path[0]` 取当前目录；当 cwd 是 WSL 目录时，它会看到
`\\wsl.localhost\Ubuntu\home\wuwa`，`sys.path[0]` 为空，所有包导入失败。
`cd` 到 `/mnt/c/...` 后，Windows 侧 cwd 显示为
`C:\Users\18452\Documents\ChatGPT\FedNoise\iadapter-baselines`，导入正常。

其它 WSL 侧细节：

- 不要用 WSL 的 `python3` 跑项目代码（没有 torch）。仅纯文本/JSON 处理可用。
- GPU 查询用仓库自带的 Windows 版：`./nvidia-smi.exe --query-gpu=memory.used,memory.total --format=csv,noheader`
- 后台进程用 `nohup ... > log 2>&1 < /dev/null &`；WSL 存活期间它就一直跑。
  进程查询：`pgrep -f "benchmark_textrepair_v1.runner run"`
- 训练输出在 `E:` 盘，WSL 下是 `/mnt/e/FedNoise/...`
- 长任务用 CLI 的后台/监控能力；前台 `sleep` 长时间会受限

**给 CLI 的首条指令（可直接粘贴）**

> 读 `reports/benchmark_textrepair_v1_20261008/HANDOFF.md`（注意 WSL 用法一节：先 `cd /mnt/c/Users/18452/Documents/ChatGPT/FedNoise/iadapter-baselines` 再调用 `./.venv/Scripts/python.exe`）。
> 先检查 `benchmark_textrepair_v1.runner` 的进度与是否有失败；若控制器已退出但未完成，用 `runner run` 续跑（已完成格会自动跳过）。
> 324 格全部完成后，按 HANDOFF 第 5 节顺序执行 audit / tables / compare / score / manifest / build_report，并把结果写入 REPORT.md。

## 2. 本轮新增的目录（历史文件一律未改动）

| 用途 | 路径 |
|---|---|
| 9 域 V1 数据与来源报告 | [reports/benchmark_text_repair_all9_v1_20261008/](reports/benchmark_text_repair_all9_v1_20261008/) |
| 数据构建代码 | [benchmark_text_repair_all9_v1/](benchmark_text_repair_all9_v1/) |
| 执行版计算包 | [method_comparison_textrepair_v1_20261008/](method_comparison_textrepair_v1_20261008/) |
| 本轮驱动/检查/评分 | [benchmark_textrepair_v1/](benchmark_textrepair_v1/) |
| 本轮报告与冻结件 | [reports/benchmark_textrepair_v1_20261008/](reports/benchmark_textrepair_v1_20261008/) |
| 新训练输出 | `E:\FedNoise\iadapter-baselines\runs\method_comparison_textrepair_v1_20261008\` |
| 新缓存 | `E:\FedNoise\textrepair_v1_20261008_cache\` |

## 3. 已完成内容

### 任务 1：V1 规则扩展到 9 域
- 冻结 **348 条**来源规则：[decision_rules_all9.json](benchmark_text_repair_all9_v1/source_evidence/decision_rules_all9.json)
  SHA256 `85152a0e584b9bb5717895596c26b903f497e3a1285b0c02c78f51021f948689`
  - 继承 4 域冻结规则 56 条（SHA256 `3d0109e3dd46e5f2a312dc59e8823d9f6d556437e7b06e6246531e5a11813eda`，逐字节不变）
  - 新增 5 域（Environment / Health_US / Traffic / SocialGood / Climate）292 条独立复核
- 9 域 V1 数据：[reports/benchmark_text_repair_all9_v1_20261008/data/V1/](reports/benchmark_text_repair_all9_v1_20261008/data/V1/)（**只有 `fact` 列变化**）
- 屏蔽 413 个条目出现次数（占 30,944 条目的 1.34%）；保留了全部数值行、预测原点、mask、评分目标
- Agriculture/Economy/Security 三域 V1 与三域先导冻结文件**逐字节一致**
- 报告：[reports/benchmark_text_repair_all9_v1_20261008/REPORT.md](reports/benchmark_text_repair_all9_v1_20261008/REPORT.md)

### 任务 2：协议冻结
- [PROTOCOL.md](reports/benchmark_textrepair_v1_20261008/PROTOCOL.md)
- [frozen_matrix_textrepair_v1.json](reports/benchmark_textrepair_v1_20261008/frozen_matrix_textrepair_v1.json) SHA256 `5ab90e412009efb6d8e3e6e7e08f97772f5fe6348d0a57aa3f38e4eaebab0e4c`（756 格）
- [execution_lock.json](reports/benchmark_textrepair_v1_20261008/execution_lock.json) SHA256 `cf7b935f3c9e56706f2fd0a9c3cb51b1d36f6c2c95cf526ecb794dfbb39a9703`（32 项）
- 配置：[configs/method_comparison_textrepair_v1_20261008.json](configs/method_comparison_textrepair_v1_20261008.json) SHA256 `013b27a845ec582b51cc046f30f5d5bb678c03fe200ecd60699af75c6fc8cf87`
- 复用清单：[legacy_numeric_reuse_manifest.json](benchmark_textrepair_v1/legacy_numeric_reuse_manifest.json) SHA256 `c3256eb039a50e0be120b79474df53dcb1ea1b22966a387b16a763b9c4c64fc2`

### 任务 3：未覆盖路径检查
- [path_checks.json](reports/benchmark_textrepair_v1_20261008/path_checks.json)：**51 项全部通过、0 失败**
- 覆盖日频（Environment L96）、周频（Energy/Health_US L24）、非 H12 窗口、此前未训练的 GPT4MTS 与 MM-TSFlib
- 项目：原点/目标/mask/标准化对齐 13/13；预测不消费未来 y/mask 13/13；文本 gather 行可见性 13/13；
  未来文本扰动不影响过去输出 3/3（容差 0）；预处理/缓存版本隔离 9/9

### 复用兼容性证明
- [reuse_verification.json](reports/benchmark_textrepair_v1_20261008/reuse_verification.json)：432/432 格哈希核验，9 域仅 `fact` 列不同
- [numeric_reuse_equivalence.json](reports/benchmark_textrepair_v1_20261008/numeric_reuse_equivalence.json)：5/5 抽样在 V1 数据上重训后 predictions/targets/mask/origins **逐位相等（容差 0）**

### 保护快照
- [protected_before.json](reports/benchmark_textrepair_v1_20261008/protected_before.json)：10,846 个历史工件哈希（启动前）

## 4. 运行状态与断点续跑

**实验已结束（324/324，0 失败），当前无进程在跑。** 如需重跑或续跑：

- 进度：见 [controller_state.json](reports/benchmark_textrepair_v1_20261008/controller_state.json)；逐格日志 [campaign.log](reports/benchmark_textrepair_v1_20261008/campaign.log)
- 顺序：按 域 → H → seed → 方法 依次执行 324 格，每格独立 OS 进程
- 实测总耗时约 7–8 小时（Environment 最慢，36 格约占一半时间）

**断点续跑**（进程若中断，重复执行即可；已完成格按工件哈希校验后自动跳过）：

```powershell
cd C:\Users\18452\Documents\ChatGPT\FedNoise\iadapter-baselines
.venv\Scripts\python.exe -m benchmark_textrepair_v1.runner plan     # 查看剩余
.venv\Scripts\python.exe -m benchmark_textrepair_v1.runner run      # 继续/启动
```

## 5. 跑完后的收尾序列

```powershell
cd C:\Users\18452\Documents\ChatGPT\FedNoise\iadapter-baselines

# 1) 完成审计（逐格核对 result.json 与工件哈希、执行锁未变）
.venv\Scripts\python.exe -m benchmark_textrepair_v1.finalize audit

# 2) 记录实际编码输入表身份
.venv\Scripts\python.exe -m benchmark_textrepair_v1.finalize tables

# 3) 保护比较（历史工件是否被改动）
.venv\Scripts\python.exe -m benchmark_textrepair_v1.finalize compare

# 4) 主统计：756 格合成 + 独立 Float64 复算（含变分 CVaR 交叉核对）
.venv\Scripts\python.exe -m benchmark_textrepair_v1.score
.venv\Scripts\python.exe -m benchmark_textrepair_v1.score --historical   # 历史参照（可选）

# 5) 工件清单与校验
.venv\Scripts\python.exe -m benchmark_textrepair_v1.finalize manifest

# 6) 生成最终报告
.venv\Scripts\python.exe -m benchmark_textrepair_v1.build_report
```

（可选）V1 数据/规则重建（幂等，不覆盖历史）：

```powershell
.venv\Scripts\python.exe -m benchmark_text_repair_all9_v1.inventory
.venv\Scripts\python.exe -m benchmark_text_repair_all9_v1.decisions
.venv\Scripts\python.exe -m benchmark_text_repair_all9_v1.build_data
.venv\Scripts\python.exe -m benchmark_text_repair_all9_v1.token_effects
.venv\Scripts\python.exe -m benchmark_text_repair_all9_v1.build_report
```

## 6. 主统计定义（冻结）

- 评分空间：全部选定数值变量的**已观测**目标元素，训练集均值/标准差标准化空间
- MAE / MSE 按有效目标元素评分
- `CVaR_q`：每个原点的已观测元素 MAE 分布，均匀原点分布的最坏 q 概率质量，边界原点取分数权重；
  与变分定义独立交叉核对（`benchmark_textrepair_v1/score.py` 的 `cell_metrics`）
- **主统计**：先在每个域/H 内对 seed 平均风险，再除以同一域/H/seed 的 iTransformer 共同骨干 CVaR90，
  对 36 个域/H 任务取**等权几何平均**；并列报告平均/中位排名、MAE、MSE、CVaR80/95、逐域/H 结果
- 预先声明的探索性分组：月频六域、daily/weekly/monthly 频率组合、频率等权敏感性

## 7. 关键科学发现（供撰写时使用）

1. **Time-MMD 文本的时间错配是系统性的，不是 13 条孤立问题**。例如 SocialGood 2001 年的条目写下
   "截至 2024 年 3 月的失业率"；Traffic 1983 年的条目写下"2001 年的最大交通量"。
   这正是 V1 存在的意义。
2. **V1 干预对 On-Adapter 实际输入是稀疏的**：On-Adapter 的 64-token 窗口在多数域未包含被屏蔽条目本身，
   行 token 变化主要来自 `clean_fact` 尾部 1200 字符窗口因删除而前移；MM-TSFlib 的 256-token 窗口会保留其中一部分。
   详见 [repair_token_effects.csv](reports/benchmark_text_repair_all9_v1_20261008/repair_token_effects.csv)。
3. **历史 756 组中 On-Adapter 在 CVaR90 上不是第一**（历史参照见
   [historical_reference.json](reports/benchmark_textrepair_v1_20261008/historical_reference.json)）；
   该参照使用旧批量编码 + 原始 V0 文本，**不能填入新主表**，只作方向性对照。

## 8. 已踩过的坑（避免重复）

| 现象 | 根因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'utils.rev_in'` / `'models.MLP'` | 同一进程内先后 build 不同作者仓库模型，顶层模块名 `utils`/`models`/`layers` 互相遮蔽 | 真实 campaign **每格独立 OS 进程**；检查脚本用 `checks.purge_author_modules()` |
| `FileNotFoundError: ...manifest.json.tmp...` | 缓存目录路径超过 Windows 260 字符上限 | 检查用短缓存根（如 `E:/FedNoise/tpc_tra`） |
| On-Adapter 扰动检查"失败" | **检查脚本缺陷**：重建模型后未 `.eval()`，dropout 处于训练态 | 补 `.eval()` 后容差 0 通过；**未放宽任何容差** |
| 控制文件报 "paused" | 配置里 control_file 写成了 WSL 路径 | 统一用 Windows 路径 |
| `Health.csv` vs `Health_US.csv` | 包内 `DOMAIN_FILES` 用 `Health.csv`；V1 数据文件名必须一致 | V1 目录内文件名 = `Agriculture.csv … Health.csv … Traffic.csv` |
| `benchmark_textrepair_v1` 导入失败 | 包名是 `method_comparison_textrepair_v1_20261008`；或 cwd 不是项目目录（`sys.path[0]` 为空） | 注意完整包名；**先 `cd` 到 `/mnt/c/.../iadapter-baselines` 再调用 venv python** |

**规则**：不要修改 `method_comparison_textrepair_v1_20261008/` 下的任何 `.py` —— 它们已被
`execution_lock.json` 固定，改动会让已完成格的 provenance 校验失败。修改驱动/评分脚本
（`benchmark_textrepair_v1/*.py`）不受影响。

## 9. 必须保留的限制

- 协议冻结晚于旧测试集被反复观察，**不是独立确认或预注册**
- 来源决定只覆盖筛查命中的候选条目；C/D 条目全部保留并披露；V1 **不等于**"所有条目均已认证无泄露"
- 数值仍使用原 `end_date` 可用性代理与回顾性目标 mask 原点网格，不证明真实发布时间与历史修订版本意义上的因果性
- 432 格为历史同协议复用结果；文本无关性由代码路径 + 数值等价重跑共同证明，但不等同于本轮重新训练
- iTransformer 为共同骨干，比值为相对量；不得宣称显著、普遍最优或 SOTA
- 不得据此声称"所有历史数值与文本均已证明严格无泄露"

## 10. 外部约束（沿用）

不租用付费资源、不发送外部消息、不创建定时任务、不停止无关进程；
不覆盖原 ZIP、历史数据、历史代码包、历史结果与检查点；
不扩展为重新采集全部 benchmark 或重新设计 On 方法本体。

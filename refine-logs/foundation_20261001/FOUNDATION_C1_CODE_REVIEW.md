# C1 冻结数值预测记录审查

结论：**NON-BLOCKING**。未发现阻止按锁定规格生成 C1 预测记录的新实现缺陷。边界计算、作者API、原目标/mask和persistence评分一致。该结论属于 same-family provisional；并非 C1 运行通过、现代模型更强或整个研究目标完成的证明。

审查对象为 WSL Ubuntu `/home/wuwa/projects/research/iadapter/iadapter/foundation_forecasts.py` 及 `refine-logs/foundation_20261001/EXPERIMENT_PLAN.md` 的 Locked C1 规格。审查时HEAD为 `35cd2f5aeee6310cb479bf621670361768ffa6a9`，C1脚本尚未跟踪、计划修改尚未提交；沿用主流程已确定的启动前提交要求，将两者一起纳入启动commit。本审查没有启动模型、训练或修改源文件。

## 具体问题与交付范围

**无新增阻断项。** 唯一尚未由本脚本完成的既定交付是 numeric-static assessment 对照：第77–81行只将 TimesFM3 和 persistence 写入 `scores.csv`，没有读取 A 的 numeric-static 预测。该缺项不阻止本次生成固定ledger；但必须在生成后完成共同assessment origin/target/mask的显式比较，才能交付计划中的现代数值路径对照。不得把脚本的 `validation.status=passed` 理解为该比较已完成，或宣称TimesFM3更强。

## 边界与原始数据证据

只从AST提取 `bounds`、`region_origins`、`phase_origins` 执行纯数组逻辑；读取CSV和既有NPZ，没有导入或运行模型。当前数据的全部2,245个预定origin满足历史长度、六步目标和成熟时刻要求：

| 数据 | 阶段 | origin数 | 首/末origin | 首/末目标索引 | 阶段右边界（不含） |
|---|---|---:|---|---|---:|
| Energy | fit | 834 | 47 / 880 | 48 / 886 | 887 |
| Energy | checkpoint | 143 | 886 / 1028 | 887 / 1034 | 1035 |
| Energy | calibration | 69 | 1034 / 1102 | 1035 / 1108 | 1109 |
| Energy | selection | 69 | 1108 / 1176 | 1109 / 1182 | 1183 |
| Energy | assessment | 291 | 1182 / 1472 | 1183 / 1478 | 1479 |
| SocialGood | fit | 499 | 35 / 533 | 36 / 539 | 540 |
| SocialGood | checkpoint | 85 | 539 / 623 | 540 / 629 | 630 |
| SocialGood | calibration | 40 | 629 / 668 | 630 / 674 | 675 |
| SocialGood | selection | 40 | 674 / 713 | 675 / 719 | 720 |
| SocialGood | assessment | 175 | 719 / 893 | 720 / 899 | 900 |

- `foundation_forecasts.py:16–20,43–48` 使用 `t+1 >= lo`、`t+6 < hi`；阶段交界舍弃跨界的五个origin，不会让六步标签/成熟时刻越过截止点。阶段开头允许上下文来自此前历史，这是预定因果上下文，不是标签泄漏。
- 独立从CSV按end_date稳定排序，重建初始60%统计量、因果前向填充、原OT目标和valid mask。将三个后段的预期数组与 `experiments/online_calibration_20261001_retry1/groups/<domain>_s<seed>/<phase>/numeric_static/predictions.npz` 核对：2个数据集 × 3个阶段 × 3个seed共18份文件，`origins`、`targets`、`valid_mask` 均 `np.array_equal` 为真，且各目录status为completed。
- Energy总1,406、SocialGood总839个预测。用于正式的本轮assessment对照应仅使用其中291与175个origin；不能把fit/checkpoint/calibration/selection指标混入assessment分母。
- `full_ot` 保留原目标和mask，不以填充值或模型输出替代缺失目标。fit段是用于以后残差头的训练特征生成，输入缩放沿用整个初始60%的训练统计量；它不是对每个早期fit origin进行实时扩展窗口标准化的验证。后续阶段统计量已经固定。

## API、冻结与评分

- C1使用和已通过C0相同的本地checkpoint、固定源版本及预测参数：一维历史OT、H6、serial batch1、排序后的官方中位数，未传入任何covariate。读取本地config确认patch32/64、9个分位数0.1至0.9；默认中位数索引4正确。C1每个输出检查 `shape == (6,)` 和有限值。它没有请求分位数，因此此次不再存在“返回分位数却遗漏检查”的C0问题。
- 对C0真实validation文件的只读核验确认：330,710,976参数、最大batch差值 `2.384185791015625e-07`、峰值 `1,332,584,960` 字节、状态passed，记录的commit为35cd2f5。C0仅作API和小规模运行依据；不能代替C1全流有限输出/冻结检查。
- `foundation_forecasts.py:34–37,79–80` 冻结参数、eval、记录并比较参数version，检查requires_grad与grad；作者API内部使用inference_mode。未创建优化器、训练过程或按得分选择checkpoint、模型、上下文。单次固定模型生成一套ledger；`pretrained_optimization_seeds=0` 不把三个A seed伪造成三次独立预训练拟合。
- `foundation_forecasts.py:56–68` 中模型只收到历史上下文，原target/mask在当次预测返回后才加入记录；`mature_at=t+6`，`updated=False`。该顺序是代码发出预测的约束，不宣称本进程从未在数据加载时读取完整CSV。
- persistence复制同一个归一化、历史填充后的最后值到六个步长。两种方法均使用同一原目标、同一mask、同一窗口，以有效元素计算MSE和MAE。评估的是各方法实际发出的点预测；TimesFM3发出中位数，不应额外宣称它是MSE意义下的条件均值。
- 统计比较时，A的三个seed是已有数值模型的不同拟合；TimesFM3只有一个固定结果。可分别与三份A结果配对、报告A的seed变化，但不能把TimesFM3重复三次当作其优化随机性或独立样本。此限制与Locked C1一致。

## 记录、失败与验证界限

输出目录要求不存在；各phase保存origin、预测、原target/mask、persistence、耗时、边界和上下文记录。全局冻结检查完成后才写 `scores.csv` 和总validation。异常将根status写为failed并保存traceback，不会自动续跑混合不同commit。已完成phase会保留自己的completed状态，因此后续汇总必须同时要求根status completed和总validation passed；不能只扫描子目录完成标记。

AST解析、全部origin边界、18份A数组的精确匹配均通过。未运行C1，未验证C1实际时间/峰值/全部输出或执行结果。预训练重叠未排除、此前已检查时期仍是探索性数据、C1没有残差适配等限制均保留。


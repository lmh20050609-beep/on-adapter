# iAdapter 修订实验结果

## 完成情况

190/190 次预定适配评估已完成；35 个骨干配置；所有保存的预测均已重新计算指标，并核对同组原点、原始目标、有效掩码及共享骨干一致。代码提交：`f68027fc8dc6a091cfc2d80d6896f67660f54651`。没有使用旧 finally_mm.csv，也没有通过选择 seed 或追加方法改善结论。

## H=6 主结果

| dataset | method | MSE | MAE | n |
|---|---|---|---|---|
| Economy | Numeric adapter | 0.301191 ± 0.017567 | 0.407477 ± 0.011565 | 5 |
| Economy | Text addition | 0.303267 ± 0.017749 | 0.408810 ± 0.012572 | 5 |
| Economy | iAdapter | 0.302811 ± 0.020286 | 0.408696 ± 0.012974 | 5 |
| Energy | Numeric adapter | 0.094009 ± 0.007192 | 0.208535 ± 0.004241 | 5 |
| Energy | Text addition | 0.097347 ± 0.007069 | 0.212689 ± 0.006381 | 5 |
| Energy | iAdapter | 0.097389 ± 0.046207 | 0.218451 ± 0.052828 | 5 |
| Security | Numeric adapter | 91.515101 ± 2.035356 | 4.893063 ± 0.086249 | 5 |
| Security | Text addition | 92.664365 ± 1.535644 | 4.957522 ± 0.051339 | 5 |
| Security | iAdapter | 91.216356 ± 3.076448 | 4.826188 ± 0.312196 | 5 |
| SocialGood | Numeric adapter | 0.866542 ± 0.030956 | 0.410124 ± 0.020556 | 5 |
| SocialGood | Text addition | 0.870328 ± 0.025706 | 0.410298 ± 0.015874 | 5 |
| SocialGood | iAdapter | 0.871040 ± 0.028510 | 0.416145 ± 0.016762 | 5 |
| Climate | Numeric adapter | 1.129030 ± 0.018063 | 0.867883 ± 0.008703 | 5 |
| Climate | Text addition | 1.130308 ± 0.017720 | 0.867712 ± 0.008753 | 5 |
| Climate | iAdapter | 1.128200 ± 0.017355 | 0.868359 ± 0.007927 | 5 |

完整 iAdapter 在三个受控方法中取得 **2/5 个数据集最低平均 MSE、1/5 个最低平均 MAE**。这是五个固定数据流上五 seed 的描述性结果，不是十数据集 SOTA 证明，也不表示统计显著。

### 配对差值

| dataset | variant | delta_mse_mean | delta_mse_sd | mse_reduction_mean |
|---|---|---|---|---|
| Climate | Numeric adapter | -0.000830 | 0.002267 | 0.07% |
| Climate | Text addition | -0.002108 | 0.002996 | 0.19% |
| Economy | Numeric adapter | 0.001620 | 0.005593 | -0.51% |
| Economy | Text addition | -0.000456 | 0.005897 | 0.18% |
| Energy | Numeric adapter | 0.003380 | 0.045146 | -3.34% |
| Energy | Text addition | 0.000042 | 0.041648 | 1.38% |
| Security | Numeric adapter | -0.298745 | 2.002850 | 0.33% |
| Security | Text addition | -1.448009 | 2.202844 | 1.57% |
| SocialGood | Numeric adapter | 0.004498 | 0.008655 | -0.53% |
| SocialGood | Text addition | 0.000713 | 0.007503 | -0.08% |

delta_mse_mean = iAdapter MSE − 对照 MSE，负值有利于 iAdapter；相对改善先在共享骨干的配对 seed 内计算，再取平均。不同域的原始 MSE 不直接混合平均。逐 seed 百分比的平均与原始 MSE 均值的相对变化不是同一个统计量，可能出现符号不同；主表最优计数始终按原始 MSE/MAE 均值判断。

### 文本本身是否有帮助

简单文本加法相较数值加法，在 H6 的 0/5 个域降低平均 MSE、1/5 个域降低平均 MAE。三种主方法的全部配对差值见 main_pairwise_analysis.md；逐 seed 数据见 main_pairwise.csv。这个结果不能用于预设文本一定有益。

低/中/高历史状态变化组内的配对差值另见 state_paired_analysis.md。每组比较同 seed、同原点、同有效目标集合；分组分数和验证集阈值已对照原始 CSV 核验。分组解释属于描述性分析，不能据此选择正式模型或声称文本/漂移因果机制。

各域最不利 seed 的明确事后案例见 failure_cases.md；它们按完整模型相对数值加法的 MSE 增量选择，全部仍在主结果中。路由集中与误差并列展示，不被解释为已经证实的失败原因。

## 消融结论

移除路由文本梯度，在 Energy、SocialGood、Security 三域中有 **3/3** 的平均 MSE 低于完整模型。完整逐项结果见 component_verdicts.csv 和 ablation_table.csv。不能据此宣称全部组件稳定必要；也不能将任一小差值自动解释成负迁移或随机波动。没有进行显著性检验。

## 效率和机制

效率见 efficiency.md；原始中位数、P95 和内存分类见 efficiency.json。完整 BERT 的批处理编码成本另见 text_encoding.json。

figures/main_paired_effects.pdf 展示全部主实验配对效果。Energy 与 SocialGood 机制图使用预定 seed=2023 的完整测试流，展示多步预测中的第一步及变量平均 gate，未挑选模型表现最好的时间片。状态变化是历史窗口的描述性统计，不是概念漂移真值；专家权重不是因果功能认证。

**路由诊断：25 次 H6 完整模型运行中，22 次存在某一专家的平均权重超过 99%。** 这是明确的强集中现象，不能把这两幅几乎单色的机制图写成“随漂移切换专家”的证据。不同 seed 的主导专家可能不同。阈值 99% 仅用于描述已记录的 gate，不用于挑选模型或删除实验；详见 routing_summary.csv。

## 不能恢复的原稿证据

- 本轮骨干为 minimal ITransformerZ，数值基线为 ADAPT-Z-style 适配，不能冒充原方法完整复现。
- 未取得 GPT4MTS/VoT 匹配协议结果，因此旧 Table 2 优越性声明仍需撤下或待来源恢复后重新比较。
- 文本使用当前数值聚合期结束时的 end_date 代理可用性，未核验真实新闻发布时间。
- 本轮只有五域 H6，以及 Economy/Energy H12；不支持十域、多骨干、多编码器、普遍反馈延迟鲁棒性声明。
- 五 seed 只刻画固定数据流上的训练随机性；不代表对新数据集的泛化置信区间。
- 当前材料没有 LaTeX 正文源码，提供可替换段落和 LaTeX 表图片段，不提供已编译完整修订论文。

## 审稿意见状态

已实质处理：本轮比较协议、指标来源、回放一致性、关键消融、完整文本编码、参数与在线成本。
部分处理：文本与状态变化关系、组件必要性、性能一致性；结论随数据限制。
尚未完成：原多模态方法匹配比较及原表完整来源恢复，不能宣称两份 review 已全部解决。

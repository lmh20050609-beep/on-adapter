# 五域五seed静态比较：review结论

75/75完成，包含72个新增单元和3个已验证pilot复用单元；25组均通过原CSV审计。来源和全部种子在`static_matrix_evidence/runs.csv`，均值/样本SD在`summary.csv`。没有失败单元被静默删除。

## 结果先行

在本轮三方法中，**静态iAdapter平均MSE最低为0/5域，平均MAE最低为1/5域（Security）**。GPT4MTS在Economy、Security、SocialGood取得最低平均MSE；数值适配器在Energy、Climate最低。不能保留“普遍优于数值方法或GPT4MTS”的表述。

下表只列平均MSE便于阅读；完整MAE及五seed波动见证据表。

| 数据集 | 数值静态适配 | 静态iAdapter | GPT4MTS静态 |
|---|---:|---:|---:|
| Economy | 0.087523 | 0.105313 | 0.061619 |
| Energy | 0.013982 | 0.027068 | 0.015348 |
| Security | 2.750218 | 2.718436 | 2.705181 |
| SocialGood | 0.146024 | 0.149492 | 0.097411 |
| Climate | 0.999289 | 1.006596 | 1.046567 |

相对数值适配器，静态iAdapter只有Security的平均MSE和MAE更低（约1.16%和0.23%）；其余四域两项平均误差都更高。Energy平均MSE增加93.60%，不能通过跨域平均掩盖。Climate和SocialGood相对数值控制的两项误差在所有五seed中都退化。相对GPT4MTS，Climate平均MAE虽更低，但只有1/5 seed改善，均值不能替代逐seed方向。

## 对审稿意见的实际帮助

- **收益不一致**：现在有五seed和三个具体对照的完整负结果，能够撤回一致/显著优越性，而不是用少数有利单元回应。
- **外部比较不公平**：新增共同OT静态任务统一数值窗口、原点、目标/掩码、时间划分和选模上限。内部方法固定插入位置并共用每seed单变量骨干；GPT4MTS是作者模型移植，不平均插入位置。SD只计算种子重复，不混入位置变化。
- **不能过度归因**：GPT4MTS使用更长历史文本CLS/512，iAdapter使用当前文本mean/64；训练阶段也不同。因此是明确配置的系统比较，不是专家结构或等信息量文本的单因素比较。
- **在线问题仍需分开**：这里在开发期不更新，内部反馈清零并保持零。不能把静态结果写成在线iAdapter无效，也不能把旧在线表与此表相减后称作在线更新收益。
- **VoT与原Table2**：VoT生成与原点/时间来源仍不可核验。本表是75单元的三方法部分外部比较，不是四方法100单元完成，也没有恢复原投稿表来源。

## 协议与统计边界

五域均为OT单变量、H6；Energy L48，其余L36。前60%训练、60%–70%选择、70%–80%开发评估。旧测试已被查看，开发流也用于此前研究，五seed不构成新数据独立确认。文本end_date仍只是可用性代理。

各域每seed预测原点数为Economy37、Energy143、Security25、SocialGood85、Climate44；有效目标元素为222、858、150、510、264。重叠目标不视作独立科研重复。SD是五seed样本SD；跨架构只按seed标签和相同流对齐，不宣称相同初始参数，不计算显著性。

新增72单元记录的运行时长之和约468.6秒，包括复用pilot的75单元合计约482.8秒。这些时间受共享骨干、缓存命中和不同训练阶段影响，仅用于执行成本记录，不作为公平加速结论。

## 后续优先级

这套静态入口已可运行且可追溯。若要回答同一延迟监督条件下在线方法之间的表现，下一步应预先定义GPT4MTS的在线扩展，用这些已固定的起始checkpoint，在同一原点成熟队列下比较，并明确命名为online extension。应同时保留各自静态结果，隔离“是否更新”这一变化。此项尚未运行，不能提前声称在线优于静态。

不据本表追加学习率、专家或文本截断搜索。已有C内容有效性门槛和E压缩精度门槛未通过的分支继续收束。最终论文是否能维持原创新定位仍取决于证据，不能以实验数量代替性能或机制支持。

## 证据入口

- `static_matrix_evidence/RESULTS.md`：完整MSE/MAE均值与SD。
- `static_matrix_evidence/paired_runs.csv`、`paired_summary.csv`：全部逐seed差值及波动。
- `static_matrix_evidence/iadapter_comparisons.csv`：相对变化与改善seed数。
- `static_matrix_evidence/validation.json`：25组原CSV核验与来源版本。
- `STATIC_MATRIX_PLAN.md`、`STATIC_MATRIX_CODE_REVIEW.md`：启动前设置与独立代码审查。

训练源提交`e32e33c4a6c6fd1a60ac8a43fc808a4be1e1660f`；复用pilot提交`6b93caf96fdb5f680fe026d182529af664cae904`。原始预测/配置/模型位于`/home/wuwa/projects/research/iadapter/experiments/external_static_matrix_20260929`及pilot目录，runs.csv逐项给出source。原190次在线结果另行只读复核通过，文件和时间戳未改变。

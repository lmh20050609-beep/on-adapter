# Review sprint：优先回应组件必要性与文本内容价值

2026-09-29。从 a5bac96 开始，执行已设计五路线中的 B 和 C1；本机串行，无外部 API，不改原 190 次运行。

## 主张和边界

主问题：去掉路由文本梯度与熵正则后，误差和跨 seed 波动是否改善？辅助问题：在相同数值、权重和反馈状态下，正确文本是否优于内容控制？结果不预设为正，不属于新测试集确认，也不代替 GPT4MTS/VoT 原生比较。

## 冻结设计

- 原始 CSV 按 end_date 排序。训练前 60%，骨干选 checkpoint 用 60%–70%，适配器开发流用 70%–80%；80%–100% 不进入本次指标。
- 标准化只由前 60% 拟合，输入历史前填，原始目标缺失掩码保留。文本仍用既有完整冻结 BERT 缓存与 end_date 代理。
- 骨干 D32、2 层，Energy L48，SocialGood L36，H6；最多30epochs，batch32，Adam lr1e-3，patience5。新骨干保存在本次独立目录，不复用原验证区间不同的 checkpoint。
- 适配器在训练流预热一次；开发流前重置 Adam、队列和反馈，保留预热权重。在线 batch1，Adam lr1e-3、weight decay1e-4、clip1，dropout0.1，EMA0.9，entropy0.02，temperature1。
- 两域 Energy/SocialGood；seeds2023/2024/2025；七方法 numeric_add/text_add/full/no_route_text_gradient/no_decisiveness/no_route_no_decisiveness/single_mlp，共42次开发流。
- 组合变体同时关闭 gate 的文本梯度输入/调制和 entropy；专家侧文本梯度保留。single_mlp 复用共享 feature/gradient/text trunk，输出为64→64→32 GELU MLP，移除 gate/多专家/entropy。
- 相同 module 名称和形状采用相同初始值：独立名称派生 seed 重置叶模块，保留零初始化 expert 的零输出层。各方法预热、开发随机流分别重置，避免新增诊断影响优化 RNG。三种骨干配置按每seed共享。
- full/text_add 的开发预测加四臂诊断：正确文本、训练前缀非空文本 embedding 均值、仅从此前可用非空记录抽取的长度匹配 donor（token长度差≤8）、当前文本置零。模型/数值/反馈保持相同，诊断不参与更新。共12条参考流×4臂=48条诊断序列。
- donor 使用独立 RNG；无合法历史 donor 时标记缺失，不用未来补样本。公平误差比较只使用参考有文本且存在匹配 donor 的共同原点，另报覆盖率。零臂同时改变当前可用性，不能单独证明语义作用；历史文本反馈固定，所以它也不是全历史无文本消融。
- 保存原点/文本/反馈来源、目标/mask、原始预测、四臂预测、donor IDs/长度、gate、残差比、更新loss、配置/提交/计时/状态。所有失败保留；不按测试效果重试、调参或删 seed。

## 运行次序和验收

1. 无训练环境 witness、协议测试，独立 reviewer 检查实现。
2. Economy/H6/2023 的 numeric_add 和 full 两个独立 sanity；full 同时检验只读诊断。先看协议/显存/耗时，不用成绩选配置。
3. 固定42单元单进程队列，逐单元保存状态；任何异常停队列。GPU共享，OOM保留失败，核对后可原配置重试，不降低正式模型规模或停止其他进程。
4. 原始 CSV 对照重算全部主指标和诊断；核对同组origin/target/mask/checkpoint、相同控制初始化、诊断不污染运行。
5. 输出开发主表、2×2交互表、内容干预表、状态/成本诊断、逐点review回复。B筛选容差默认相对MSE/MAE 1%，仅作工程候选规则，不是统计非劣效；三seed只作筛选。

预算：42正式开发流+2 sanity；≤7个新骨干（6正式+1 sanity）；48诊断序列嵌在12条正式参考流内。先测sanity实际耗时再估计。后续渠道、额外延迟、压缩和外部基线按本轮证据排序，不自动展开全网格。

第一轮完成并不等于全部 review 已解决。至少应实质给出组件交互、文本利用和所有负结果的可核验回答，保留外部公平比较和真实发布时间的缺口。

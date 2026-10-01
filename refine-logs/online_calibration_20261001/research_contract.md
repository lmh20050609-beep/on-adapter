# Research Contract：异步文本路径的成熟风险校准

## Selected Idea

来源：refine-logs/innovation_20261001/FINAL_PROPOSAL.md。使用已发出的数值与文本预测，在全H目标成熟后估计残差与分支差异的对齐，校准每个h的混合权重。已有原预测等权诊断提供探索线索，但自适应机制尚无结果。

## Core Claims（待检验）

1. 文本路径的互补性随预测长度/反馈历史变化，低维在线组合能在特定设置提供增量价值。
2. 同一账本可分开文本表示、分支参数更新和组合器更新的贡献，复杂路由的必要性需要比较证明。
3. 小幅或异质收益也是可报告进展；不预设全域SOTA、显著性或语义因果。

## Method Summary

冻结已核验数值骨干。numeric与文本适配器独立更新。以原预测n/m计算d=m−n、r=Y−n；成熟且有文本/有效目标的cell提供A+=rd、B+=d²。alpha=clip((A+lambda·alpha0)/(B+lambda),0,1)。校准先验来自未用于模型参数或checkpoint选择的70–75%尾段。全文使用相同原点、原CSV目标掩码和预测先于更新顺序。

## Experiment Design

Energy/SocialGood，H6，三个seed，60/10/5/5/20。固定alpha、等权、端点Hedge、混合候选Hedge、共享/按h LS；LLM5/raw5；静态/在线分支分离；原full在线参考。主指标MSE、次指标MAE，跨seedSD和配对差值。RTX4060 Laptop本机。

## Current Results

尚无本pilot结果。已有N1/N3与等权诊断独立保留，不当作本pilot数字。

## Status

- 方向与两轮方法审阅已完成（同模型族暂定意见）
- 实现、原点协议测试、sanity和完整账本待完成
- 自适应结果与论文整合待完成

# iAdapter 剩余目标推进计划

2026-10-01。用户确认继续完成iAdapter剩余的扩域、基线比较和论文整合。前一轮完成证据保留；当前整体目标仍active，不能把本阶段完成等同全部完成。

## 顺序与范围

1. N1：相同字符集合和BERT64token预算内，改为近期记录先编码。只交换当前已选中/已经裁剪的两条时间记录块，保持块内fact/preds顺序、cleaning、原1200字符裁剪结果；不读新增原文。完整冻结12层BERT编码。两域Energy/SocialGood、OT、H6、seeds2023–2025，旧共同60/10/10开发协议与数值骨干。当前768维text_add静态/在线精确复用12格，新recent768两mode12格、6warm；原数值、裸骨干、staticGPT与LLM5作为系统参照。新recent与旧current参数初始化一致、训练流程一致，唯一设计改动是已选字符块次序。
2. N1 probe：复用旧current warm权重进行6次online同状态donor诊断重放，与已保存原预测一致后才使用；recent另外6序列。重放计为额外诊断执行，不称新增独立seed或主方法结果。保存当前/近期实际可见record ID、token和原点上下文。
3. N2：核验Agriculture/Health原数值列、身份、时序、有效目标、文本来源。Health需明确美国national unweighted ILI而非Africa；Agriculture需确认retail broiler composite。与官方Time-MMD numerical逐日期核对，保留本地合并文件，不覆盖。真实publication-time仍单列未确认。
4. N3：扩域锁定候选为上一轮LLM5简单加法，不按新域成绩再选表示。两域、H6、L36、2023–2027、60/20/20，frozenbone/numeric_online/rawPCA5_online/LLM5_static/LLM5_online，共50评价/10共享骨干/30warm。候选不改LLM/prompt/学习率/宽度，仅新增两个明确目标描述。旧原严格门槛保留，用户要求后的追加域验证单独编号。先完成schema和实现审查、首组sanity，再队列；全失败域也交付。N1研究独立，不把其最好结果偷偷代替已经锁定的扩域候选。
5. N4：继续追溯原生VoT/原表来源和publication-time记录，先核实作者公开源；外部API或来源缺失不造数据、proxy明确命名。GPT4MTS强静态参照已具备，不重新调坏在线policy获得有利排名。
6. N5：把新增证据更新成完整论文论证及回复。缺失原LaTeX不算已恢复；可先完成可整合的章节、图表和来源清单，再推进全稿整合。

## N1执行规定

先N1输入审计/必要tests/fresh代码审查，然后冻结源码提交，编码缓存，Energy/2023两mode sanity，再其余五组串行队列。GPU只用本机，不停止用户进程；BERT预处理batch8，可因OOM降低；正式batch1、网络尺寸、seed/horizon不变。

预测MSE/MAE按原CSV/mask重算；同起点static/online首7预测一致，queue成熟一次、末尾不flush、骨干不变。编码输入每条来自origin≤t的已选record，新的字节集合与当前已裁剪字符串严格一致，token上限64。未来标签/文本/数值变动不影响当前输入选择。N1比较的是相同预算的不同可见内容，不要求BERT向量数值一致。

报告所有seed平均、sample SD、配对差值、MSE/MAE方向，任何局部均值收益都是进展；单seed正向仅记个例。小收益不自动叫稳定显著或因果机制，退化不抹掉。无参数/温度/专家/LR/prompt网格。

## 当前可完成与待补条件

可执行：近期文本对照、公开源schema核验、冻结BERT预处理、追加域训练实现与本机队列、表图/章节整合。需继续检索：VoT原生来源、原表日志、真实发布时间、全稿源。具体缺口不是停止全部工作的理由。

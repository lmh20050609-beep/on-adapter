from pathlib import Path
import datetime,json,shutil
BASE=Path(__file__).resolve().parent
LOG=BASE/'refine-logs'
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
text=(LOG/'round-0-initial-proposal.md').read_text(encoding='utf-8')
def rep(old,new):
    global text
    assert text.count(old)==1,old[:80]
    text=text.replace(old,new)

rep('主要贡献候选是“文本价值随成熟反馈和 horizon 变化的实证与机制”，不是新命名的线性集成。','主要贡献候选是“异步外生文本路径的互补性如何随 horizon 和反馈历史变化”的实证发现，以及低维机制能否替代复杂路由。公式、在线组合和原点成熟协议均已有先例；不将新命名的线性集成作为独创算法。')
rep('3. 首个对照按当前统一协议，在 t+H 以后一次消费该原点全部有效目标，更新每个 h 的 A_h、B_h。两条适配器仍按既有全 horizon 成熟协议更新。','3. 主 pilot 使用既有 benchmark 回放的全窗口成熟假设：origin+H 行可用后才消费该原点全部有效目标。它是记录索引可用性模型，不宣称真实标签 publication-time 已核验。两条适配器继续独立按同一全窗口协议更新，校准系数不缩放它们的梯度。')
rep('4. 单独变体允许校准器在 t+h 的真实目标已可用时消费 (origin,h) 一次，两条适配器仍保持 t+H 更新。这是新的反馈粒度设置，必须单列；若目标并非逐点发布，则不运行这个变体。','4. 逐 h 变体暂不执行：需要 availability_time(origin,h,channel,target_version) 的真实来源，不能用 origin+h 自动构造真实发布时间。取得依据后，逐 h 校准器和全窗口校准器使用各自合法成熟集合，分支更新仍一致；该变体单列，也不宣称部分真值反馈本身创新。')
rep('5. 更新采用标准化目标单位，A/B 对每个 h 分开累计；成熟事件到来时指数遗忘，λ 收缩到训练流估计的 α₀。全缺失目标不更新；B 接近零时使用先验；无文本时强制用数值预测。','5. 更新采用标准化目标单位。对每个 h，a_h=Σ_c valid·(Y-n)·(m-n)，b_h=Σ_c valid·(m-n)²，按有效 cell 的原损失归约，不在另一路改按 origin 平均。每个当前记录索引 t 的预测保存后，统一对所有 A/B 衰减 ρ^(t-last_t) 一次，再按排序稳定的成熟集合累加；多个标签同一时刻到达不产生额外衰减。无文本原点输出 n，其成熟事件不加入文本价值累计量；仍保留统一时间衰减。原点/lead/channel/目标版本各只消费一次，缺失 cell 不加入。λ_h 收缩到校准先验 α₀,h；B 近零或无有效校准样本退回 α₀,h。')
rep('校准初始值来自训练流按时间发出的预测及其成熟标签，不使用离线拟合到同一训练样本后的回放误差冒充在线预热。验证流状态不带入测试。','''正式 A pilot 采用 60/10/5/5/20 的时间划分：0–60% 拟合骨干/适配器/PCA/标准化；60–70% 只选择骨干 checkpoint；70–75% 是标签未参与参数形成的校准片段；75–80% 选择组合器规则；80–100% 评估。Energy/SocialGood 已有骨干的 checkpoint 选择边界为 70%，可复用；Agriculture/Health 的 N3 骨干曾用完整 60–80% 选择，不能用于这套干净校准流程，扩域时需重选 checkpoint。

校准片段上两条分支冻结在训练预热权重，真实发出 n/m；只使用在该片段末前已全窗口成熟、有文本且有效的目标估计 α₀,h 和 mean(d²)_h。α₀,h=clip(A_cal/B_cal,0,1)，无样本回到 0.5。固定归一化/PCA 仍只拟合最初 60%。验证使用这些先验并复制分支权重，允许按照所比较的分支更新规则进行在线回放；其系数、优化器、分支状态和队列全部丢弃。

评估重新载入训练预热权重，A/B 清零，清空反馈/优化器/队列，仅通过 α₀,h、λ_h 注入校准先验；不把校准 A/B 和同批伪样本先验重复计入。校准片段最后尚未成熟的记录不带入评估。原数据多个时段已经被研究者查看，这仍是探索性重划分，不冒充独立未见确认。后续未见时间段确认需另取得新数据。''')
rep('候选超参数至多四组：遗忘系数 ρ∈{1,0.98}，先验伪样本强度 κ∈{1,10}，λ=κ·训练流中 d² 的均值，加固定 epsilon。训练 α₀ 用训练有效样本的受约束最小二乘估计；验证选一组，所有测试域/seed 锁定选择规则。不按测试成绩选择每域的最佳路径。若训练预测对不存在，先补训练流发出预测日志，不能直接以测试记录估计 α₀。','候选超参数至多四组：ρ∈{1,0.98}，κ∈{1,10}，λ_h=κ·校准片段有效 cell 的 mean(d²)_h+1e-8。验证以每个数据域相对同一数值分支的 MSE 比值平均，再在所有 seed 平均，选择唯一一组；平局按较小复杂度/ρ=1 的固定顺序处理。所有评估锁定这一选择，不为每个测试域挑最好配置。没有合法校准发出日志时先补日志，不以测试记录估计先验。')
rep('数值路径、文本路径、训练确定的固定 α、两个分支简单等权、delayed Hedge、所提受约束最小二乘。先离线按时间重放已有发出预测，成本很低；只有有价值信号后才接入实时执行器。','''在同一发出预测账本上比较数值路径、文本路径、校准确定的固定 α_h、等权、delayed endpoint Hedge、预定义 α∈{0,0.25,0.5,0.75,1} 的 mixed-candidate delayed Hedge，以及所提最小二乘。Hedge 使用相同成熟集合/horizon 粒度，正先验下界 1e-6，学习率候选四组 η∈{0.01,0.1,1,10}，预算与 LS 相同，不引用未满足的有界损失 regret 保证。混合候选 Hedge 防止仅靠两个端点对照制造 LS 的优势。

名字拆开：branch-static 表示分支参数冻结；branch-online 表示分支按成熟队列更新；gate-fixed 表示系数固定；gate-adaptive 表示系数变化。gate-fixed+branch-online 不能称 static system。Health 的 LLM5-static 强结果始终保留。共享一个 α 与 H 个 α 作为同日志轻量对照，不另训练神经网络。

已有发出预测可先做纯诊断和相同初值的冷启动回放，标为“后见结果的算法诊断”；它缺合法训练校准先验，不能替代上述正式 pilot，也不进行测试超参数选择。''')
rep('2. **机制区别**：固定 α vs delayed Hedge vs最小二乘；全窗口成熟 vs逐 h 成熟。画 α、成熟标签数量、分支差异与发出预测误差的对齐轨迹。后验 oracle 系数只作为上界诊断，绝不参与当前预测。','2. **机制区别**：固定 α、端点 Hedge、混合候选 Hedge、LS；共享 α 与 per-horizon α。全窗口成熟为当前可执行规则，逐 h 留待真实标签时间证据。画 α、成熟标签数量、分支差异与原预测误差；后验 oracle 仅作可达到空间诊断。若收益只回到 numeric 水平且 α≈0，则结论是抑制弱路径，不是利用了语义。')
rep('3. **文本必要性与简化**：LLM5 vs同维原文本；仅取训练库的历史错配文本；替换为无文本数值路径。若只获集成收益，不宣称语义推理机制。对比完整版和简化版总成本，包含双路径和编码成本。','3. **文本必要性与简化**：LLM5 vs同维原文本；同一文本路径换成预算匹配、仅来自当前前已可用训练库的历史错配/空文本，不随机全测试洗牌。报告 numeric、text 与混合各自误差，避免把参数估计差异当文本因果。成本包含双路径预测/更新、共同冻结前缀、冻结编码、两套 replay 缓存和校准账本。α=0 也通常仍需事先计算 m，不能宣称关闭权重就节省了文本推理。')
rep('| [Adaptive Hedging under Delayed Feedback](https://arxiv.org/abs/1902.10433) | 延迟监督的专家权重聚合 | delay-aware 本身已有理论；必须比较 delayed Hedge，不编造新 regret 保证 |','''| [Adaptive Hedging under Delayed Feedback](https://arxiv.org/abs/1902.10433) | 延迟监督的专家权重聚合 | delay-aware 本身已有理论；必须比较 delayed Hedge，不编造新 regret 保证 |
| [Downside-Controlled Online Forecast Combination, 2026-09-24](https://arxiv.org/abs/2609.29096) | 原点发出预测、成熟损失、冻结/静态/在线路径和 held-out Hedge warm start | 最直接协议近邻；本方案只能以外生文本互补性与具体 horizon 风险利用作为待验证差异 |
| [OneNet, NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/file/dd6a47bc0aad6f34aa5e77706d90cdc4-Paper-Conference.pdf) | 双预测器和在线凸组合 | 双路径在线加权本身不新；CPU 对照不必复制 RL 网络 |''')
rep('论文主线候选：“冻结骨干下，依据延迟到达的真实预测误差估计文本分支的增量价值，实现可解释的在线多模态适配。”','论文主线候选：“在冻结骨干的异步文本预测设置中，检验成熟标签上的数值残差与文本路径差异的对齐如何随 horizon 变化，以及低维在线混合是否足以替代复杂路由。”')
preface='''# 第一轮针对性修订

问题锚点在完整修订方案中逐字保留。

Anchor Check：仍研究冻结骨干、异步文本、延迟监督中的预测改进。没有转向普遍防退化理论或基础模型排行榜。收益小不是退回原因。

Simplicity Check：主线为文本路径互补性的验证与最小利用机制。保留 H 个系数；去掉逐 h 执行承诺，暂不加可靠性网络、MMD、动态专家、LLM agent。B/C/D 是后续独立试验。

接受审阅修改：增加直接最近邻；单列清洁校准/验证/评估边界；固定遗忘/掩码/无文本事件；拆分 gate 与 branch 的 static/online；增加混合候选 Hedge；逐 h 等待真实时间；核算双路径成本。

未采用的扩张：不为获得 9 分加新模块或巨大矩阵，不设置最低收益百分比。

## 完整修订方案

'''
(LOG/'round-1-refinement.md').write_text(preface+text,encoding='utf-8')
(LOG/f'FINAL_PROPOSAL_{stamp}.md').write_text(text,encoding='utf-8');shutil.copy2(LOG/f'FINAL_PROPOSAL_{stamp}.md',LOG/'FINAL_PROPOSAL.md')
state=dict(phase='refine',round=1,agent_id='/root/first_principles_method_review',last_score=7.95,last_verdict='REVISE',status='in_progress',max_rounds=2,timestamp=datetime.datetime.now().astimezone().isoformat())
(LOG/f'REFINE_STATE_{stamp}.json').write_text(json.dumps(state,indent=2),encoding='utf-8');shutil.copy2(LOG/f'REFINE_STATE_{stamp}.json',LOG/'REFINE_STATE.json')
print('REFINED_PROPOSAL',LOG/'round-1-refinement.md')

from pathlib import Path
import json,datetime,shutil,re,csv
BASE=Path(__file__).resolve().parent
LOG=BASE/'refine-logs'
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
text=(LOG/'FINAL_PROPOSAL.md').read_text(encoding='utf-8')
old='正式 A pilot 采用 60/10/5/5/20 的时间划分：'
assert text.count(old)==1
text=text.replace(old,'正式 A pilot 采用 60/10/5/5/20 的时间划分：')
boundary='''

所有阶段按目标终点和成熟时间截断，而非仅筛 origin：用于拟合、checkpoint 选择、校准或规则选择的所有目标必须在该阶段截止边界前已可用；跨界未成熟窗口不消费。历史 lookback 可跨左边界。Energy/SocialGood 现有骨干复用前必须核验选择目标也未越过 70%，不能只看 origin 范围。评估只使用完整 H 步目标落在评估区域的公共原点，末尾不补算。

Hedge 的每个 η 候选都从同一合法校准账本，用冻结 n/m 和同一成熟集合回放取得正初始权重，随后复制到验证与评估；不携带验证权重。endpoint 和五个固定混合候选按每个 h 的有效 cell 的 squared-error sum 更新，与 LS 的统计单位一致。1e-6 floor 在初始化及每次归一化后都施加，随后再次归一化；全缺失和无文本不提供损失证据，已选规则保持一致。

Health 的旧 N3 静态优势保留为历史参考。未来新分割/新 checkpoint 的 static/online 必须重新生成匹配对照，不把旧 N3 数字拼入新 A 主表计算提升。B_cal,h≤1e-8（包括有样本但分支一致）时 α₀,h=0.5。
'''
needle='## 四、备选路线，分别尝试'
assert text.count(needle)==1
text=text.replace(needle,boundary+'\n'+needle)
text=text.replace('α 接近零：它是文本分支无益的实证结果；不通过改测试规则强迫使用文本。','α 接近零：说明当前历史证据下文本路径权重很小，还须结合成熟样本量与内容对照辨别原因；不通过改测试规则强迫使用文本。')
text=text.replace('随后对新固定切分/新时间段确认。','随后对新固定切分继续探索；仅新增且未参与设计的时间段用于独立确认。')
scores='''| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 9.5 | 8.0 | 6.5 | 8.5 | 9.0 | 8.0 | 6.5 | 7.95 | REVISE |
| 2 | 9.5 | 9.0 | 7.0 | 8.8 | 9.0 | 8.5 | 7.0 | 8.42 | REVISE |
'''
def save(name,body):
    p=LOG/name
    if p.exists():shutil.copy2(p,p.with_name(p.stem+'_before_'+stamp+p.suffix))
    q=p.with_name(p.stem+'_'+stamp+p.suffix);q.write_text(body,encoding='utf-8');shutil.copy2(q,p)

anchor=text[text.index('## 问题锚点'):text.index('## 一、第一性原理推导')]
initial=(LOG/'round-0-initial-proposal.md').read_text(encoding='utf-8')
assert anchor==initial[initial.index('## 问题锚点'):initial.index('## 一、第一性原理推导')]
save('FINAL_PROPOSAL.md',text)
(LOG/'round-2-refinement.md').write_text('# 第二轮执行定义补齐\n\n目标/成熟时间边界、Hedge 同账本初始化、旧 Health 参考隔离、零分母与低权重限定已补。两轮完成，不继续追分。以上最后文字补丁未再单独评审；评分仍对应原第二轮。\n\n'+text,encoding='utf-8')
(LOG/'score-history.md').write_text('# Score Evolution\n\nCALIBRATION: none；同模型族暂定意见。\n\n'+scores,encoding='utf-8')
summary='''# 两轮方法细化记录

日期：2026-10-01。用户要求第一性原理、积极创新、小幅及有波动收益可接受。

最终评分 8.42/10，REVISE；新颖性 5.5/10，PROCEED WITH CAUTION。未达到 READY 的主要原因是新发现仍待实证，不能以加模块或润色填补。早期查新通过，可以推进小 pilot；无最低提升门槛。

'''+anchor+'''\n## 变化与收束

| Round | 核心问题 | 已修订 | 剩余 |
|---|---|---|---|
| 1 | 最近邻遗漏；训练先验可能来自已见目标；状态/遗忘不唯一；Hedge只比端点 | 加最近论文、清洁校准切分、掩码/无文本/衰减、混合候选Hedge、双路径成本 | 目标跨界、Hedge初始信息、旧参考隔离 |
| 2 | 阶段目标越界可能；Hedge未定义校准初始化；旧Health不是新分割对照 | 最终方案已补三项和零分母回退 | 真正运行校准pilot、内容必要性和完整成本 |

问题锚点逐字保留。主线只有文本路径互补性与最小风险利用；B语义瓶颈、C基础模型、D历史检索独立推进。没有加入GRU路由、MMD、动态专家或额外LLM agent。近日方法提供可借用机制和强对照，已有公式/协议不宣称独创。

本次还完成同账本原预测诊断，见 ../diagnostics_retry1/RESULTS.md。后验 oracle 明确不能部署；诊断不替代合法校准pilot。所有原数据时段已经被查看，不冒充独立未见确认。
'''
save('REVIEW_SUMMARY.md',summary)
raw1=(LOG/'round-1-review.md').read_text(encoding='utf-8');raw2=(LOG/'round-2-review.md').read_text(encoding='utf-8')
save('REFINEMENT_REPORT.md',summary+'\n## 完整评分\n\n'+scores+'\n## 原始审阅全文\n\n'+raw1+'\n\n'+raw2)
save('NOVELTY_REPORT.md','''# 查新结论

截至2026-10-01，核查原始论文方法与最新六个月预印本。完整检索式、最近邻与差异见 round-1-review.md；两轮请求/回复保留，审阅为同模型族暂定意见。

Recommendation: PROCEED WITH CAUTION。Score: 5.5/10。

最直接近邻是 [Downside-Controlled Online Forecast Combination](https://arxiv.org/abs/2609.29096)，已包含原预测缓存、成熟损失、冻结/静态/在线路径及校准warm start。[REST-TS](https://arxiv.org/abs/2606.19413) 已包含文本残差监督。[TESS](https://arxiv.org/abs/2603.12664) 已包含结构化语义瓶颈和受监督置信门控。

尚可核验的差异：异步外生文本路径的残差对齐如何随horizon变化，以及同一原预测账本中能否分离内容价值、在线分支价值和一般集成收益。该具体发现未被一篇检索到的论文完整预先包含，允许小pilot。算法部件已有；差异有多大要由真实结果回答，不预设顶会贡献成立。
''')
state=dict(phase='done',round=2,agent_id='/root/first_principles_method_review',last_score=8.42,last_verdict='REVISE',novelty_verdict='PROCEED WITH CAUTION',status='completed',completed_scope='method refinement and early novelty check only; experiments and whole iAdapter goal remain active',max_rounds=2,timestamp=datetime.datetime.now().astimezone().isoformat())
save('REFINE_STATE.json',json.dumps(state,ensure_ascii=False,indent=2))
request=json.loads((LOG/'review-request-2.json').read_text(encoding='utf-8'))
prompt=request.get('prompt') or request.get('message') or json.dumps(request,ensure_ascii=False)
(LOG/'review-prompt-2.txt').write_text(prompt,encoding='utf-8')
print('FINALIZED_METHOD_PLAN_AND_TWO_ROUND_HISTORY')

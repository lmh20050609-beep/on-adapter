# A：成熟反馈校准执行记录

全部预定评估完成：96条神经预测流（校准18、选择36、评估42），216条同账本组合序列，24轮隔离成本测量。复用既有骨干/训练预热，在线分支发生真实参数更新，不将216条组合序列称为216次神经训练。74项测试通过；首次日志序列化失败独立保留，新retry1输入复用和预测等同修复已记录。

| 阶段 | 状态 |
| --- | --- |
| source + 74 tests + code review + CUDA witness | passed |
| 2008 unique fixed input priors (470 new / 1538 reused) | completed |
| six calibration/selection groups + global rule lock | completed |
| six assessment groups + 216 mix sequences | completed |
| original-target/mask/maturity audit + independent raw witness | passed |
| cost + mechanism figures | completed |
| result-to-claim | partial / medium / same-family provisional |
| residual diagnostic | completed, post-hoc read-only |
| complete VoT comparison / C-B routes / source recovery | unfinished |

执行版本：3706909132eee80f35b4034702bf1b002cd682ea；准备/标签生成86895055。60/10/5/5/20、两个已见域、H6、三seed；规则先选择再assessment。LS/shared LS锁定rho=1/kappa=10；Hedge eta=10。完整科研目标保持active。

# 成熟反馈文本路径校准：result-to-claim 审查

- `claim_supported: partial`
- `confidence: medium`（对表内描述性结论信心高；对机制与迁移性信心有限）
- `review_independence: same-family`
- `acceptance_status: provisional`，待外部独立审查，不构成外部接受
- `integrity_status: unavailable`：未发现本 pilot 的 `EXPERIMENT_AUDIT.json`；已有确定性验证、代码复核及本次原始数据复核，不将其冒称完整独立实验审计。

本轮支持一个可复核、结果异质的探索性校准案例；不支持主要在线组合场景具有稳定增量、不支持文本语义互补性已经建立，也不支持 LS/Hedge 构成新算法。小于 1% 的正收益保留为进展，Energy 的反例及强静态基线同时保留。

## 审查范围与证据

读取 `research_contract.md`、`EXPERIMENT_PLAN.md`、`reports/validation.json`、`reports/branch_summary.csv`、`reports/summary.csv`、`reports/paired_summary.csv`、`reports/content_summary.csv`、`reports/RESULTS.md`、`BRANCH_PAIRED_SUMMARY.csv`、`cost/validation.json`、`cost/summary.csv`、两份预检查文件，并检查原始 WSL 目录中的配置、锁定参数、预测 NPZ 和更新 JSONL。原始目录为 `/home/wuwa/projects/research/iadapter/experiments/online_calibration_20261001_retry1`；神经与组合器执行配置记录 commit `3706909132eee80f35b4034702bf1b002cd682ea`，生成和后处理可能记录各自后续代码版本，不混作同一阶段版本。

本次没有调用项目评分函数：从原 `data/Energy.csv` 和 `data/SocialGood.csv` 按 `end_date` 稳定排序，独立重建 OT 缺失掩码、前 60% 训练期标准化和评估窗口，复算 36 条分支序列及 18 条 LS 组合序列。36 条包括六组的 numeric online/static、LLM5 online/static、raw5 online 与 full online；18 条为两域三 seed 的全部三个 LLM5 组合场景。Energy 每个流 291 个原点、1,746 个有效 cell；SocialGood 每个流 175 个原点、1,017 个有效 cell、33 个缺失 cell。原点、原 CSV 真值及掩码一致，分数与原记录吻合；18 条 LS 的预测均等于所保存 alpha 对两个已发出预测的凸组合，最大误差小于 1e-12；每个原点只在全 H=6 成熟时消费一次，最后六个尚未成熟原点未被 flush。证据保存为 `RESULT_CLAIM_RAW_WITNESS.json`。

全量 `reports/validation.json` 另报告 96 条神经预测流、42 条评估分支与 216 条组合评估序列通过。216 是相同分支账本上的组合序列，不是 216 次神经训练。本次抽样复核与该全量审计一致，但本审查没有独立逐一重跑全部 216 条。训练预热和 checkpoint 来源沿用 0–60%/60–70%，本轮没有新 fit；指定 online 分支有实际参数变化，static 分支位移为零。校准、规则选择、评估分别是 70–75%、75–80%、80–100%，阶段状态重置。`selection/locked.json` 的 LS/shared LS 为 rho=1、kappa=10，两个 Hedge 的 eta=10；不能将这个选择结果解释为已证明遗忘因子必要。

全部时期此前已被研究查看，因此时序隔离和规则锁定只保证本轮协议的执行，不把已见时期转成未触碰确认集。首次 sanity 的日志序列化失败与原预测等同修复证据被保留，失败目录不进入正式汇总；未发现由该失败直接推翻本轮表内分数的依据。预检查 9 项 verified 仅验证数字存在；原 CSV 邻接字符串匹配的误报修复不代替语义判断。

## 1. 数据支持什么

以下百分比是三 seed 平均 MSE 的相对变化，Δ=(候选/对照−1)×100%，负值为改善；seed 计数与 SD 是优化随机性描述，不是独立时间重复或显著性检验。

| LLM5 LS 场景与对照 | Energy ΔMSE | SocialGood ΔMSE | 改善 seed 数（Energy / SocialGood） |
|---|---:|---:|---:|
| 主要：numeric-online + text-online，相对 numeric-online | +2.7998% | −0.3650% | 0/3 / 1/3 |
| 次要：numeric-online + text-static，相对 numeric-online | −8.6994% | −1.4554% | 3/3 / 3/3 |
| 次要：numeric-static + text-static，相对 numeric-static | +0.8485% | +0.4733% | 0/3 / 0/3 |

主场景的 MSE 均值是 Energy 0.09370755、SocialGood 0.84496391。SocialGood 相对 numeric-online 的小幅改善应报告；但三个配对绝对差依次为 +0.00081598、−0.01039340、+0.00029220，因此均值改善主要来自 seed 2024。Energy 三个差均为正（+0.00091492、+0.00122038、+0.00552127）。不能以次场景替换预先指定主场景。

分支表示比较也存在值得保留的有限正证据：LLM5-online 相对 raw5-online，Energy 平均 MSE 改善 7.1296%（2/3），SocialGood 改善 0.6341%（3/3）。静态比较并不一致：Energy 的 LLM5-static 相对 raw5-static 改善 5.0377%，只有 1/3 seed 改善；SocialGood 反而退化 0.3359%，0/3 改善。这支持“在所测在线分支条件下，LLM5 表示的分支风险均值优于 raw5”，不把它升级成语义信息的独立贡献。

分离账本具有诊断价值：相同评估原点与掩码下，numeric-online 相对 numeric-static 的 MSE 增加 Energy 39.5498%、SocialGood 3.4607%，LLM5-online 相对 LLM5-static 增加 35.3029%、1.4492%，均为 0/3 seed 改善。它直接指出本设置中在线更新是重要退化源，因而“加入静态文本分支后缓解在线数值分支退化”和“文本带来强数值基线之外的新信息”必须分开。

较小实现有可测成本优势：双分支适配器总参数 32,128，相对 full_online 的 166,832 少 80.7423%（均不含骨干，text-static 场景该总量也不等于正在更新参数量）。双在线更新中位数 Energy 18.717 ms、SocialGood 16.339 ms，低于 full_online 的 21.689、23.991 ms；但双在线预测为 10.311、9.469 ms，高于 full_online 的 7.488、8.744 ms。可以报告参数减少和该隔离微基准的更新成本下降，不能报告全面加速。192 bytes 仅为 gate 四个 float64 数组的 payload，不是总内存、总状态或通信成本。

## 2. 数据不支持什么

**不支持 LS 的主要在线组合优于已有简单集成或单分支。** 在主要场景，LS 相对等权 MSE 分别退化 2.5850%/0.0779%，相对 mixed Hedge 退化 0.5938%/0.0811%，相对 LLM5-online 退化 1.5809%/0.3959%；这些均值没有给出按 h LS 的必要性。主要场景其他组合族也没有超过本域强 numeric-static 的平均 MSE。

**次场景的正收益不证明突破强基线。** numeric-online + LLM5-static 的 LS 仍输给 LLM5-static：Energy +22.0676%、SocialGood +0.7361%，均 0/3 seed 改善；也输给 numeric-static。两个 static 分支的 LS 虽能相对等权在 Energy 改善 0.6620%、SocialGood 改善 0.0434%，且两域各有 2/3 seed 改善，但仍分别比 numeric-static 差 0.8485%、0.4733%。这些小幅相对等权进展保留，不能写成新增文本信息已被证实。

**不支持文本当前语义有效或因果贡献已经确定。** 同一已由原文本训练/更新形成的状态下，zero_current_text 的 ΔMSE（探针−原预测）为 Energy LLM5-online −0.01152143、raw5-online −0.01951590，SocialGood 分别 −0.00635281、−0.00612840；四组均 3/3 seed 改善。历史 donor 的平均变化更小，但 donor 是训练前缀中的历史替换，未验证语义或五维表示确实不匹配。该结果是当前文本输入的诊断反证，不是完整无文本学习政策的收益，也不能推出“文本永久无用”或文本具有确定负因果效应。

**不支持新算法、显著性、未见期泛化或端到端部署结论。** LS/Hedge 是既有组合方法，本轮没有新理论或算法创新证明；三 seed 共用同一时期，窗口与 horizon 重叠。合并 end_date 只是可用性代理，未核实原新闻 publication-time。470 个新增唯一文本加 1,538 个复用输入并不提供语义真值；格式全部有效与语义正确无关。84.875 秒仅是新增 470 个输入的生成成本，非全量 2,008 个输入的生成总成本。成本来自固定输入/成熟记录的隔离微基准，排除生成、编码、输入读取、日志等，不能等同部署吞吐率。

## 3. 缺少的关键证据

1. **强基线之外的残差信息。** 需证明当前文本能预测强 numeric-static 未解释的残差，而非一般收缩、分支随机性或补偿 online 退化；目前最佳可解释机制仍不唯一。
2. **文本内容和学习轨迹分离。** 当前 zero/donor 仅替换推理输入，过去原文本已影响参数。需要小规模、匹配训练和全 H 成熟更新的真实无文本/内容对照，且 donor 若用于语义判断应验证差异与时间可用性。
3. **外推与不确定性。** 当前已见双域 H6 和三个优化 seed 不足以支撑普适效应。若要升级为确认性主张，需要不曾用于研究选择的新时期或数据，并以时间块为单位处理重叠相关性；无需为了认可小幅进展人为设置 2% 或显著性门槛。
4. **时效与来源。** publication-time、完整 VoT 先验比较及原投稿/官方来源恢复仍是独立未完成事项。不能以本 pilot 替代这些工作。

## 4. 修订后的可用主张

> 在已被研究查看的 Energy 与 SocialGood、OT/H6、三个优化 seed 上，我们构建了按全预测窗口成熟后一次性消费反馈的可复核账本，区分静态/在线神经分支与低维组合器。校准显示收益依赖分支更新状态：预先指定的在线数值+在线 LLM5 场景中，按 horizon LS 相对在线数值分支在 SocialGood 平均 MSE 改善 0.365%，但仅 1/3 seed 改善，在 Energy 退化 2.800%；静态文本次场景相对在线数值分支两域均改善，却未超过强静态数值基线或静态文本分支。LLM5-online 相对 raw5-online 有小幅或异质的分支层面优势。该结果支持报告探索性增量及在线更新退化诊断，尚不足以确认文本语义互补性或自适应组合优于简单方法。

成本可以另附一个受限句子：所测双分支方案的适配器参数更少、隔离微基准更新更便宜，但双在线预测更慢；组合器 192 bytes 是数组 payload。

## 5. 有限、针对性的下一步

1. 先用已有账本进行**诊断性残差分解**：以强 numeric-static 为锚，按 h 报告 r=Y−n、d=m−n 的交叉项 2αrd 与惩罚项 α²d²，分开描述文本可用性、当前 zero/donor 和 seed。该事后分解只解释现有结果，不再用 assessment 选新门控阈值或报告其最优值为验证结果。
2. 下一轮优先收窄到 **C：强冻结数值基线**。已有 static/static 混合本身仍输 numeric-static，因此不能仅重跑同样组合便宣称解决问题；先要求 **B：文本能解释该基线的独有残差** 的证据。若有合理方向，只做一个小型残差文本修正方案和其必要的 numeric-only、零文本、既有 raw5 对照，保持参数更新真实、反馈成熟规则不变、预算与数据边界明确。无需增添巨型路由或模块网格。
3. 在新时期上锁定上述一个机制后进行小规模确认；如残差证据仍为空，保留冻结基线和本轮有价值的异质/负结果，收窄方法主张。无须为了越过人为收益门槛扩大搜索。

本审查未启动训练、改动科学源码或预测结果，也未标记总研究目标完成。完整 VoT 对比、原始来源缺口及后续创新路线仍在本 pilot 之外。

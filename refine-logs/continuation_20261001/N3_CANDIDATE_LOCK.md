# 追加域候选锁定

候选锁定为上一轮LLM5简单融合，未按追加域测试选择；N1近期文本研究独立，不将其最好的结果替换进N3。Agriculture月度retailbroilercomposite/HealthUSNationalunweightedILI%。Health保留5组来源重复calendarinterval的不同YEAR/WEEK真实记录，H6为后续6条reportedobservations，明确不称处处恰好6个calendarweeks。

OT,H6,L36,seeds2023–2027，60/20/20。骨干最多30epoch、batch16/lr.001/patience5；d32/2层，10共享骨干。frozenbone、numeric_online、rawPCA5_online、LLM5_static/online共50评价；numeric/raw/LLM各warm一次，共30。5维raw/LLM同初始化、同adapter结构，PCA/std仅训练前缀；缺文后投影置零。只用本机，相同encoder和0.5B固定weights/SYSTEM/greedy/max8，新增目标描述。≤1885unique新输入，新增Qwen生成累计600秒wall-time预算（含模型加载、tokenization及失败调用）。BERT预处理和PCA另记成本，不占生成预算。

所有原点/target/mask与原CSV核对，骨干不变，标签成熟一次，LLMstaticonline首7一致。fresh代码审查PASS（same-family provisional）；原ZIP、训练前缀PCA/std未来扰动及生成完成/失败/预算控制流检查通过。新增端到端GPU执行、完整Torch测试及首Agriculture/2023 sanity尚未完成，不能把静态检查当实验成功。部署源码待WSL恢复后应用并提交。

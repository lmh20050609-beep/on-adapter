# A1 固定配置扩展：五域、五seed、三种静态方法

运行前固定。沿用STATIC_PILOT_PLAN.md全部方法定义，不因Economy/2023的误差调整设置。Economy、Energy、Security、SocialGood、Climate × seeds2023–2027 × numeric_static / iadapter_static / gpt4mts_static，共75个单元。直接引用已验证的Economy/2023三个pilot结果，新增72个；不重跑后择优。各组内部共享自己的OT骨干，不能复用多变量骨干。

H6；Energy L48，其余L36；60%训练、60%–70%选模、70%–80%开发评分。所有方法只使用OT数值历史，保持原掩码/尺度。共同选模上限30epochs、Adam lr1e-3、batch16、patience5。作者GPT4MTS始终六层、768维、patch8/stride4；minimal骨干始终两层、32维。内部适配器的训练流预热一遍和静态评分的零反馈沿用pilot。编码规则不改，GPT4MTS CLS512前80%缓存按域共享，Economy直接复用已验证pilot缓存。

这是一张**部分外部方法的开发静态比较表**，不是原四方法100单元矩阵的完成标记。VoT来源缺口未解决，仍缺席；原生在线扩展也未加入。不同模型的文本历史长度/池化和训练阶段仍不同，保留这些披露。既有测试与开发信息已被查看，不称独立确认。

执行按域、seed、方法串行；任一NaN、资源错误、原点/目标或原CSV审计失败即停止扩展，保留失败记录并修复。未根据成绩新增超参/结构候选。前一阶段每组三方法约14.2秒仅作小域参考，较大域和更多epoch实际时间另记录，不能直接承诺75单元耗时。

输出`experiments/external_static_matrix_20260929/`，每dataset/seed独立子目录，顶层队列登记来源路径。每组运行后立即调用原CSV审计；最终严格检查75个唯一键，没有缺失/重复/失败单元才能汇总。报告各域MSE/MAE的五seed均值、样本SD和逐seed差值；native与内部仅按相同seed标签和相同流对齐，不能宣称跨架构初始参数相同，不做显著性推断。

新增只有dataset/seed参数、缓存路径和队列/汇总；原训练核心train_native、静态evaluate与OT数据处理不得改变。用源AST核验这些函数和共同核心文件，另核对默认pilot配置保持一致；复用原组先重做原CSV审计。主表不平均不同域的原始误差以掩盖退化，也不将静态分数与原在线分数混合。

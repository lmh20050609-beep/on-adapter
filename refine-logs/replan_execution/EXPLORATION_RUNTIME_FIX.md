# 已发生的批处理数值修复

首个8条批推理触发实际NaN并停止，未添加任何新生成记录，原30条缓存保留。对8条已缓存训练输入复核：每一步1063552个NaN，非正/负inf抑制值；两条独立singleton输出全部finite，与原预检UNKNOWN一致。原始诊断在experiments/replan_20260930/b3/batch_diagnostic.json，未读取未来数值标签。

因此仅将生成encoder batch由8降为1，保留FP16/greedy/max8/模型/prompt/输入预算。增加finally预算记账与failed状态；失败时缓存保留，不把运行成功误写为已完成。第一失败调用整体工具墙钟7.8693406秒作为内部耗时保守上界记入累计预算；诊断也计入。不要把这个上界当精确GPU计时。预算仍3600秒。

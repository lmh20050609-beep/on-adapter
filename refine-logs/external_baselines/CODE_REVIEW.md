# GPT4MTS 接入运行前审查

GPT-6-Astra xhigh，新上下文，只读审查；same-family provisional。首次代理请求因上游502失败，重试后完成。

结论：无阻塞项，可以执行一次 integration smoke；尚不代表正式比较已通过。

确认两处AST变换精确，RevIN实现保留，六层GPT2/patch8/stride4与冻结规则正确；同实例eval的batch2等价对照与batch1修复合理；两个历史窗口分别截止t/t+1，一次更新只用t对应的真实未来OT目标。输出不报告基线准确率。

两条非阻塞建议已修正：明确文本先经过现有清洗/合并/裁剪再使用完整BERT CLS512；原batch1回归只接受einops维度错误，其他异常不再被记录为成功复现。

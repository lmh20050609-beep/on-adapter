# 同预算近期文本实现状态

生产仓库WSL控制连接失败，当前代码保存为source_patch，尚未写入生产仓库、尚未GPU执行。text_order纯字符变换3项标准库unittest通过；recent_text完整编码/回放/队列模块语法编译通过，fresh代码审查PASS（same-family provisional）。审查另核对1903条真实文本、5309个正文span及12个旧结果单元；仍需恢复WSL后的真实编码/sanity。

对照固定：现有current768与新recent768，同64token/相同已经裁剪的字符块，仅交换两条时间记录先后，不改变块内fact/preds。12新评估、6warm、6current诊断重放、12donor；同数值骨干、参数初始化、60/10/10/H6/seeds2023–2025。当前诊断重放必须匹配旧原预测；强数值/裸骨干/staticGPT/LLM5参考保留。

部署顺序：将source_patch中的两个新模块及test复制入WSL；源码审查/完整测试/提交后，python -m iadapter.recent_text --stage prepare；--stage campaign --limit 1完成Energy/2023 sanity；同源码--stage campaign完成其余5组；原CSV核验和结果分析。旧缓存/结果不覆盖，不借用Windows缺Torch环境冒充GPU实验成功。

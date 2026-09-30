# 候选成本测量设置

仅新增iadapter/candidate_cost.py包装器，复用既有efficiency.measure；没有改动任何正式训练/预测模块。两域×raw5/LLM5/numeric_online×3轮=18轮，各50预热/200测量，cached预测和fixed成熟记录更新分别计时；每轮重新加载warm副本，原权重/预测文件不写入。

计时包含CUDA同步、实际文本投影、预测骨干与成熟损失/梯度/clip/Adam/反馈刷新；不含队列、日志、输入加载、BERT/LLM预处理。这不是完整部署吞吐。报告各轮median/P95，再给三轮中位数及峰值allocated最大值；调用不是独立数据重复，不做显著性。

LLM真实生成与失败/诊断累计成本另列；CPU文本缓存只报5维float32数组payload，不能当整个Python进程RSS。静态GPT等强参考使用此前系统成本记录，不把未测项设为零。产物在profile_output/e003_20261001，最终复制入证据包。

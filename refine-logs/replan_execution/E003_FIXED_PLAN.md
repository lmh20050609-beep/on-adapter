# E003：小幅局部内容信号的简单接入

用户要求后预先固定：B3 Energy LLM5相对metadata P1平均MSE降低1.1149%，3/3seed改善，donor更差；尽管比rawPCA5更差且SocialGood退化，仍继续一套相同5维简单融合验证。这不篡改原B3两域门槛。

选择LLM而非规则，因为规则的历史donor平均反而更好，未满足本轮“内容线索”条件。raw与LLM均5维，同一个5→96→64原加法结构，ReLU/dropout0.1/LayerNorm64，保留数值反馈，无文本反馈、专家或entropy。仅更换首层输入尺寸，共享全部同形状初始参数；每表示每domain/seed训练流warm一次再同warm分叉static/online。原点、mask、学习率、骨干、切分均沿用。使用B3已经训练段拟合的rawPCA与LLM标准化，不重新拟合选择段。

两域×三seed×两表示×两mode＝24新评价、12warm；12条在线同状态historical donor（参数和历史反馈不因探针改变）。先Energy/2023 raw、LLM两组4格sanity，再全部12组持久队列串行。评分核对原CSV；结果包括numeric在线、裸骨干及staticGPT参考，不声称完全等容量外部比较。

报告小幅、局部、波动的平均变化与逐seed差值；不以低于2%拒绝进展。若候选任一域在线相对numeric或raw平均MSE改善，记录其具体进展；若static更好，不把表示收益归因在线过程。扩域不因单个最好seed自动触发。本轮先完成这套简单接入，不增加路由网络、LR网格和更多LLM。

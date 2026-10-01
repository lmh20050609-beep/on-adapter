# 已发出预测的残差分解（事后诊断）

以同 seed 的 numeric-static 为锚 n，r=Y−n，d=候选原预测−n。报告 ΔMSE=mean(d²)−2mean(rd)。正交叉项表示候选修正与残差方向对齐，但必须大于修正幅度惩罚才降低误差。这里 d 包含两种学习政策差异，不能直接解释为文本因果贡献。LS 项使用此前已锁定并发出的实际预测，不拟合任何 assessment 系数。所有结果为已见时期的诊断，不选择新阈值或 oracle。

| dataset | candidate | subset | lead | n | valid_cells_per_seed | numeric_static_mse | alignment_benefit | difference_penalty | delta_mse | delta_mse_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Energy | LLM5_online | all | 0 | 3 | 1746 | 0.065321 | -0.001002 | 0.025926 | 0.026928 | 0.012346 |
| Energy | LLM5_static | all | 0 | 3 | 1746 | 0.065321 | -0.001114 | 0.001744 | 0.002859 | 0.001989 |
| Energy | LS_numeric_online_text_online | all | 0 | 3 | 1746 | 0.065321 | -0.001329 | 0.027057 | 0.028387 | 0.011403 |
| Energy | LS_numeric_online_text_static | all | 0 | 3 | 1746 | 0.065321 | -0.004338 | 0.013567 | 0.017904 | 0.006607 |
| Energy | LS_numeric_static_text_static | all | 0 | 3 | 1746 | 0.065321 | -0.000402 | 0.000153 | 0.000554 | 0.000339 |
| Energy | full_online | all | 0 | 3 | 1746 | 0.065321 | 0.000825 | 0.103285 | 0.102460 | 0.095282 |
| Energy | numeric_online | all | 0 | 3 | 1746 | 0.065321 | -0.000962 | 0.024872 | 0.025834 | 0.009012 |
| Energy | raw5_online | all | 0 | 3 | 1746 | 0.065321 | 0.000251 | 0.034261 | 0.034010 | 0.019012 |
| Energy | raw5_static | all | 0 | 3 | 1746 | 0.065321 | -0.000706 | 0.005770 | 0.006476 | 0.007407 |
| SocialGood | LLM5_online | all | 0 | 3 | 1017 | 0.819692 | -0.009659 | 0.012282 | 0.021940 | 0.006077 |
| SocialGood | LLM5_static | all | 0 | 3 | 1017 | 0.819692 | -0.007031 | 0.002886 | 0.009918 | 0.007304 |
| SocialGood | LS_numeric_online_text_online | all | 0 | 3 | 1017 | 0.819692 | -0.013213 | 0.012059 | 0.025272 | 0.008484 |
| SocialGood | LS_numeric_online_text_static | all | 0 | 3 | 1017 | 0.819692 | -0.011226 | 0.004798 | 0.016024 | 0.007521 |
| SocialGood | LS_numeric_static_text_static | all | 0 | 3 | 1017 | 0.819692 | -0.003230 | 0.000650 | 0.003879 | 0.000959 |
| SocialGood | full_online | all | 0 | 3 | 1017 | 0.819692 | -0.001032 | 0.027838 | 0.028870 | 0.024812 |
| SocialGood | numeric_online | all | 0 | 3 | 1017 | 0.819692 | -0.016544 | 0.011823 | 0.028367 | 0.014714 |
| SocialGood | raw5_online | all | 0 | 3 | 1017 | 0.819692 | -0.011858 | 0.015453 | 0.027311 | 0.007357 |
| SocialGood | raw5_static | all | 0 | 3 | 1017 | 0.819692 | -0.004263 | 0.002877 | 0.007140 | 0.007060 |

lead=0 为全 H 汇总；其余逐 lead。text_available/no_text 按原点已有文本ID分组，是描述性分组，不推断因果。

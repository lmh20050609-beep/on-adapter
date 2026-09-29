# 共同延迟协议在线扩展

75/75运行完成，25组原CSV/起始权重/成熟事件核验通过。GPT4MTS是新定义的online extension；内部方法保留原反馈机制。起始权重与静态表完全相同，前7个尚未更新的预测在fp32容差内一致。

| dataset | method | n | mse_mean | mse_sd | mae_mean | mae_sd | updated | origins |
|---|---|---|---|---|---|---|---|---|
| Economy | numeric_online | 5 | 0.085257 | 0.014368 | 0.233085 | 0.022598 | 31 | 37 |
| Economy | iadapter_online | 5 | 0.097277 | 0.01891 | 0.251902 | 0.028118 | 31 | 37 |
| Economy | gpt4mts_online_extension | 5 | 0.086951 | 0.012681 | 0.234353 | 0.016382 | 31 | 37 |
| Energy | numeric_online | 5 | 0.014084 | 0.000705 | 0.090779 | 0.002605 | 137 | 143 |
| Energy | iadapter_online | 5 | 0.020119 | 0.007165 | 0.104631 | 0.014458 | 137 | 143 |
| Energy | gpt4mts_online_extension | 5 | 0.02739 | 0.004321 | 0.111559 | 0.011173 | 137 | 143 |
| Security | numeric_online | 5 | 2.782187 | 0.103676 | 1.22492 | 0.01478 | 19 | 25 |
| Security | iadapter_online | 5 | 2.77954 | 0.110431 | 1.226069 | 0.010977 | 19 | 25 |
| Security | gpt4mts_online_extension | 5 | 5.208139 | 0.203605 | 1.781644 | 0.049025 | 19 | 25 |
| SocialGood | numeric_online | 5 | 0.144483 | 0.032429 | 0.220996 | 0.022733 | 79 | 85 |
| SocialGood | iadapter_online | 5 | 0.146866 | 0.032663 | 0.223024 | 0.022681 | 79 | 85 |
| SocialGood | gpt4mts_online_extension | 5 | 0.420954 | 0.058396 | 0.363127 | 0.024381 | 79 | 85 |
| Climate | numeric_online | 5 | 1.012202 | 0.014017 | 0.78221 | 0.00813 | 38 | 44 |
| Climate | iadapter_online | 5 | 1.009755 | 0.012885 | 0.783035 | 0.007228 | 38 | 44 |
| Climate | gpt4mts_online_extension | 5 | 1.413786 | 0.044122 | 0.939274 | 0.025238 | 38 | 44 |

## Online减Static

负值表示在线误差更低。内部方法同时启用参数更新与反馈刷新，不能单独归因于缓存。

| dataset | method | n | delta_mse_mean | delta_mse_sd | delta_mae_mean | delta_mae_sd | mse_improved_seeds | mae_improved_seeds |
|---|---|---|---|---|---|---|---|---|
| Economy | numeric_online | 5 | -0.002265 | 0.0014 | -0.00277 | 0.002106 | 4 | 4 |
| Economy | iadapter_online | 5 | -0.008036 | 0.01052 | -0.012897 | 0.018011 | 5 | 4 |
| Economy | gpt4mts_online_extension | 5 | 0.025332 | 0.002817 | 0.035581 | 0.006685 | 0 | 0 |
| Energy | numeric_online | 5 | 0.000102 | 0.000125 | 0.000187 | 0.000579 | 0 | 2 |
| Energy | iadapter_online | 5 | -0.00695 | 0.007417 | -0.015869 | 0.015216 | 4 | 4 |
| Energy | gpt4mts_online_extension | 5 | 0.012042 | 0.003039 | 0.018917 | 0.006462 | 0 | 0 |
| Security | numeric_online | 5 | 0.031969 | 0.012372 | 0.006527 | 0.002723 | 0 | 0 |
| Security | iadapter_online | 5 | 0.061104 | 0.025035 | 0.010431 | 0.004185 | 0 | 0 |
| Security | gpt4mts_online_extension | 5 | 2.502958 | 0.204332 | 0.554928 | 0.04133 | 0 | 0 |
| SocialGood | numeric_online | 5 | -0.001542 | 0.00119 | -0.00056 | 0.001272 | 4 | 3 |
| SocialGood | iadapter_online | 5 | -0.002625 | 0.00445 | -0.000796 | 0.003922 | 4 | 3 |
| SocialGood | gpt4mts_online_extension | 5 | 0.323543 | 0.057728 | 0.167008 | 0.021288 | 0 | 0 |
| Climate | numeric_online | 5 | 0.012914 | 0.002885 | 0.004487 | 0.001491 | 0 | 0 |
| Climate | iadapter_online | 5 | 0.003159 | 0.001986 | 0.001208 | 0.000608 | 0 | 0 |
| Climate | gpt4mts_online_extension | 5 | 0.367219 | 0.044565 | 0.149512 | 0.015619 | 0 | 0 |

文本历史/编码与训练阶段的跨方法差异仍存在；各方法仅共享原点/目标、可用性代理与成熟更新机会。五seed是固定开发流上的优化重复，无显著性或独立泛化声明。仍无可信VoT来源，不恢复原Table2。

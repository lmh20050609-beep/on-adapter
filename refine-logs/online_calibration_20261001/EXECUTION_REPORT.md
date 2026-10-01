# 按第一性原理推进 iAdapter：本轮交付与路线

本轮已经跑完。问题收窄为“文本能否提供强数值预测未解释的信息”，用原点发出的预测与成熟反馈检验，不把增加组件本身当作进步。

1. **保留小收益。** 主LS相对numeric-online在SocialGood改善0.365%（1/3 seed），Energy退化2.800%（0/3）。LLM5-online相对同维raw5-online分别改善7.130%（2/3）和0.634%（3/3）。static/static LS相对等权改善0.662%和0.0434%，两域均2/3。
2. **说明收益来源。** numeric-online+LLM5-static相对numeric-online改善Energy8.699%、SocialGood1.455%（各3/3），但仍输text-static和numeric-static。两域numeric-online分别比其static差39.550%和3.461%，本设置在线更新退化不能当成新增文本信息。
3. **把复杂度收益分开报告。** 双分支适配器32,128参数，完整模型166,832，减少80.742%；更新微基准较便宜，但双在线预测更慢。192bytes仅gate数组payload。470个新增文本生成84.875秒，不能写成全部2008输入成本。
4. **下一条可检验路线。** C先检查一个冻结现代数值基础模型的真实接口/显存/数据边界，再让B的小残差头解释该模型的独有误差。必要对照为冻结基线、数值残差头、raw5、LLM5和真实零文本学习政策。不重复在assessment上调A的门控，不先扩专家或堆GRU/MMD。

表内证据为已见时期的探索，三seed是优化随机性。end_date仍为文本可用性代理。完整VoT比较、原投稿表格来源和未见期确认仍未完成。独立复算36分支/18 LS序列通过；完整审查判定partial，同模型族暂定，不冒称外部接受。

## 同协议分支

| dataset | method | n | mse_mean | mse_sd | mae_mean | mae_sd |
| --- | --- | --- | --- | --- | --- | --- |
| Energy | LLM5_online | 3 | 0.092249 | 0.010214 | 0.201771 | 0.005464 |
| Energy | LLM5_static | 3 | 0.068180 | 0.001194 | 0.181446 | 0.002928 |
| Energy | full_online | 3 | 0.167781 | 0.097491 | 0.262765 | 0.056724 |
| Energy | numeric_online | 3 | 0.091155 | 0.006814 | 0.201177 | 0.005236 |
| Energy | numeric_static | 3 | 0.065321 | 0.002251 | 0.175329 | 0.001841 |
| Energy | raw5_online | 3 | 0.099331 | 0.017942 | 0.205888 | 0.015456 |
| Energy | raw5_static | 3 | 0.071797 | 0.009518 | 0.187617 | 0.013987 |
| SocialGood | LLM5_online | 3 | 0.841632 | 0.018035 | 0.399678 | 0.012706 |
| SocialGood | LLM5_static | 3 | 0.829610 | 0.016419 | 0.395310 | 0.008900 |
| SocialGood | full_online | 3 | 0.848562 | 0.017204 | 0.409600 | 0.016207 |
| SocialGood | numeric_online | 3 | 0.848059 | 0.023594 | 0.401397 | 0.013070 |
| SocialGood | numeric_static | 3 | 0.819692 | 0.012413 | 0.392895 | 0.009778 |
| SocialGood | raw5_online | 3 | 0.847003 | 0.018096 | 0.402736 | 0.011315 |
| SocialGood | raw5_static | 3 | 0.826832 | 0.015113 | 0.394932 | 0.007819 |

## 预声明主场景：numeric-online + LLM5-online

| dataset | representation | numeric_mode | text_mode | family | n | mse_mean | mse_sd | mae_mean | mae_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Energy | LLM5 | online | online | endpoint_hedge | 3 | 0.093588 | 0.009282 | 0.202708 | 0.006162 |
| Energy | LLM5 | online | online | equal | 3 | 0.091346 | 0.008662 | 0.201121 | 0.005524 |
| Energy | LLM5 | online | online | fixed | 3 | 0.092300 | 0.008673 | 0.201339 | 0.005442 |
| Energy | LLM5 | online | online | ls | 3 | 0.093708 | 0.009264 | 0.202854 | 0.006263 |
| Energy | LLM5 | online | online | mixed_hedge | 3 | 0.093154 | 0.008873 | 0.202462 | 0.005969 |
| Energy | LLM5 | online | online | shared_ls | 3 | 0.093897 | 0.009701 | 0.203008 | 0.006783 |
| SocialGood | LLM5 | online | online | endpoint_hedge | 3 | 0.844398 | 0.018923 | 0.400729 | 0.012518 |
| SocialGood | LLM5 | online | online | equal | 3 | 0.844306 | 0.020036 | 0.399841 | 0.012449 |
| SocialGood | LLM5 | online | online | fixed | 3 | 0.844470 | 0.020903 | 0.400311 | 0.013078 |
| SocialGood | LLM5 | online | online | ls | 3 | 0.844964 | 0.018724 | 0.401324 | 0.012315 |
| SocialGood | LLM5 | online | online | mixed_hedge | 3 | 0.844279 | 0.019322 | 0.400391 | 0.012494 |
| SocialGood | LLM5 | online | online | shared_ls | 3 | 0.845274 | 0.020522 | 0.400765 | 0.012389 |

## 隔离成本（毫秒）

| dataset | policy | adapter_parameters | prediction_median_ms | prediction_p95_ms | update_median_ms | update_p95_ms |
| --- | --- | --- | --- | --- | --- | --- |
| Energy | full_online | 166832 | 7.488122 | 11.010197 | 21.688836 | 25.888049 |
| Energy | numeric_online_text_online | 32128 | 10.311347 | 16.949801 | 18.717156 | 22.736515 |
| Energy | numeric_online_text_static | 32128 | 8.777867 | 14.527205 | 9.578796 | 12.384347 |
| Energy | numeric_only | 10496 | 3.081689 | 8.268974 | 5.777310 | 9.388045 |
| SocialGood | full_online | 166832 | 8.743700 | 10.806350 | 23.991403 | 29.772691 |
| SocialGood | numeric_online_text_online | 32128 | 9.468509 | 14.386642 | 16.339240 | 20.565151 |
| SocialGood | numeric_online_text_static | 32128 | 8.200586 | 14.502289 | 7.868939 | 10.224369 |
| SocialGood | numeric_only | 10496 | 5.981065 | 7.452177 | 7.981045 | 10.804897 |

成本每域每政策3轮，50次预热/200次测量；包含两条路径独立计算和预测转CPU，排除读入、队列日志、离线编码生成。重复固定成熟记录仅用于计时，不是部署吞吐率。完整原始预测、掩码、原点、反馈、锁定配置、失败记录和文本cache另封存，既有权重仍保留在WSL。

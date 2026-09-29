# 原生外部基线：接入证据与剩余条件

2026-09-29。GPT4MTS 作者模型已完成实际GPU前向/梯度接入检查；VoT 已完成官方来源与预生成预测审计。尚未生成外部性能排名。

## GPT4MTS 已解决什么

官方来源固定在[作者源码提交](https://github.com/Flora-jia-jfr/GPT4MTS-Prompt-based-Large-Language-Model-for-Multimodal-Time-series-Forecasting/tree/c85e74b2fd7048f5ef601453e5bcb9f6073cd3b4)。项目内`external/gpt4mts/`保留未修改原文件，ZIP CRC检查通过。GPT2 safetensors来自openai-community/gpt2，revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`，548,105,171字节，已通过实际预训练模型加载。

`iadapter/native_gpt4mts.py`载入作者模型，保留六层GPT2、多token历史prompt、patch8/stride4、RevIN和作者冻结规则。与投稿包单token的GPT4MTS_z分开。两项运行兼容变换有明确记录：本地权重路径；移除summary均值后会丢失batch维度的无参squeeze。原始文件没有被改写。

实际检查使用Economy、OT单变量、L36/H6，原点295/296；源提交`aad297d8bd79ca5dfa844e5090b6d9b17e62a6a5`，工作树干净。结果：

| 检查 | 结果 |
|---|---|
| 同一权重、batch2、eval：原forward与兼容forward | 逐元素完全相同 |
| 原batch1错误 | 已复现EinopsError：期望3维、收到2维 |
| 兼容batch1预测 | `[1,6,1]`，全部有限 |
| 实际目标更新 | 一次masked OT损失更新，6个有效目标；输出头发生变化 |
| 冻结路径 | 所有冻结参数梯度均为None；输入/文本/输出投影梯度存在且有限 |
| 可训练参数 | 1,445,384 |
| GPT4MTS总参数 | 82,551,560，**不含**独立冻结BERT编码器 |
| GPT4MTS阶段峰值allocated显存 | 379,315,200字节，约361.7MiB；BERT已先释放，非完整系统峰值 |

文本使用当前数据清洗规则后每个历史时刻的完整12层BERT CLS/512；它与原iAdapter主表的masked mean/64不同。编码器明确eval且冻结。每字段子句数量、两行合并和1200字符裁剪仍保留，不能称作未经裁剪的完整原文。

环境新增官方PyPI `einops==0.8.1`，安装时`--no-deps`；torch/transformers/numpy/pandas版本未变。新代理按环境文档执行CUDA和einops见证均通过。最初缺依赖的尝试在模块导入时退出，未产生实验结果目录；此次成功记录单独保存。

这些检查证明作者模型能够在本机执行，且兼容修复没有改变被检查的batch2输出。一次更新不证明预测效果、收敛或通用等价性，不能计入主表。

## VoT 当前的具体缺口

来源固定在[官方提交](https://github.com/decisionintelligence/VoT/tree/f769621da9efde3e82a1975f478efdec83dcbd10)，完整归档86个文件。已读取10份发布的预生成预测JSON，但没有将其用于训练或评分。

1. `data_provider/data_loader.py:250`查找`stat_predictions_ecnu_rag`，实际发布目录为`stat_predictions_rag`。隔离执行原`get_text_pred`方法，已复现`text_pred_json`未赋值的UnboundLocalError。目录问题可以修，但不能修复来源缺口。
2. 十份JSON的每条记录都只有`Prediction`数组，缺少预测原点时间、生成模型/配置、输入文本ID、检索可见范围和生成时刻。不能验证这些预测满足新协议的信息截止点。
3. `data/add.py`读取`integrated_predictions_stat/stage1_train_optimized`和`stage2_final_predictions`再合并输出；这些输入目录不在发布快照中。本次没有找到可用这些材料重建生成过程的完整链路。
4. 加载器用窗口起始索引取预测数组；改变共同任务的lookback后，必须重新建立“索引—原点—目标”映射，不能仅截取数组前H项就当成新协议的预测。原脚本的lookback与新比较计划不同。
5. 部分JSON数组长度并不等于文件名标称长度，原加载器会截断或插值。清单已保留全部长度分布；这本身不证明泄漏，但新比较不能隐去这些处理。

这些是当前复现与来源核验缺口，不是已经证明VoT论文存在标签泄漏。也不能用投稿包里的VoT代理替换后声称原生VoT已复现。

## 审稿回复现在可以怎样写

ECmL F2、LCo1 P2需要的首先是公平比较的证据。现在可以具体说明：已取得并固定作者实现，区分原生模型移植与投稿包代理；GPT4MTS运行兼容性已验证；原Table2仍不具备统一统计口径与来源记录，因此撤回其比较优越性声明。不能把本轮“接入成功”写成“比较完成”。

下一阶段优先运行Economy/H6的共同OT静态任务：新建单变量iAdapter骨干和作者GPT4MTS，统一原点/目标/有效掩码、时间划分、可见文本约束，记录各自真正消费的历史文本范围、编码方式和训练选择预算。原多变量分数、在线更新分数不能混入这一静态表；若文本历史长度不同，应明确作为模型输入设计差异披露，不能称输入完全相同。

VoT需要的是原点映射及生成记录，或在已授权资源下可重新运行并记录的生成链路；不是再补一个文件夹名字。此缺口不会阻止GPT4MTS与内部静态对照继续推进。

## 文件入口与复核

- `gpt4mts_smoke_validation.json`：实际GPU检查记录。
- `vot_prior_audit.json`：十份预测清单、路径错误见证与缺失材料。
- `gpt4mts_source.json`、`vot_source.json`、`gpt2_weights_source.json`：来源版本。
- `environment_spec.json`、`environment_validation.json`：依赖与运行见证。
- `CODE_REVIEW.md`：新上下文运行前审查（same-family provisional）。

所有本地路径相对`/home/wuwa/projects/research/iadapter`；解释器`/home/wuwa/miniconda3/envs/llm/bin/python`。只读重建VoT来源审计：`python -m iadapter.vot_source_audit`。再次执行GPT4MTS接入检查须指定新目录：`python -m iadapter.native_gpt4mts_smoke --output experiments/native_gpt4mts_smoke_repeat`；已存在目录会拒绝覆盖。

# 原生外部比较：接入阶段

当前先完成 A1 的来源与接口核验，尚不启动100单元矩阵。原始多变量结果不搬入新的OT单变量任务。

固定官方来源：GPT4MTS `c85e74b2fd7048f5ef601453e5bcb9f6073cd3b4`，VoT `f769621da9efde3e82a1975f478efdec83dcbd10`。源码下载到external，原文件保留；不将投稿包GPT4MTS_z/VoT_z冒称原生。官方仓库未列出明确LICENSE文件，下载内容供本地核查，暂不作为自有源码重新分发。

GPT2权重来源openai-community/gpt2，revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`，safetensors。本机已有完整BERT-base。没有安装新的环境、租用GPU或调用付费API。

GPT4MTS保留作者模型的六层GPT2、8步patch、stride4、多token历史prompt、RevIN和参数冻结规则（GPT2仅LN/wpe可训练）。只作两项明确兼容变换：将作者绝对cache/Hub加载改为固定本地目录；移除summary平均后的无参squeeze，保留batch=1与单patch维度。原始源码不改；在隔离命名空间编译变换后的AST，避免多个仓库的utils/models冲突。

接入smoke：同一实际模型权重，在eval模式验证batch=2时原forward与修复forward完全相同；验证原batch=1错误被修复；用真实Economy历史窗口和真实未来OT目标做一次有掩码的更新，确认只有规定参数可更新。这个更新只证明可训练，不是正式性能实验。文本沿用data.text_rows：每字段最多3个清洗后子句、当前与前一行合并、1200字符裁剪；之后使用完整12层BERT的CLS/512编码，而非原iAdapter的masked mean/64，不表示使用了完整原文。冻结BERT显式eval。sanity输出只报告形状、有限性、梯度路径、参数和显存，不将其误差写成基线结果。

后续共同任务：先Economy、H6、seed2023，前60%训练、60%–70%选择、70%–80%开发静态评估；输入只含OT历史，所有方法同原点/目标/掩码。iAdapter需要新单变量骨干，在训练流预热后冻结参数，评估不更新，也不从验证/测试标签刷新反馈。GPT4MTS保留原生架构和文本历史序列，按相同信息可用性代理选择文本；编码与预算差异披露。准确训练预算和缓存规范在该阶段运行前另行固定，不用本次sanity择优调参。

VoT先核验事件生成、检索与prior来源；仓库中若有预生成预测，未溯源前只读审计，不用于评分或训练。原生生成链路未就绪不阻止GPT4MTS与内部静态对照推进。

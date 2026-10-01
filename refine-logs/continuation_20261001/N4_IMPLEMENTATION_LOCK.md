# N4 原生模型接入检查锁定

本文件先于新接入代码写入。当前目标是完成 VoT native PatchTST 的依赖/实际形状/真实目标更新 smoke；不是以这个小检查替代完整基线比较。

依据作者固定提交 f769621da9efde3e82a1975f478efdec83dcbd10，直接载入 PatchTST_clip 定义，提取原 exp 文件的统计文本编码、contrastiveLoss_tr_sea 和 correction/frequency_decomposition 方法。避免导入与本任务无关的数据集/CLI，作者数学实现不修改。唯一实际权重路径替换为已有本地 GPT2，禁止联网下载。GPT2六层、768维、冻结eval；统计文本与192token预算保持源方法。

Smoke 固定 SocialGood，seed2025，L8/H6，与发布脚本当前激活域、seed和窗口对应。模型 d512/e2/heads8/ff2048/dropout.1 来自 run_clip.py 默认值。使用最初两个训练原点；对齐阶段需batch2（batch1的对比分类损失退化），同时独立检查batch1三个forward形状。数值/对齐/融合各做一次更新，使用原CSV真实OT目标及其原始有效掩码，原始窗口与source CSV先核对；它不是完整三阶段训练或协议一致的性能评价。

作者已发布 SocialGood length12先验仅用于检查AFF输入接入，显式采用源loader的起始窗口ID convention并截取前6值；记录原索引、完整长度和已截断状态。其真实生成原点/文本/检索 provenance仍未验证。不得把输入先验当ground truth，也不报告正式 MSE/MAE、seed收益、显著性或排名。没有使用0.5B枚举label伪造多步预测。

数值阶段lr1e-5（decomp_w1e-4）；对齐lr1e-4、排除patch_embedding/encoder；融合lr1e-3（decomp_w1e-4）是脚本枚举的一个固定值，只用于 smoke，不选最优lr。融合起始 decomp_w六项设0.5，与脚本和原exp赋值一致。既有作者batch相关全局均值/标准差保留，不要求batch2与逐样本预测一致；这是需披露的源模型行为。

独立检查确认：作者SocialGood为原900行的前892行，余下8行（2024-05到12月）OT缺失。Smoke核验这份前缀与缺失尾行，保留原900行数据、540行训练前缀/scaler/mask。原点7/8及其输入/真实目标位于共同前15行，发布loader窗口startID0/1；记录900/892及缺失尾行政策。发布prior共有893条，而作者892行/L8仅885个窗口；此数量差异保留，不作生成provenance证明。

GPU执行前需要 fresh code review；WSL当前无响应，代码先staged。端到端检查必须覆盖finite、shape、原CSVtarget/mask、GPT2冻结不变、数值head更新、对齐attention更新且excluded骨干不变、融合decomp_w更新，保存失败。结果仅能证明这个接入检查通过；完整训练、checkpoint selection、原点可用性与公平评分必须后续完成。

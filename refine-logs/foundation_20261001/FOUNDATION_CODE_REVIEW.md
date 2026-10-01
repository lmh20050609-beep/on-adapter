# C0 TimesFM3 预检代码审查

结论：**NON-BLOCKING**。未发现固定作者 API、参数数目、历史窗口或目标对齐方面的阻断错误。该结论仅适用于四个校准 origin 的集成预检，属于 same-family provisional 静态审查；不是运行通过证明、现代基线优劣比较或总目标完成。

审查日期：2026-10-01。规范源码位置：`/home/wuwa/projects/research/iadapter`（WSL Ubuntu）。项目 HEAD 在审查时为 `a23cc8ea9c7b4da37fa01ae0648b76ddc505ce4e`。未启动模型、训练或下载，未修改科研代码。

## 具体问题

1. **NON-BLOCKING：启动前提交约束不能识别未跟踪的新脚本。** `foundation_preflight.py:30` 使用 `revision()['dirty']`，而 `util.py:21` 明确传入 `--untracked-files=no`。审查时 `git status --short` 显示 `iadapter/foundation_preflight.py` 和 `refine-logs/foundation_20261001/` 都未跟踪，因此该检查此时可能通过，却把不包含预检代码的旧 HEAD 写进运行来源。按已有计划，在启动前提交这两个文件并核对 tracked/clean 即可消除此问题，无须新增哈希体系或改动公共版本函数。

2. **NON-BLOCKING：batch2 返回的完整分位数尚未执行与 singleton 相同的检查。** `foundation_preflight.py:77–82` 只比较 `forecast`，没有核对每个 batch 输出的 `quantiles.shape == (6,9)`、全部有限、以及 `forecast == quantiles[:,4]`。若分位数尾部出现非有限值而中位数有限，现有比较仍可能通过；这比计划的“非有限输出即停止”覆盖范围窄。建议复用第64–68行检查，并确认 batch 返回两项。没有观察到实际非有限输出；这是有限的验证覆盖缺口。

## 已核对的证据

- **API 与 batch 配置正确。** `timesfm3/__init__.py` 正确导出 `TimesFM3Forecaster`、`ModelConfig`。作者 `timesfm3_forecaster.py:38–104` 的 forecaster config 为可变 dataclass；`predict_batch:643–650` 每次读取当前 `per_core_batch_size`。因此把 `model.config` 从1改为2会形成真实二项 batch。目录 checkpoint 通过本地 `PyTorchModelHubMixin.from_pretrained(..., local_files_only=True)` 加载；未通过模型仓库获取自定义模型代码。
- **330,710,976 是参数个数。** 由 `model.py`、`dense.py`、`transformer.py` 的默认结构独立计算：输入残差块2,129,920；20层327,843,200；输出头737,856；总和330,710,976。RoPE timescale 是非持久 buffer，不纳入参数数目。公式使用1280维、16头、20层、输入patch32、输出patch64、9分位数；没有把权重文件字节数或冻结后的可训练参数数目误作总参数数目。加载后 `sum(p.numel())` 的实际断言仍须在 C0 中通过。
- **点预测为官方中位数。** 作者 `timesfm3_forecaster.py:721–758` 先按分位数排序，再返回索引4；预检的 `(6,)` 点预测和 `(6,9)` 分位数形状符合一维输入语义。`TimesFM3Evaluator` 的官方 benchmark 默认还启用对称平均和非负截断；本脚本显式使用 forecaster 默认关闭这两者，适合所声明的 integration scope，不能把该运行称为 evaluator 默认 benchmark。
- **上下文和标签边界正确。** `historical_context` 取闭区间 `[t-L+1,t]` 的OT；未来标签为 `[t+1,t+6]`。两种 covariates 均未传入。上游 `data.py` 使用前向填充和初始60%统计量归一化；`full_ot` 只切片原来的OT目标及原 valid mask，并未用预测替代标签。当前CSV行数与纯算术边界核验结果：Energy 1479行，origins1034、1035，校准标签区间[1035,1109)；SocialGood 900行，origins629、630，区间[630,675)。四个窗口长度和六步标签均满足边界。
- **冻结路径与有限检查基本正确。** 作者解码运行在 `torch.inference_mode()`，wrapper 在加载后冻结参数并设为eval，没有优化器或训练调用；结束时检查参数version、requires_grad和grad。检查过的作者前向代码未出现权重写入。singleton 对全部输出检查有限值，batch point 的allclose会拒绝非有限中位数。fp32输入、官方F32权重和默认fp32模型初始化一致；此处未实测加载后的dtype或显存。
- **输出和范围表达适当。** 新运行目录要求不存在，失败写入failed状态，保存原target/mask、origin、数据集和上下文边界；masked metric只作为诊断。预训练重叠未排除、end_date代理时间线和镜像运输来源均披露。镜像digest只作为运输校验，不解释为独立取得的官方权重哈希。审查时权重下载尚在进行、目录仅有part文件，最终receipt/config/权重完整性由既定下载流程落实；本次未认证这些文件。

静态验证仅运行了Python AST解析、源码结构的整数参数计算、CSV行数与origin边界计算，均通过。实际加载、CUDA显存峰值、有限输出、batch一致性和冻结断言留给已限定的C0运行。


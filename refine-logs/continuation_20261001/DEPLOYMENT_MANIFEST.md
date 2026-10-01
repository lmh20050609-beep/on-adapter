# 恢复后的应用顺序

本目录是待应用补丁与可读证据包，本提交同步远端代码与材料，WSL生产工作区仍待应用。生产根目录应为 `/home/wuwa/projects/research/iadapter`，不能创建第二个同名训练目录或从 Windows 快照启动替代实验。

先检查 Git 状态与原CSV；只复制以下源文件：

- source_patch/iadapter/text_order.py → iadapter/text_order.py
- source_patch/iadapter/recent_text.py → iadapter/recent_text.py
- source_patch/iadapter/extension.py → iadapter/extension.py
- source_patch/iadapter/native_vot.py → iadapter/native_vot.py
- source_patch/iadapter/native_vot_smoke.py → iadapter/native_vot_smoke.py
- source_patch/tests/test_text_order.py → tests/test_text_order.py

source_patch/iadapter/__init__.py 只用于Windows纯函数测试，不复制覆盖生产文件。原目录、缓存、旧CSV及旧预测记录不删除。

必要检查在原 llm 环境中执行：标准库 unittest 全套、模块编译、数据核对、原测试的成熟回放及未来扰动检查。检查通过后提交所有预期实验源代码/文档，并记录提交号；队列进行期间不改源码。

```bash
cd /home/wuwa/projects/research/iadapter
/home/wuwa/miniconda3/envs/llm/bin/python -m unittest discover -s tests
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.recent_text --stage prepare
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.recent_text --stage campaign --limit 1
# 原CSV检查、旧预测一致性及第一个组完整通过后
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.recent_text --stage campaign
```

N1先得到结果，再启动N3；N3候选不随N1成绩更改。

```bash
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.extension --stage prepare
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.extension --stage generate
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.extension --stage campaign --limit 1
# 完成Agriculture/2023 sanity、记录分类质量与全部控制结果后
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.extension --stage campaign
```

N3生成预算为新Qwen调用累计600秒，含加载和失败，不含另记的BERT/PCA成本。generation_status显存字段是加载完成后的生成阶段峰值（包括常驻权重），不是模型加载瞬态峰值。输出token字段来自实际生成。

最后分析saved predictions，不把失败或不完整group计入成功；将新表、配对记录和结果边界合入本目录替换稿，再同步已授权的GitHub仓库。VoT接入按 VOT_SOURCE_AND_INTEGRATION.md 分阶段执行，不能把枚举分类器直接当成多步数值生成器。

N4原生PatchTST接入代码及三阶段smoke已通过独立代码检查（N4_CODE_REVIEW.md），仍需实际GPU运行。原始external/vot和本地GPT2文件必须存在；执行一次完整smoke，失败先读status.json，不原样重跑。

```bash
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.native_vot_smoke
```

smoke不输出正式基线分数。它使用共同早期窗口和原900行CSV真实目标；作者892行仅用于原始数据前缀核对，缺失尾行不从本地删除。正式VoT比较仍按来源和原点映射要求继续，不将smoke通过冒充完整复现。

本文件列出待执行命令，不表示命令已运行。WSL重启授权尚未收到；不执行任何结束用户进程的动作。

远端取得说明：恢复后先核对Git状态、未提交/未跟踪文件与远端差异，再取得本提交。以下复制清单适用于Windows补丁恢复；从Git取得的相同文件无需重复覆盖。保留未知本地修改。

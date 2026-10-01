# Continuation evidence — current status

WSL/CUDA recovered; 65 tests pass. N1/N3 complete and native VoT three-stage integration passes. See [current tracker](CONTINUATION_TRACKER.md), [runtime audit](RUNTIME_STATUS.json), [updated paper text](MANUSCRIPT_INTEGRATED.md) and [new result reports](live_results/). Formal VoT prior provenance and full matched comparison remain open.

The initial source review and deployment material is retained below as history; any pending-GPU statement there is superseded by the current tracker.

# iAdapter 后续修订：代码与证据同步

本提交增加已完成静态检查的近期文本、追加域和原生VoT接入代码，以及论文/回复整合材料。来源基础提交为c62791c9e2700de322ac2a3739ac6a2fd9a31630，既有结果和原方法保留。

**当前没有N1/N3的新GPU结果，nativeVoT smoke也尚未运行。** 独立静态审查PASS不等于端到端实验通过。文本发布时间仍是end_date代理；VoT发布prior的生成来源仍未验证。

## 新模块

- `iadapter/text_order.py`、`iadapter/recent_text.py`：同一保留字符预算，交换两条时间记录先后；固定旧预测诊断与新静态/在线评价。
- `iadapter/extension.py`：锁定原E003 LLM5候选，在Agriculture/Health运行五seed对照，保留原CSV目标/掩码及所有配对差值。
- `iadapter/native_vot.py`、`iadapter/native_vot_smoke.py`：作者PatchTST与原数学helpers接入，三阶段真实目标更新检查；只做接入，不输出正式基线分数。
- `tests/test_text_order.py`：已通过的标准库字符/正文保留检查。完整Torch测试与真实GPU运行仍待原环境恢复。

N4检查保留了实际发现：作者SocialGood为原900行的前892行，余下8行OT缺失。原900行/scaler/mask不删除；早期共同窗口/真实目标核对通过。published prior893条不证明原生成原点正确，不能伪装ground truth。

## 证据与论文材料

本目录包含N1/N3/N4静态审查、Agriculture/Health数值身份核验、VoT来源差异、E003表格及配对表、论文整合稿和原review九点回复。E003展示数字来自此前已验证结果，不是本轮新增实验。`build_manuscript.py`可从仓库中的既有结果再生成整合表和草稿；不得填写待跑阶段的虚构数字。

## 下一步

唯一生产目录仍为WSL的`/home/wuwa/projects/research/iadapter`。WSL命令/文件共享目前不可用，因此本提交是GitHub代码和材料同步，生产工作区尚待部署。恢复后先核对工作区现有/未提交/未跟踪文件与远端差异，保留已有文件，再取得本提交；不覆盖未知修改。

随后按`DEPLOYMENT_MANIFEST.md`在原llm环境完成必要测试和sanity，再执行N1、锁定N3及VoT接入/正式比较，最后更新论文、回复和结果边界。原Table2来源、真实publication-time及nativeVoT可信生成链仍是未解决项。这里不声明完整复现、普遍SOTA或论文已完成。

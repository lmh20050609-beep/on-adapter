# 修改前完整备份

创建时间：2026-09-29 00:38:34（WSL 本地时间）。

归档：`/home/wuwa/backups/iadapter/20260929-003818/original-and-revision-before-review-changes.tar`

大小：4,154,245,120 字节。内容包含整个原项目
`adaptz_paper_work/ADAPT-Z-main` 与当时的整个 `iadapter_revision`，包括数据、
权重、代码、Git 历史、配置、190 次运行及报告。

在源目录未修改时执行 tar --compare，退出码 0，无差异输出。原始机器可读记录位于
同目录的 `backup-verification.json`。

恢复到一个新目录，避免覆盖当前工作：

```bash
mkdir -p /home/wuwa/restore_iadapter_review
tar -xf /home/wuwa/backups/iadapter/20260929-003818/original-and-revision-before-review-changes.tar -C /home/wuwa/restore_iadapter_review
```

本次备份完成后才修改修订仓库。详细执行计划另存于 Windows 工作区
`output/iadapter-review-completion-plan.md`。

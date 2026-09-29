# iAdapter 目录与补充材料

主目录：`/home/wuwa/projects/research/iadapter`。
Windows 文件管理器入口：`\\wsl.localhost\Ubuntu\home\wuwa\projects\research\iadapter`。

```text
iadapter/
├── iadapter/                  已验证的修订 Python 包
├── data -> submission/code/data  当前数据入口（项目内部链接）
├── tests/                     协议及结果核查测试
├── runs/                      190 次正式运行与原始预测
├── checkpoints/               35 个共享骨干
├── weights/                   现有本地 BERT 权重
├── cache/                     已编码文本缓存
├── reports/                   新主表、消融、效率、图和论文替换文本
├── docs/
│   └── review_routes/         五版方案和实验跟踪表
└── submission/
    ├── archives/              用户提供的原始 ZIP
    ├── documents/             技术补充、复现清单 PDF
    ├── code/                  ZIP 中 iAdapter/ 的完整内容
    │   ├── adapt_z_multimodal.py
    │   ├── models/            GPT4MTS_z.py、VoT_z.py 等
    │   ├── data/              完整投稿数据，40 个 CSV
    │   └── scripts/
    ├── audit/                 补充包静态核查与差异证据
    └── IMPORT_MANIFEST.json   复制校验和目录迁移记录
```

按用户要求，旧 `adaptz_paper_work` 项目目录与 `iadapter_revision` 兼容链接已清理，
研究目录只保留当前 `iadapter` 项目。原始完整归档仍保留在项目外的备份目录。

DATA_ROOT 现在使用本项目的 `data`，指向已导入的 `submission/code/data`。
旧数据的 39 个文件与对应投稿文件逐字节一致；投稿包额外提供 new_metr.csv。
迁移没有改变模型配置、划分或预测。merged/raw Climate 依然是不同材料，不能混用。

历史运行 JSON 中原始来源路径不改写。`docs/PATH_MIGRATION.json` 保存 39 个数据文件
和 35 个 checkpoint 的旧路径、当前相对路径及 SHA-256；审计和效率工具通过
`iadapter/evidence_paths.py` 校验内容后解析旧记录。这样不再依赖旧目录或外部兼容链接。
旧聊天中的绝对文件链接可能已失效，应使用本页的新主目录。

完整 BERT 文件已在 `weights/bert-base-uncased`；预处理直接复用这些文件，不再要求
旧目录的压缩包。需要恢复缺失权重时可使用保留的完整备份，或提供
`weights/bert-base-uncased.zip`。

常用入口：

```bash
cd /home/wuwa/projects/research/iadapter
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.campaign --status
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.campaign --audit-completed
```

投稿代码与可信修订代码分别保留；本次没有将投稿包中的旧执行器覆盖到修订实现上，
也没有启动训练。完整包可用于后续逐模块迁移，基线身份与协议差异见
`submission/audit/REPORT.md`。

源码、配置、核查文档纳入 Git；原始 ZIP、PDF、数据与权重保存在本地且加入忽略规则。
这些忽略规则不删除文件。原完整备份仍位于 `/home/wuwa/backups/iadapter/20260929-003818/`。

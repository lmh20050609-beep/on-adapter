# iAdapter 目录与补充材料

主目录：`/home/wuwa/projects/research/iadapter`。
Windows 文件管理器入口：`\\wsl.localhost\Ubuntu\home\wuwa\projects\research\iadapter`。

```text
iadapter/
├── iadapter/                  已验证的修订 Python 包
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

原 `iadapter_revision` 目录已改名。旧路径现在只是指向主目录的符号链接，
用于兼容历史结果中的绝对 checkpoint 路径及已有文件链接，不是第二份代码副本。

原始 `adaptz_paper_work/ADAPT-Z-main` 保留。现有修订配置的 DATA_ROOT 仍指向该目录内的
历史数据源，保留 190 次实验的来源一致性；本次同步没有改训练配置、数据划分或预测。
投稿数据的独立副本位于 `submission/code/data`。两者不应仅因目录名称相同就混用，
尤其是 merged 与 raw Climate 数据存在字段和任务差异。

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

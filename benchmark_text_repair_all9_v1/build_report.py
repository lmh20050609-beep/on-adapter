"""Assemble the nine-domain repair report from the frozen artifacts."""
import collections
import csv
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
REPORT = ROOT / 'reports/benchmark_text_repair_all9_v1_20261008'
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood',
           'Security', 'Economy', 'Climate', 'Agriculture']


def load(name):
    return json.loads((REPORT / name).read_text(encoding='utf-8'))


def main():
    inventory = load('text_inventory_all9.json')
    coverage = {row['domain']: row for row in load('coverage_all9.json')}
    invariance = {row['domain']: row for row in load('data_field_invariance_all9.json')}
    reproduction = load('frozen_v1_reproduction.json')
    summary = json.loads((BASE / 'source_evidence/decision_summary.json').read_text(encoding='utf-8'))
    effects = {row['domain']: row for row in load('token_effects.json')}

    per_split = collections.defaultdict(collections.Counter)
    for item in inventory:
        per_split[(item['domain'], item['split'])]['items'] += 1
        per_split[(item['domain'], item['split'])][item['classification']] += 1
        per_split[(item['domain'], item['split'])]['masked'] += item['action'] == 'mask_item'

    lines = []
    lines.append('# 九域最小文本修复（V1）数据与来源报告\n')
    lines.append('2026-10-08。目标：在保持数值数据、划分、目标、mask 与反馈规则不变的前提下，'
                 '把已确认的文本时间错配最小修复扩展到全部 9 个基准域，并冻结完整实验协议。\n')
    lines.append('**V1 只屏蔽已确认晚于描述时点的独立条目**；来源分级为 C/D 的条目一律保留并披露，'
                 '筛查未命中不等于已认证。本报告不读取任何模型成绩。\n')

    lines.append('## 1. 模型实际读取的文本字段\n')
    lines.append('| 方法 | 读 `fact` | 输入路径 |\n|---|---|---|')
    lines.append('| On-Adapter | 是 | `online.ResidualOnline` 经 `data.prepared_text.pooled(origin)`，'
                 '可视行 (origin-1, origin) 拼接，64 token mean pooling |')
    lines.append('| GPT4MTS | 是 | `multimodal.GPTForecast` 经 `text.gpt(origin, lookback)`，512 token cls |')
    lines.append('| MM-TSFlib | 是 | `multimodal.MMD` 经 `text.tokens(origin)`，GPT2 6 层，256 token |')
    lines.append('| iTransformer / ADAPT-Z / DSOF / Proceed | 否 | 代码路径不访问 `data.texts` / `prepared_text` |')
    lines.append('\n`preds`、`Final_Output`、`Final_Search_*` 任何方法都不读。\n')

    lines.append('## 2. 来源分级与屏蔽\n')
    lines.append('规则文件 `source_evidence/decision_rules_all9.json`，SHA256 `%s`；'
                 '继承 4 域冻结规则 %d 条（SHA256 `%s`），新增 5 域独立复核 %d 条。\n'
                 % (summary['sha256'], summary['inherited_rules'],
                    summary['inherited_sha256'], summary['new_rules']))
    lines.append('| 域 | 独立条目 | A | B（屏蔽） | C（保留披露） | D（未核验保留） | 屏蔽条目占该域 | 被改动行数 |\n'
                 '|---|---|---|---|---|---|---|---|')
    for domain in DOMAINS:
        row = coverage[domain]
        lines.append('| %s | %d | %d | %d | %d | %d | %.3f%% | %d |'
                     % (domain, row['items'], row['A'], row['B'], row['C'], row['D'],
                        100 * row['masked_items'] / row['items'], row['modified_rows']))
    total_items = sum(coverage[d]['items'] for d in DOMAINS)
    total_masked = sum(coverage[d]['masked_items'] for d in DOMAINS)
    lines.append('| **合计** | %d | %d | %d | %d | %d | %.3f%% | %d |'
                 % (total_items, sum(coverage[d]['A'] for d in DOMAINS),
                    sum(coverage[d]['B'] for d in DOMAINS), sum(coverage[d]['C'] for d in DOMAINS),
                    sum(coverage[d]['D'] for d in DOMAINS),
                    100 * total_masked / total_items,
                    sum(coverage[d]['modified_rows'] for d in DOMAINS)))
    lines.append('\n屏蔽的是条目出现次数；同一原始文本在多行重复出现时按行分别屏蔽。'
                 '未删除任何数值行、预测原点或评分目标。\n')

    lines.append('## 3. 各域各划分覆盖率\n')
    lines.append('| 域 | 划分 | 条目 | A | B | C | D | 屏蔽 | 条目保留率 |\n|---|---|---|---|---|---|---|---|---|')
    for domain in DOMAINS:
        for split in ['train', 'val', 'test']:
            cell = per_split[(domain, split)]
            if not cell['items']:
                continue
            lines.append('| %s | %s | %d | %d | %d | %d | %d | %d | %.4f |'
                         % (domain, split, cell['items'], cell['A'], cell['B'], cell['C'], cell['D'],
                            cell['masked'], 1 - cell['masked'] / cell['items']))

    lines.append('\n## 4. 实际 token 变化\n')
    lines.append('按每种方法冻结的 tokenizer、token 上限与文本映射规则，比较 V0 与 V1 下每行实际模型输入。\n')
    lines.append('| 域 | 方法 | 含屏蔽条目的行 | 其中屏蔽条目仍落在 token 窗口内 | 行 token 变化 | test 原点 | test 有 token 变化的原点 |\n'
                 '|---|---|---|---|---|---|---|')
    for domain in DOMAINS:
        for method in ['On-Adapter', 'GPT4MTS', 'MM-TSFlib']:
            m = effects[domain]['methods'][method]
            lines.append('| %s | %s | %d | %d | %d | %d | %d |'
                         % (domain, method, m['rows_whose_raw_cell_contains_a_masked_item'],
                            m['of_those_rows_the_masked_item_survives_into_the_token_window'],
                            m['rows_with_changed_tokens'],
                            m['per_split']['test']['origins'],
                            m['per_split']['test']['origins_with_changed_tokens']))
    lines.append('\nOn-Adapter 的 64-token 窗口在多数域并未包含被屏蔽的条目本身，'
                 'token 变化主要来自 `clean_fact` 尾部 1200 字符窗口因删除而前移；'
                 'MM-TSFlib 的 256-token 窗口则会保留其中相当一部分。\n')

    lines.append('## 5. 身份与复现\n')
    lines.append('| 域 | V1 文件 SHA256 | 非 fact 列全等 |\n|---|---|---|')
    for domain in DOMAINS:
        row = invariance[domain]
        lines.append('| %s | `%s` | %s |' % (domain, row['v1_sha256'][:32] + '…',
                                            row['only_fact_column_changed']))
    lines.append('\nAgriculture/Economy/Security 三域 V1 与三域先导冻结文件逐字节一致：%s。\n'
                 % all(v['identical'] for v in reproduction.values()))

    lines.append('## 6. 必须保留的限制\n')
    lines.append('- 来源复核只覆盖“正文中出现晚于描述时点年份”的候选条目；未命中不代表无问题，'
                 '也未认证为 A。')
    lines.append('- C/D 条目全部保留在实际数据中，仅作披露；V1 不等于“所有条目均已认证”。')
    lines.append('- 数值仍使用原 `end_date` 可用性代理与回顾性目标 mask 原点网格；'
                 '不证明真实发布时间与历史修订版本意义上的完整因果性。')
    lines.append('- 来源决定不读取模型成绩，不因结果调整屏蔽规则。')
    lines.append('- 本阶段不扩展到其他 seed 或 SocialGood 主表；SocialGood 的 V1 已包含在本 9 域数据中，'
                 '但不单列额外先导。')
    (REPORT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('report written', REPORT / 'REPORT.md')


if __name__ == '__main__':
    main()

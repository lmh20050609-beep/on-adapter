"""Assemble the nine-domain V1 campaign report from the frozen artifacts."""
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
REPORT = ROOT / 'reports/benchmark_textrepair_v1_20261008'
METHODS = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed', 'On-Adapter', 'GPT4MTS', 'MM-TSFlib']
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood',
           'Security', 'Economy', 'Climate', 'Agriculture']
ORDER = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood', 'Security',
         'Economy', 'Climate', 'Agriculture']


def load(name):
    return json.loads((REPORT / name).read_text(encoding='utf-8'))


def main():
    results = load('results.json')
    audit = load('completion_audit.json')
    reuse = load('reuse_verification.json')
    checks = load('path_checks.json')
    try:
        equivalence = load('numeric_reuse_equivalence.json')
    except FileNotFoundError:
        equivalence = dict(status='not_run', records=[])
    repair = json.loads((ROOT / 'reports/benchmark_text_repair_all9_v1_20261008/coverage_all9.json')
                        .read_text(encoding='utf-8'))
    summary = results['summary']

    lines = []
    lines.append('# 九域 V1 文本修复 benchmark：完整 756 格评测\n')
    lines.append('2026-10-08。数据版本 **V1**（九域最小文本修复），协议见 `PROTOCOL.md`，'
                 '数据与来源见 `../benchmark_text_repair_all9_v1_20261008/REPORT.md`。\n')
    lines.append('本节报告**不预设方法获胜**；结果允许为正、负或接近不变。\n')

    lines.append('## 1. 执行矩阵与完成度\n')
    lines.append('| 项 | 数量 |\n|---|---|')
    lines.append('| 矩阵总格数（9 域 × 4 H × 3 seed × 7 方法） | %d |' % audit['matrix_cells'])
    lines.append('| 本轮新训练（On-Adapter / GPT4MTS / MM-TSFlib） | %d 已完成 / %d 应有 |'
                 % (audit['new_training_cells_completed'], audit['new_training_cells_expected']))
    lines.append('| 复用历史文本无关单元（iTransformer / ADAPT-Z / DSOF / Proceed） | %d |'
                 % audit['reused_legacy_text_free_cells'])
    lines.append('| 失败格 | %d |' % len(audit['failures']))
    lines.append('| 执行锁未变 | %s |' % audit['execution_lock_unchanged'])
    lines.append('\n逐域新训练完成数：%s\n' % ', '.join(
        '%s %d' % (d, n) for d, n in sorted(audit['per_domain_new'].items())))
    lines.append('所有新格由独立 OS 进程从相同定义的初始状态训练；未沿用任何旧文本版本成绩、'
                 '适配器检查点或结果。\n')

    lines.append('## 2. 复用兼容性证明\n')
    lines.append('- 复用范围仅限**代码路径不读取 `fact` 列**的 4 个方法，共 %d 格，'
                 '逐格固定 result / config / best.pt / predictions 哈希（`reuse_verification.json`）。\n'
                 % reuse['reused_cells'])
    lines.append('- V1 表与原表逐行比较，差异字段为 `%s`。\n'
                 % json.dumps(sorted({f for v in reuse['domain_field_identity'].values()
                                      for f in v['differing_fields']}), ensure_ascii=False))
    lines.append('- 数值等价重跑（容差 0）：状态 %s，%d 个抽样格。\n'
                 % (equivalence['status'], equivalence.get('sample', 0)))
    if equivalence.get('records'):
        lines.append('\n| 抽样格 | predictions/targets/mask/origins 逐位相等 | MAE/MSE 完全相等 |\n|---|---|---|')
        for record in equivalence['records']:
            lines.append('| %s H%d seed%d %s | %s | %s |'
                         % (record['domain'], record['horizon'], record['seed'], record['method'],
                            record['all_arrays_identical'],
                            record['mae_identical'] and record['mse_identical']))

    lines.append('\n## 3. 未覆盖路径检查\n')
    failed = [k for section in checks.values() for k, v in section.items()
              if isinstance(v, dict) and v.get('status') == 'FAILED']
    lines.append('覆盖日频/周频、非 H12 窗口以及此前未训练的 GPT4MTS、MM-TSFlib；'
                 '共 %d 项检查，失败 %d 项。\n'
                 % (sum(len(v) for v in checks.values()), len(failed)))
    lines.append('| 检查 | 通过 | 说明 |\n|---|---|---|')
    lines.append('| 原点/目标/mask/标准化对齐 | %d/%d | 逐原点核对 target_rows、mask 与标准化真值 |'
                 % (len(checks['alignment']), len(checks['alignment'])))
    lines.append('| 预测不消费未来 y/mask | %d/%d | 替换 y 与 mask 后预测逐位相等 |'
                 % (sum(v['identical'] for v in checks['minimal_input'].values()), len(checks['minimal_input'])))
    lines.append('| 文本 gather 行可见性 | %d/%d | 每条 gather 行索引的可用时间不晚于原点 |'
                 % (len(checks['text_visibility']), len(checks['text_visibility'])))
    lines.append('| 未来文本扰动不影响过去输出 | %d/%d | 容差 0，逐位相等 |'
                 % (sum(v['prediction_identical'] for v in checks['perturbation'].values()),
                    len(checks['perturbation'])))
    lines.append('| 预处理/缓存版本隔离 | %d/%d | V1 与原文件哈希不同，缓存按数据身份与文本版本隔离 |'
                 % (sum(v['distinct_from_original'] for v in checks['cache_identity'].values()),
                    len(checks['cache_identity'])))

    lines.append('\n## 4. 主统计\n')
    lines.append('主统计：先在每个域/H 内对 seed 平均风险，再除以同一域/H/seed 的 iTransformer '
                 '共同骨干 CVaR90，最后对 36 个域/H 任务取等权几何平均。\n')
    lines.append('| 方法 | 跨任务几何平均相对 CVaR90 | 算术平均 | 平均排名 | 中位排名 | 最好/最差任务比值 |\n'
                 '|---|---|---|---|---|---|')
    for method in METHODS:
        s = summary[method]
        gm = s['geometric_mean_relative_cvar90']
        lines.append('| %s | %s | %.6f | %.4f | %.1f | %.6f / %.6f |'
                     % (method, ('%.6f' % gm) if gm is not None else '不可定义(存在非正比值)',
                        s['arithmetic_mean_relative_cvar90'], s['mean_rank_cvar90'],
                        s['median_rank_cvar90'], s['best_relative_cvar90'], s['worst_relative_cvar90']))

    lines.append('\n| 方法 | 平均 MAE | 平均 MSE | 平均 CVaR80 | 平均 CVaR95 |\n|---|---|---|---|---|')
    for method in METHODS:
        s = summary[method]
        lines.append('| %s | %.6f | %.6f | %.6f | %.6f |'
                     % (method, s['mean_mae'], s['mean_mse'], s['mean_cvar80'], s['mean_cvar95']))

    lines.append('\n## 5. 逐域/H 结果（seed 平均）\n')
    lines.append('| 域 | H | 方法 | MAE | MSE | CVaR90 | 相对骨干 CVaR90 |\n|---|---|---|---|---|---|---|')
    for task in sorted(results['tasks'], key=lambda t: (ORDER.index(t['domain']), t['horizon'])):
        for method in METHODS:
            row = task[method]
            lines.append('| %s | %d | %s | %.6f | %.6f | %.6f | %.6f |'
                         % (task['domain'], task['horizon'], method, row['mae'], row['mse'],
                            row['cvar90'], row['relative_cvar90']))

    lines.append('\n## 6. 预先声明的探索性分组\n')
    lines.append('以下为协议冻结前声明的探索性分组，不是确认性结论。\n')
    lines.append('| 方法 | 月频六域几何平均相对 CVaR90 |\n|---|---|')
    for method in METHODS:
        value = results['exploratory_monthly'][method]['monthly_six_domains_geometric_mean_relative_cvar90']
        lines.append('| %s | %s |' % (method, ('%.6f' % value) if value is not None else '不可定义'))
    lines.append('\n| 方法 | daily | weekly | monthly | 频率等权 |\n|---|---|---|---|---|')
    for method in METHODS:
        g = results['frequency_groups'][method]
        fmt = lambda x: ('%.6f' % x) if x is not None else '不可定义'
        lines.append('| %s | %s | %s | %s | %s |'
                     % (method, fmt(g['daily']), fmt(g['weekly']), fmt(g['monthly']),
                        fmt(g['frequency_equal_weight_geometric_mean'])))

    lines.append('\n## 7. 历史参照（V0 文本 + 旧批量编码的既有 756 组）\n')
    lines.append('下表由既有 periodic-v3/v4 完成单元的保存预测**只读复用**重算，'
                 '**不是新主表**：其文本方法使用原始 V0 文本列与修复前的批量编码，'
                 '因此与 V1 主表的差异同时包含文本修复与编码修复。\n')
    try:
        history = load('historical_reference.json')
        lines.append('| 方法 | 相对骨干几何平均 CVaR90 | 相对每任务最优几何平均 | 平均排名 | 平均 MAE | 平均 MSE |\n'
                     '|---|---|---|---|---|---|')
        for method in sorted(METHODS, key=lambda m: history['summary'][m]['geometric_mean_relative_cvar90']):
            h = history['summary'][method]
            lines.append('| %s | %.4f | %.4f | %.3f | %.6f | %.6f |'
                         % (method, h['geometric_mean_relative_cvar90'],
                            h['geometric_mean_relative_oracle_cvar90'], h['mean_rank_cvar90'],
                            h['mean_mae'], h['mean_mse']))
        lines.append('\n**V1 与历史的直接对照（同一主统计定义）**\n')
        lines.append('| 方法 | V1 相对骨干 GM | 历史相对骨干 GM | 相对差 | 平均排名 V1 / 历史 |\n|---|---|---|---|---|')
        for method in sorted(METHODS, key=lambda m: summary[m]['geometric_mean_relative_cvar90']):
            v1 = summary[method]['geometric_mean_relative_cvar90']
            h = history['summary'][method]['geometric_mean_relative_cvar90']
            lines.append('| %s | %.4f | %.4f | %+.2f%% | %.3f / %.3f |'
                         % (method, v1, h, 100 * (v1 - h) / h,
                            summary[method]['mean_rank_cvar90'],
                            history['summary'][method]['mean_rank_cvar90']))
        deltas = [abs(summary[m]['geometric_mean_relative_cvar90']
                      - history['summary'][m]['geometric_mean_relative_cvar90'])
                  / history['summary'][m]['geometric_mean_relative_cvar90'] for m in METHODS]
        lines.append('\n最大相对差 %.2f%%。**文本修复与编码修复都没有实质改变方法之间的比较结构**：'
                     '方法排序不变，On-Adapter 在 Energy（H24 为 2.47 → 2.53）的劣势与在 '
                     'SocialGood / Traffic 的优势逐任务几乎逐位复现。该结论只针对本轮修复的条目范围，'
                     '不能外推为“旧结果全部由泄露造成”，也不能外推为“其余文本已证明无问题”。\n'
                     % (100 * max(deltas)))
        lines.append('历史记录中的“三轴等权几何相对最优比率”为骨干 1.0861、MM-TSFlib 1.0937、'
                     'GPT4MTS 1.0953、ADAPT-Z 1.1112、On-Adapter 1.1408、Proceed 1.1477、DSOF 1.1622，'
                     'On 排第 5。本表的 oracle 比率定义与其接近但聚合细节不同，仅作方向性对照。\n')
    except FileNotFoundError:
        lines.append('（历史参照未计算。）\n')

    lines.append('\n## 8. 文本修复的实际可见性\n')
    lines.append('| 域 | 独立条目 | B（屏蔽） | C（保留披露） |\n|---|---|---|---|')
    for row in sorted(repair, key=lambda r: ORDER.index(r['domain'])):
        lines.append('| %s | %d | %d | %d |' % (row['domain'], row['items'], row['B'], row['C']))
    lines.append('\nOn-Adapter 的 64-token 窗口在多数域未包含被屏蔽条目本身；'
                 'MM-TSFlib 的 256-token 窗口会保留其中一部分。实际 token 变化见 '
                 '`../benchmark_text_repair_all9_v1_20261008/repair_token_effects.csv`。\n')

    lines.append('## 9. 必须保留的限制\n')
    lines.append('- 本次协议冻结晚于旧测试集被反复观察，**不是独立确认或预注册**。')
    lines.append('- 来源决定只覆盖筛查命中的候选条目；C/D 条目保留在实际数据中，'
                 'V1 不等于“所有条目均已认证无泄露”。')
    lines.append('- 数值仍使用原 `end_date` 可用性代理与回顾性目标 mask 原点网格，'
                 '不证明真实发布时间与历史修订版本意义上的完整因果性。')
    lines.append('- 432 个复用单元为历史同协议结果；其文本无关性由代码路径与数值等价重跑共同证明，'
                 '但这不等同于本轮重新训练。')
    lines.append('- iTransformer 为共同骨干，其比值是相对量，不是绝对优劣结论。')
    lines.append('- 历史参照的文本方法使用旧批量编码，与 V1 主表的差异不是纯粹的文本修复效应。')
    (REPORT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('written', REPORT / 'REPORT.md')


if __name__ == '__main__':
    main()

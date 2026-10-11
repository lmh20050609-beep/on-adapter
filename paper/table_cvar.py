"""LaTeX for the tail-risk main table: per-dataset CVaR90 (mean over four horizons and seeds 2025-2027) for the ten main-table
methods, plus summary rows from results/new_baselines/results_main_cosa.json. Marks decided on printed values; ties share the mark.
Ratio-to-backbone rows (S, #Better, #Harm, Worst) are not marked for the backbone itself, which is their reference."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results/paper'
P = json.loads((R / 'table_domain.json').read_text()); M = json.loads((R.parent / 'new_baselines/results_main_cosa.json').read_text())['main']
OURS = P['ours_key']; D = P['per_domain']; BB = 'iTransformer'
COLS = [(OURS, r'\textbf{On-Adapter}'), ('COSA', r'COSA \ry{2026}'), ('ADAPT-Z', r'ADAPT-Z \ry{2025}'), ('DSOF', r'DSOF \ry{2025}'),
        ('Proceed', r'Proceed \ry{2025}'), ('SOLID', r'SOLID \ry{2024}'), ('ADCSD', r'ADCSD \ry{2024}'),
        ('MM-TSFlib', r'MM-TSFlib \ry{2024}'), ('GPT4MTS', r'GPT4MTS \ry{2024}'), (BB, r'iTransformer \ry{2024}')]
ROWS = [('Agriculture', 'Agriculture'), ('Climate', 'Climate'), ('Economy', 'Economy'), ('Energy', 'Energy'), ('Environment', 'Environment'),
        ('Health_US', 'Health(US)'), ('Security', 'Security'), ('SocialGood', 'SocialGood'), ('Traffic', 'Traffic')]


def mark(vals, fmt, lower=True, skip=()):
    shown = {m: fmt % v for m, v in vals.items()}; key = lambda m: float(shown[m]) * (1 if lower else -1)
    u = sorted(set(key(m) for m in vals if m not in skip))
    return {m: shown[m] if m in skip else r'\best{%s}' % shown[m] if key(m) == u[0] else r'\second{%s}' % shown[m] if len(u) > 1 and key(m) == u[1] else shown[m] for m in vals}


def row(name, cells): return name + ' & ' + ' & '.join(cells[m] for m, _ in COLS) + r' \\'


n = len(COLS)
L = [r'\begin{table*}[htbp]', r'\centering',
     r'\caption{Tail-risk results of online multimodal forecasting on the nine Time-MMD datasets. Each origin is scored by its mean absolute error over observed targets, and CVaR$_{90}$ is the mean of the worst 10\% of these per-origin errors within a test stream (lower is better). Dataset rows average CVaR$_{90}$ over four prediction horizons and three random seeds (2025, 2026, 2027). Summary rows are computed over the 36 dataset--horizon tasks: $S$ is the geometric mean over tasks of the seed-averaged CVaR$_{90}$ ratio to the iTransformer backbone (the reference, $S=1$); Avg.~Rank ranks the ten methods by CVaR$_{90}$ in each task and seed; \#Better and \#Harm count tasks whose ratio is below 1 and above 1.02; Worst is the largest task ratio. The best result in each row is shown in \textcolor{red}{\textbf{red}} and the second best is \textcolor{blue}{\underline{underlined}}; ratio-based rows exclude the backbone itself.}',
     r'\label{tab:tail_results}', r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{1.12}', r'\resizebox{\textwidth}{!}{%',
     r'\begin{tabular}{l|' + 'c' * n + '}', r'\toprule',
     r'\textbf{Models} & ' + ' & '.join(h for _, h in COLS) + r' \\', r'\midrule',
     r'\multicolumn{%d}{l}{\textit{CVaR$_{90}$ per dataset}} \\' % (n + 1), r'\midrule']
for key, name in ROWS: L.append(row(name, mark({m: D[m][key]['cvar90'] for m, _ in COLS}, '%.3f')))
L += [r'\midrule', r'\multicolumn{%d}{l}{\textit{Summary over 36 tasks}} \\' % (n + 1), r'\midrule']
g = lambda k: {m: M[m][k] for m, _ in COLS}
L.append(row(r'$S$ $\downarrow$', mark(g('S'), '%.4f', skip=(BB,))))
L.append(row(r'Avg.~Rank $\downarrow$', mark(g('mean_rank'), '%.2f')))
for k, nm, low in (('tasks_better_than_backbone', r'\#Better $\uparrow$', False), ('harm_tasks_over_2pct', r'\#Harm $\downarrow$', True)):
    c = mark({m: v for m, v in g(k).items()}, '%d', lower=low, skip=(BB,)); c[BB] = '--'; L.append(row(nm, c))
c = mark(g('worst_task_ratio'), '%.3f', skip=(BB,)); c[BB] = '--'; L.append(row(r'Worst $\downarrow$', c))
L += [r'\bottomrule', r'\end{tabular}%', '}', r'\end{table*}']
(R / 'table_cvar.tex').write_text('\n'.join(L) + '\n', 'utf-8'); print('\n'.join(L))

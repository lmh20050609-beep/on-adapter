"""LaTeX for the summary table (methods x metrics) from results/paper/metric_scan.json.
Marks use competition ranking on printed values (ties share a rank; a mark is given only to ranks 1 and 2).
The backbone is the reference of every ratio-based column and is not marked there."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results/paper'; X = json.loads((R / 'metric_scan.json').read_text())
OURS = [m for m in X if m.startswith('On-Adapter')][0]; BB = 'iTransformer'
ROWS = [(OURS, r'\textbf{On-Adapter (Ours)}'), ('COSA', r'COSA \ry{2026}'), ('ADAPT-Z', r'ADAPT-Z \ry{2025}'), ('DSOF', r'DSOF \ry{2025}'),
        ('Proceed', r'Proceed \ry{2025}'), ('SOLID', r'SOLID \ry{2024}'), ('ADCSD', r'ADCSD \ry{2024}'),
        ('MM-TSFlib', r'MM-TSFlib \ry{2024}'), ('GPT4MTS', r'GPT4MTS \ry{2024}'), (BB, r'iTransformer \ry{2024} (backbone)')]
n_cells = 108
COLS = [  # key, fmt, lower-is-better, backbone is reference (not marked / shown as --)
    (lambda v: v['S_mae'], '%.4f', True, 'mark_skip'), (lambda v: v['S_mse'], '%.4f', True, 'mark_skip'),
    (lambda v: v['S_cvar80'], '%.4f', True, 'mark_skip'), (lambda v: v['S_cvar90'], '%.4f', True, 'mark_skip'),
    (lambda v: v['rank_cvar90'], '%.2f', True, None), (lambda v: 100 * v['top5'] / n_cells, '%.1f', False, None),
    (lambda v: v['harm>0.02'], '%d', True, 'dash'), (lambda v: v['worst_task'], '%.3f', True, 'dash'),
    (lambda v: v['sum_gain_pct'] / v['sum_harm_pct'] if v['sum_harm_pct'] else None, '%.2f', False, 'dash')]
cells = {m: [] for m, _ in ROWS}
for f, fmt, low, bb in COLS:
    shown = {m: fmt % f(X[m]) for m, _ in ROWS if not (bb == 'dash' and m == BB)}
    ranked = [m for m in shown if not (bb and m == BB)]; sgn = 1 if low else -1
    for m, _ in ROWS:
        if m not in shown: cells[m].append('--'); continue
        if m not in ranked: cells[m].append(shown[m]); continue
        rank = 1 + sum(sgn * float(shown[o]) < sgn * float(shown[m]) for o in ranked)
        cells[m].append(r'\best{%s}' % shown[m] if rank == 1 else r'\second{%s}' % shown[m] if rank == 2 else shown[m])
L = [r'\begin{table*}[htbp]', r'\centering',
     r'\caption{Overall comparison on the nine Time-MMD datasets (36 dataset--horizon tasks, three random seeds 2025, 2026, 2027; 108 test streams). For each task we average the ratio of each metric to that of the iTransformer backbone over seeds; $S$ is the geometric mean of this ratio over tasks ($S<1$: better than the backbone). CVaR$_{\alpha}$ is the mean of the worst $(1-\alpha)$ fraction of per-origin MAEs within a test stream. Avg.~Rank ranks the ten methods by CVaR$_{90}$ in each stream; Top-5 is the share of streams in which a method ranks among the best five. \#Harm counts tasks whose CVaR$_{90}$ ratio exceeds 1.02 and Worst is the largest task ratio. Gain/Harm is $\sum_t \max(0, 1-r_t)\,/\,\sum_t \max(0, r_t-1)$ over the task CVaR$_{90}$ ratios $r_t$, i.e.\ the total improvement delivered per unit of total degradation. The best result in each column is shown in \textcolor{red}{\textbf{red}} and the second best is \textcolor{blue}{\underline{underlined}}; the backbone is the reference of ratio-based columns and is not ranked there.}',
     r'\label{tab:overall}', r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{1.12}', r'\resizebox{\textwidth}{!}{%',
     r'\begin{tabular}{l|cc|ccc|cccc}', r'\toprule',
     r' & \multicolumn{2}{c|}{\textbf{Point accuracy}} & \multicolumn{3}{c|}{\textbf{Tail risk}} & \multicolumn{4}{c}{\textbf{Reliability}} \\',
     r'\textbf{Models} & $S_{\mathrm{MAE}}\downarrow$ & $S_{\mathrm{MSE}}\downarrow$ & $S_{\mathrm{CVaR}_{80}}\downarrow$ & $S_{\mathrm{CVaR}_{90}}\downarrow$ & Avg.~Rank$\downarrow$ & Top-5 (\%)$\uparrow$ & \#Harm$\downarrow$ & Worst$\downarrow$ & Gain/Harm$\uparrow$ \\',
     r'\midrule']
for m, name in ROWS:
    if m == BB: L.append(r'\midrule')
    L.append(name + ' & ' + ' & '.join(cells[m]) + r' \\')
L += [r'\bottomrule', r'\end{tabular}%', '}', r'\end{table*}']
(R / 'table_overall.tex').write_text('\n'.join(L) + '\n', 'utf-8'); print('\n'.join(L))

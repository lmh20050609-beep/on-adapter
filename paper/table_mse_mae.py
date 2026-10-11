"""LaTeX for paper Table 2 (tab:main_results) from results/paper/table_domain.json.
Best / second best per column among the ten methods, decided on the printed (3-decimal) values; ties share the mark."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results/paper'; J = json.loads((R / 'table_domain.json').read_text())
P = J['per_domain']; OURS = J['ours_key']
COLS = [(OURS, r'\textbf{On-Adapter}'), ('COSA', r'COSA \ry{2026}'), ('ADAPT-Z', r'ADAPT-Z \ry{2025}'), ('DSOF', r'DSOF \ry{2025}'),
        ('Proceed', r'Proceed \ry{2025}'), ('SOLID', r'SOLID \ry{2024}'), ('ADCSD', r'ADCSD \ry{2024}'),
        ('MM-TSFlib', r'MM-TSFlib \ry{2024}'), ('GPT4MTS', r'GPT4MTS \ry{2024}'), ('iTransformer', r'iTransformer \ry{2024}')]
ROWS = [('Agriculture', 'Agriculture'), ('Climate', 'Climate'), ('Economy', 'Economy'), ('Energy', 'Energy'), ('Environment', 'Environment'),
        ('Health_US', 'Health(US)'), ('Security', 'Security'), ('SocialGood', 'SocialGood'), ('Traffic', 'Traffic')]
n = len(COLS); L = []
L += [r'\begin{table*}[htbp]', r'\centering',
      r'\caption{Online multimodal time series forecasting results on the nine Time-MMD datasets, averaged over four prediction horizons and three random seeds (2025, 2026, 2027). Following the sampling frequency of each dataset, the look-back length is fixed to 8, 24, or 96, paired with prediction horizons $\{6, 8, 10, 12\}$, $\{12, 24, 36, 48\}$, and $\{48, 96, 192, 336\}$, respectively. Forecasts are issued at every origin and ground truth becomes available only after the full horizon has elapsed. The best result in each column is shown in \textcolor{red}{\textbf{red}} and the second best is \textcolor{blue}{\underline{underlined}}.}',
      r'\label{tab:main_results}', r'\setlength{\tabcolsep}{2.8pt}', r'\renewcommand{\arraystretch}{1.12}', r'\resizebox{\textwidth}{!}{%',
      r'\begin{tabular}{l|' + 'c' * (2 * n) + '}', r'\toprule', r'\textbf{Models}']
L += ['  & \\multicolumn{2}{c}{%s}' % h for _, h in COLS]; L[-1] += r' \\'
L += [r'\midrule', 'Metric & ' + ' & '.join(['MSE & MAE'] * n) + r' \\', r'\midrule']
for i, (key, name) in enumerate(ROWS):
    cells = {}
    for met in ('mse', 'mae'):
        vals = {m: round(P[m][key][met], 3) for m, _ in COLS}; u = sorted(set(vals.values()))
        for m, v in vals.items():
            s = '%.3f' % v; cells[(m, met)] = r'\best{%s}' % s if v == u[0] else r'\second{%s}' % s if len(u) > 1 and v == u[1] else s
    L.append(name)
    for j in range(0, n, 3):
        L.append('  & ' + ' & '.join('%s & %s' % (cells[(m, 'mse')], cells[(m, 'mae')]) for m, _ in COLS[j:j + 3]))
    L[-1] += r' \\'; L.append(r'\bottomrule' if i == len(ROWS) - 1 else r'\midrule')
L += [r'\end{tabular}%', '}', r'\end{table*}']
(R / 'table_mse_mae.tex').write_text('\n'.join(L) + '\n', 'utf-8'); print('\n'.join(L))

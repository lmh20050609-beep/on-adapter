"""Round-1 post-mortem (exploratory; test already scored). Per task: original On, lr0 (a=0), RC, validation RC ratio."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'scripts')
import seed2_round1_rc as rc
R1 = Path('reports/seed2_goal_20261009/round1')
res = json.loads((R1 / 'results.json').read_text('utf-8')); sel = json.loads((R1 / 'selection.json').read_text('utf-8'))
w = sel['winner']; key = '%s|%s|%s' % (w['base'], w['a'], w['gamma'])
rows = []
for t in res['per_task']:
    d, h = t['task'].split(' H'); h = int(h)
    tr, _ = rc.load('test', d, h, 'lr0', 0)
    ref = rc.cvar90(rc.per_origin_mae(tr['backbone'], tr['targets'], tr['mask']))
    lr0 = rc.cvar90(rc.per_origin_mae(tr['predictions'], tr['targets'], tr['mask'])) / ref
    rows.append((t['task'], t['On-Adapter']['rel_cvar90'], lr0, t['On-Adapter-RC/r0']['rel_cvar90'], sel['per_task'][key][t['task']],
                 min(t[m]['rel_cvar90'] for m in ['ADAPT-Z', 'DSOF', 'Proceed', 'GPT4MTS', 'MM-TSFlib'])))
print('%-16s %7s %7s %7s %7s %7s' % ('task', 'origOn', 'lr0', 'RC', 'valRC', 'bestOth'))
for r in rows: print('%-16s %7.4f %7.4f %7.4f %7.4f %7.4f' % r)
gm = lambda x: math.exp(np.mean(np.log(x)))
for d in ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood', 'Security', 'Economy', 'Climate', 'Agriculture']:
    s = [r for r in rows if r[0].startswith(d)]
    print('%-12s origOn %.4f lr0 %.4f RC %.4f valRC %.4f  sum log RC %+.4f' % (d, gm([r[1] for r in s]), gm([r[2] for r in s]), gm([r[3] for r in s]), gm([r[4] for r in s]), sum(math.log(r[3]) for r in s)))
print('ALL lr0 %.4f RC %.4f' % (gm([r[2] for r in rows]), gm([r[3] for r in rows])))

import sys; sys.path.insert(0, 'scripts')
import numpy as np, seed2_round1_rc as rc
t, _ = rc.load('val', 'Agriculture', 6, 'lr0', 0)
p, y, m, o = t['predictions'], t['targets'].astype(float), t['mask'], t['origins']
base = rc.calibrate(p, y, m, o, 6, .5, .97)
for i in (10, 40, len(o) - 1):
    y2 = y.copy(); fut = o + 6 > o[i]           # origins not fully matured at forecast i (incl. i itself and later)
    y2[fut] = y2[fut] + 1e3
    q = rc.calibrate(p, y2, m, o, 6, .5, .97)
    assert np.array_equal(q[:i + 1], base[:i + 1]), i
print('boundary ok: forecasts up to t unchanged when unmatured targets are perturbed')

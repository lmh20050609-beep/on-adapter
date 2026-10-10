import sys; sys.path.insert(0, 'scripts')
import numpy as np, seed2_round1_rc as r1, seed2_round2_pc as r2
for d, h in [('Climate', 6), ('Energy', 24), ('Environment', 96)]:
    t, _ = r1.load('val', d, h, 'lr0', 0)
    a = r1.calibrate(t['predictions'], t['targets'], t['mask'], t['origins'], h, .75, 1.)
    b = r2.calibrate(t['predictions'], t['targets'], t['mask'], t['origins'], h, 'step', .75, 1., 12)
    assert np.allclose(a, b, rtol=0, atol=1e-12), (d, h, np.abs(a - b).max())
    y2 = t['targets'].astype(float).copy(); i = len(t['origins']) // 2
    y2[t['origins'] + h > t['origins'][i]] += 1e3
    p1 = r2.calibrate(t['predictions'], t['targets'], t['mask'], t['origins'], h, 'phase', .75, 1., r2.period(d))
    p2 = r2.calibrate(t['predictions'], y2, t['mask'], t['origins'], h, 'phase', .75, 1., r2.period(d))
    assert np.array_equal(p1[:i + 1], p2[:i + 1]), 'phase boundary'
print('step == round1 calibrate; phase calibration ignores unmatured targets')

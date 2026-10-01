import unittest
import numpy as np
from iadapter.online_mix import MomentGate,HedgeGate,FixedGate,moment_prior,replay,warm_hedge


def ledger(H=3,rows=12):
    n=np.zeros((rows,H,1));m=np.ones_like(n);y=np.full_like(n,.75)
    return dict(origins=np.arange(20,20+rows),numeric=n,text=m,targets=y,
                valid_mask=np.ones_like(n,dtype=bool),available=np.ones(rows,dtype=bool))


class OnlineMixTests(unittest.TestCase):
    def test_phase_target_end_never_crosses_boundary(self):
        from iadapter.calibration_pilot import region_origins,bounds
        for n in [900,1479]:
            b=bounds(n)
            for H in [1,6,12]:
                for phase,lo,hi in [('calibration',b['checkpoint_end'],b['calibration_end']),
                                    ('selection',b['calibration_end'],b['selection_end']),
                                    ('assessment',b['selection_end'],n)]:
                    o=region_origins(n,48,H,phase)
                    self.assertTrue(np.all(o+1>=lo));self.assertTrue(np.all(o+H<hi))

    def test_future_label_invariance_and_maturity(self):
        x=ledger();prior=moment_prior(x,100);gate=MomentGate(prior)
        a,e=replay(x,gate)
        changed={k:v.copy() for k,v in x.items()};changed['targets'][4:]=99.
        b,_=replay(changed,gate)
        # Label for origin24 matures at27, after that origin's emitted forecast.
        np.testing.assert_array_equal(a['predictions'][:8],b['predictions'][:8])
        self.assertEqual([v['origin'] for v in e],list(range(20,29)))
        self.assertTrue(all(v['update_at']>=v['mature_at'] for v in e))
        np.testing.assert_array_equal(a['feedback_origins'][:4],[-1]*4)

    def test_masked_values_cannot_change_prior_or_state(self):
        x=ledger();x['valid_mask'][::2,0]=False
        y={k:v.copy() for k,v in x.items()};y['targets'][~y['valid_mask']]=np.nan
        p=moment_prior(x,100);q=moment_prior(y,100)
        np.testing.assert_array_equal(p['alpha0'],q['alpha0'])
        for gate in [MomentGate(p),HedgeGate(3,.1)]:
            a,_=replay(x,gate);b,_=replay(y,gate)
            np.testing.assert_array_equal(a['predictions'],b['predictions'])

    def test_no_text_emits_numeric_and_not_value_evidence(self):
        x=ledger();x['available'][:]=False
        p=moment_prior(x,100);np.testing.assert_array_equal(p['count'],[0]*3)
        a,e=replay(x,MomentGate(p,rho=.98))
        np.testing.assert_array_equal(a['predictions'],x['numeric'])
        self.assertTrue(all(not z['value_updated'] for z in e))

    def test_closed_form_matches_independent_grid_minimum(self):
        prior=dict(H=1,shared=False,alpha0=np.asarray([.2]),mean_d2=np.asarray([.4]))
        gate=MomentGate(prior,kappa=1.)
        n=np.asarray([[.3,-.2]]);m=np.asarray([[.9,.4]]);y=np.asarray([[.6,.7]])
        gate.observe(n,m,y,np.ones_like(n,dtype=bool))
        grid=np.linspace(0,1,100001)
        loss=((y[0]-n[0]-grid[:,None]*(m[0]-n[0]))**2).sum(1)+gate.lam[0]*(grid-.2)**2
        self.assertAlmostEqual(float(gate.alpha()[0]),float(grid[np.argmin(loss)]),places=4)

    def test_horizon_and_shared_priors_and_zero_denominator(self):
        x=ledger(2,4);x['targets'][:,0]=0.;x['targets'][:,1]=1.
        p=moment_prior(x,100);q=moment_prior(x,100,shared=True)
        np.testing.assert_array_equal(p['alpha0'],[0,1]);np.testing.assert_array_equal(q['alpha0'],[.5])
        np.testing.assert_array_equal(MomentGate(q).alpha(),[.5,.5])
        x['text']=x['numeric'].copy();p=moment_prior(x,100)
        np.testing.assert_array_equal(p['alpha0'],[.5,.5])

    def test_decay_once_per_clock_with_gap(self):
        prior=dict(H=1,shared=False,alpha0=np.asarray([.5]),mean_d2=np.asarray([1.]))
        gate=MomentGate(prior,rho=.5);gate.advance(5)
        gate.observe(np.zeros((1,1)),np.ones((1,1)),np.ones((1,1)),np.ones((1,1),bool))
        gate.advance(7);self.assertEqual(gate.A[0],.25);self.assertEqual(gate.B[0],.25)
        with self.assertRaises(AssertionError):gate.advance(7)

    def test_hedge_uses_identical_calibration_data(self):
        x=ledger();x['targets'][:]=1.
        w,ids=warm_hedge(x,100,1.)
        self.assertEqual(ids,x['origins'].tolist());self.assertTrue(np.all(w[:,1]>.99))
        mixed,_=warm_hedge(x,100,1.,mixed=True)
        self.assertTrue(np.all(mixed[:,-1]>mixed[:,0]))
        with self.assertRaises(AssertionError):warm_hedge(x,22,1.)

    def test_all_variants_convex_and_replay_does_not_mutate_initial_gate(self):
        x=ledger();p=moment_prior(x,100)
        for gate in [MomentGate(p),HedgeGate(3,10.),FixedGate(np.full(3,.5))]:
            initial=gate.alpha().copy();a,_=replay(x,gate)
            self.assertTrue(np.all(a['predictions']>=x['numeric']))
            self.assertTrue(np.all(a['predictions']<=x['text']))
            np.testing.assert_array_equal(initial,gate.alpha())
            b,_=replay(x,gate);np.testing.assert_array_equal(a['predictions'],b['predictions'])


if __name__=='__main__':unittest.main()

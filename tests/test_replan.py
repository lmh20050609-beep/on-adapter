import copy,unittest
from unittest import mock
import numpy as np
import pandas as pd
import torch
from iadapter.config import Config
from iadapter.models import Backbone
from iadapter.review_sprint import initialize_adapter
from iadapter.shared_online import InternalBackend,OnlineRunner
from iadapter.replan_audit import reconstruct
from iadapter.data import text_rows
from iadapter.text_signal_probe import fit_ridge,pca_basis
from iadapter.util import seed_all
from iadapter.feedback_factorial import full_start
from iadapter.external_static import pilot_config
from test_protocol import toy

class ReplanTests(unittest.TestCase):
    def test_factorial_loader_accepts_matrix_method_key(self):
        with mock.patch('iadapter.external_online.pilot_config',side_effect=lambda variant,**kwargs:pilot_config(variant,device='cpu',**kwargs)):
            c,data,backend,source,previous,initial,frozen=full_start(dict(dataset='Energy',seed=2023,method='iadapter_U1F0'))
        self.assertEqual(c.device,'cpu');self.assertEqual(data.columns,['OT']);self.assertTrue(initial);self.assertTrue(frozen)

    def test_factorial_switches_have_exact_parameter_cache_boundaries(self):
        data=toy();c=Config(device='cpu');seed_all(43);bone=Backbone(36,6,2,d_model=32);initial=initialize_adapter(c,2)
        origins=list(range(35,49))
        for u,f in [(False,False),(True,False),(False,True),(True,True)]:
            ad=copy.deepcopy(initial);before=copy.deepcopy(ad.state_dict());backend=InternalBackend(c,data,bone,ad)
            seed_all(12);arrays,events,_,stats=OnlineRunner(c,data,backend,update_parameters=u,refresh_feedback=f).run(origins)
            changed=any(not torch.equal(v,before[n]) for n,v in ad.state_dict().items());self.assertEqual(changed,u)
            self.assertEqual([e['origin'] for e in events],list(range(35,43)))
            if not u:self.assertTrue((arrays['parameter_update_origins']==-1).all())
            if not f:self.assertFalse(arrays['numeric_feedback_snapshots'].any());self.assertFalse(arrays['text_feedback_snapshots'].any())
            else:self.assertTrue(arrays['numeric_feedback_snapshots'][7:].any());self.assertTrue(arrays['text_feedback_snapshots'][7:].any())
            self.assertTrue(all(p.grad is None for p in bone.parameters()))
    def test_text_span_reconstruction_matches_existing_budget(self):
        frame=pd.DataFrame({'fact':['A documented increase in market demand is reported. '*35,'A new market report suggests that demand has decreased.',''], 'preds':['A clear forecast of future market development is given.']*3})
        texts,ids=text_rows(frame);rebuilt=reconstruct(frame)
        self.assertEqual(texts,[x[0] for x in rebuilt])
        for t,(_,spans) in enumerate(rebuilt):self.assertTrue(all(rid in ids[t] and int(rid.split(':')[0])<=t and 0<=a<=b<=len(texts[t]) for rid,a,b in spans))
    def test_ridge_ignores_invalid_training_targets(self):
        rng=np.random.default_rng(0);x=rng.normal(size=(30,5));y=rng.normal(size=(30,6));mask=np.ones_like(y,dtype=bool);mask[5,2]=False
        a=fit_ridge(x,y,mask,1.);y[5,2]=1e9;b=fit_ridge(x,y,mask,1.);np.testing.assert_array_equal(a,b)
    def test_pca_fit_excludes_future_and_missing_content_is_zero(self):
        rng=np.random.default_rng(4);emb=rng.normal(size=(60,30));available=np.ones(60,dtype=bool);available[3]=False
        one,basis=pca_basis(emb,available,40,5);emb[40:]+=500;two,other=pca_basis(emb,available,40,5)
        np.testing.assert_array_equal(one[:40],two[:40]);np.testing.assert_array_equal(basis['basis'],other['basis']);self.assertFalse(one[3].any())

if __name__=='__main__':unittest.main()

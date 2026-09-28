import copy,unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from iadapter.config import Config,matrix
from iadapter.data import Data,causal_fill,text_rows
from iadapter.engine import MatureQueue,Stream
from iadapter.models import Backbone,Adaptation,SoftMoE
from iadapter.util import masked_loss,metrics,seed_all

torch.set_num_threads(1)

def toy():
    rng=np.random.default_rng(10); x=rng.normal(size=(105,2)).astype(np.float32)
    return Data('synthetic_test_only',x,x.copy(),np.ones_like(x,dtype=bool),np.zeros_like(x,dtype=bool),['known event description']*105,[[f'{i}:fact'] for i in range(105)],[str(i) for i in range(105)],['a','b'],65,85,np.zeros(2),np.ones(2),'synthetic_unit_test',rng.normal(size=(105,768)).astype(np.float32))

class ProtocolTests(unittest.TestCase):
    def test_matrix(self):
        jobs=matrix(); self.assertEqual(len(jobs),190)
        self.assertEqual(len({(j['dataset'],j['horizon'],j['seed'],j['variant']) for j in jobs}),190)
        self.assertEqual(len({(j['dataset'],j['horizon'],j['seed']) for j in jobs}),35)
    def test_queue_various_horizons_and_chunks(self):
        for H in [1,6,8,12,20]:
            for B in [1,8,16]:
                q=MatureQueue(); consumed=[]
                for start in range(0,51,B):
                    for t in range(start,min(51,start+B)):
                        q.push(SimpleNamespace(origin=t,mature=t+H))
                        consumed.extend(r.origin for r in q.pop_ready(t))
                self.assertEqual(consumed,list(range(51-H)))
    def test_causal_fill(self):
        a=np.array([[np.nan],[1.],[np.nan],[5.]])
        b=a.copy(); b[-1]=999
        np.testing.assert_array_equal(causal_fill(a)[:3],causal_fill(b)[:3])
        self.assertEqual(causal_fill(a)[0,0],0.)
    def test_masked_targets(self):
        p=torch.tensor([1.,5.],requires_grad=True); y=torch.tensor([2.,999.]); m=torch.tensor([True,False])
        loss=masked_loss(p,y,m); loss.backward()
        self.assertEqual(loss.item(),1.); self.assertEqual(p.grad[1].item(),0.)
        self.assertEqual(metrics(p.detach().numpy(),y.numpy(),m.numpy())['mse'],1.)
    def test_text_future_invariance(self):
        f=pd.DataFrame({'fact':['The production volume increased substantially during this month.']*4,'preds':['']*4})
        a,ids=text_rows(f); f.loc[3,'fact']='A different future event changes the anticipated business outlook.'
        b,_=text_rows(f); self.assertEqual(a[:3],b[:3]); self.assertTrue(all(int(i.split(':')[0])<=2 for i in ids[2]))
    def test_split_backbone_equivalent(self):
        seed_all(1); b=Backbone(36,6,2,d_model=32,e_layers=2).eval(); x=torch.randn(2,36,2)
        from iadapter.legacy_core import ITransformerZ
        old=ITransformerZ(36,6,2,d_model=32,e_layers=2).eval(); old.load_state_dict(b.state_dict())
        with torch.no_grad(): torch.testing.assert_close(b(x),old(x)['pred'])
    def test_ablation_paths(self):
        feat=torch.randn(1,2,32); hg=torch.randn(2,32); e=torch.randn(1,64); ge=torch.randn(64); off=torch.randn(1,8)
        def make(v):
            seed_all(4); return SoftMoE(Config(variant=v,device='cpu'),2).eval()
        for variant in ['no_text_gradient','no_text','no_numeric_gradient','no_distribution']:
            a=make(variant)
            args=[feat,hg,e,ge,off]; changed=[v.clone() for v in args]
            if variant=='no_text_gradient': changed[3]+=30
            if variant=='no_text': changed[2]+=30; changed[3]+=30
            if variant=='no_numeric_gradient': changed[1]+=30
            if variant=='no_distribution': changed[4]+=30
            torch.testing.assert_close(a(*args),a(*changed))
        a=make('no_route_text_gradient'); a(feat,hg,e,ge,off); w=a.last_gate.clone()
        a(feat,hg,e,ge+10,off); torch.testing.assert_close(w,a.last_gate)
        a=make('uniform'); a(feat,hg,e,ge,off); torch.testing.assert_close(a.last_gate,torch.full_like(a.last_gate,.2)); self.assertEqual(float(a.aux_loss),0.)
        a=make('no_decisiveness'); a(feat,hg,e,ge,off); self.assertEqual(float(a.aux_loss),0.)
    def test_full_soft_gate(self):
        a=SoftMoE(Config(device='cpu'),2).eval()
        a(torch.randn(1,2,32),torch.zeros(2,32),torch.zeros(1,64),torch.zeros(64),torch.zeros(1,8))
        self.assertTrue((a.last_gate>0).all())
    def test_stream_future_labels_and_numeric_no_text(self):
        data=toy(); c=Config(variant='numeric_add',device='cpu')
        seed_all(3); model=Backbone(36,6,2,d_model=32); initial=copy.deepcopy(model.state_dict()); a=Adaptation(c,2); astate=copy.deepcopy(a.state_dict())
        # The same origin forecasts remain unchanged when all not-yet-observed
        # targets/text at the first test origin are modified.
        seed_all(5); out,ev,stats=Stream(c,data,model,a).run('test',origins=list(range(84,94)))
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(v,astate[k]) for k,v in a.state_dict().items()))
        modified=copy.deepcopy(data); modified.target[85:]+=100; modified.embeddings[:]=np.nan
        a2=Adaptation(c,2); a2.load_state_dict(astate)
        seed_all(5); out2,_,_=Stream(c,modified,model,a2).run('test',origins=[84])
        np.testing.assert_array_equal(out['predictions'][0],out2['predictions'][0])
        self.assertEqual([e['origin'] for e in ev],[84,85,86,87])
        self.assertTrue(all(e['origin']==e['text_origin']==e['state_origin'] for e in ev))

if __name__=='__main__': unittest.main()

import copy,unittest
import numpy as np
import torch
from torch import nn
from iadapter.config import Config
from iadapter.models import Backbone
from iadapter.review_sprint import initialize_adapter
from iadapter.engine import Stream
from iadapter.shared_online import InternalBackend,NativeBackend,OnlineRunner
from iadapter.util import seed_all
from test_protocol import toy

class TinyNative(nn.Module):
    def __init__(self):
        super().__init__();self.out=nn.Linear(36,6);self.text=nn.Linear(768,6);self.frozen=nn.Parameter(torch.ones(1),requires_grad=False)
    def forward(self,x,iteration,summary):return (self.out(x[:,:,0])+self.text(summary.mean(1))).unsqueeze(-1)

def univariate():
    d=toy();d.x=d.x[:,:1].copy();d.target=d.target[:,:1].copy();d.valid=d.valid[:,:1].copy();d.columns=['OT'];return d

class SharedOnlineTests(unittest.TestCase):
    def test_internal_matches_existing_stream_exactly(self):
        d=toy();origins=list(range(84,95))
        for variant in ['numeric_add','full']:
            c=Config(variant=variant,device='cpu');seed_all(5);bone=Backbone(36,6,2,d_model=32);ad=initialize_adapter(c,2);replica=copy.deepcopy(ad)
            seed_all(12);old,old_events,_=Stream(c,d,bone,ad).run('test',origins=origins)
            backend=InternalBackend(c,d,bone,replica);seed_all(12);new,events,contexts,_=OnlineRunner(c,d,backend).run(origins)
            for name in ['predictions','targets','valid_mask','gate_weights','feedback_origins']:np.testing.assert_array_equal(old[name],new[name])
            self.assertEqual(old_events,events)
            for name,value in ad.state_dict().items():torch.testing.assert_close(value,replica.state_dict()[name],rtol=0,atol=0)
            self.assertTrue((new['parameter_update_origins'][:7]==-1).all());self.assertEqual(new['parameter_update_origins'][7],84)
    def test_native_future_inputs_labels_and_exact_maturity(self):
        d=univariate();changed=copy.deepcopy(d);changed.target[95:]+=999;changed.embeddings[95:]+=999
        c=Config(device='cpu');seed_all(7);a=TinyNative();b=copy.deepcopy(a);origins=list(range(84,95))
        one=NativeBackend(c,d,a,d.embeddings);two=NativeBackend(c,changed,b,changed.embeddings)
        seed_all(8);x,events,_,stats=OnlineRunner(c,d,one).run(origins)
        seed_all(8);y,events2,_,_=OnlineRunner(c,changed,two).run(origins)
        np.testing.assert_array_equal(x['predictions'],y['predictions']);self.assertEqual(events,events2)
        self.assertEqual([e['origin'] for e in events],list(range(84,89)));self.assertEqual(stats['updated'],5)
        self.assertEqual(stats['pending'],6);self.assertTrue((x['feedback_origins']==-1).all())
        self.assertEqual(float(a.frozen),1.);self.assertIsNone(a.frozen.grad)
    def test_missing_targets_are_consumed_without_update(self):
        d=univariate();d.valid[85:95]=False;c=Config(device='cpu');model=TinyNative();before=copy.deepcopy(model.state_dict())
        arrays,events,_,stats=OnlineRunner(c,d,NativeBackend(c,d,model,d.embeddings)).run(list(range(84,95)))
        self.assertEqual(stats['updated'],0);self.assertEqual(stats['consumed'],5)
        self.assertTrue(all(e['status']=='skipped_missing_target' for e in events));self.assertTrue((arrays['parameter_update_origins']==-1).all())
        for n,v in model.state_dict().items():torch.testing.assert_close(v,before[n],rtol=0,atol=0)

if __name__=='__main__':unittest.main()

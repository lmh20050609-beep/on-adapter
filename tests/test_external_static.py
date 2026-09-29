import copy,unittest
import numpy as np
import torch
from torch import nn
from iadapter.external_static import evaluate,ot_data,pilot_config
from iadapter.models import Backbone
from iadapter.review_sprint import initialize_adapter
from test_protocol import toy

class NativeStub(nn.Module):
    def forward(self,x,iteration,summary):
        return x[:,-1:,:].repeat(1,6,1)+summary.mean().reshape(1,1,1)*.001

class StaticTests(unittest.TestCase):
    def test_ot_extraction_preserves_original_scale_and_masks(self):
        d,b=ot_data();self.assertEqual(d.columns,['OT']);self.assertEqual(d.x.shape,(338,1))
        self.assertEqual((d.train_end,d.val_end),(253,296));self.assertEqual(d.origins('test',36,6),list(range(295,332)))
    def test_static_internal_future_labels_and_parameter_invariance(self):
        d=toy();d.x=d.x[:,:1].copy();d.target=d.target[:,:1].copy();d.valid=d.valid[:,:1].copy();d.columns=['OT']
        changed=copy.deepcopy(d);changed.target[:]=1000
        for variant in ['numeric_add','full']:
            c=pilot_config(variant,'cpu');model=Backbone(36,6,1,d_model=32).eval();ad=initialize_adapter(c,1)
            before=copy.deepcopy(ad.state_dict());a,ctx=evaluate(d,c,model,ad);b,_=evaluate(changed,c,model,ad)
            np.testing.assert_array_equal(a['predictions'],b['predictions'])
            for n,v in ad.state_dict().items():torch.testing.assert_close(v,before[n],rtol=0,atol=0)
            self.assertTrue(all(not x['updated_after_prediction'] for x in ctx))
            if variant=='numeric_add':self.assertTrue(all(not x['text_ids'] for x in ctx))
    def test_native_window_never_reads_future_text(self):
        d=toy();d.x=d.x[:,:1].copy();d.target=d.target[:,:1].copy();d.valid=d.valid[:,:1].copy();d.columns=['OT']
        c=pilot_config(device='cpu');embedding=d.embeddings.copy();first=d.origins('test',36,6)[0]
        a,ctx=evaluate(d,c,NativeStub(),cls_embeddings=embedding)
        embedding[first+1:]+=10000;b,_=evaluate(d,c,NativeStub(),cls_embeddings=embedding)
        np.testing.assert_array_equal(a['predictions'][0],b['predictions'][0])
        self.assertTrue(all(int(x.split(':')[0])<=first for x in ctx[0]['text_ids']))

if __name__=='__main__':unittest.main()

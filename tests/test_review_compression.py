import copy,unittest
import numpy as np
import torch
from iadapter.config import Config
from iadapter.models import Backbone
from iadapter.engine import Stream
from iadapter.review_compression import jobs,initialize_projection,METHODS
from iadapter.review_sprint import initialize_adapter
from iadapter.util import seed_all
from test_protocol import toy

class CompressionTests(unittest.TestCase):
    def test_grid_and_unchanged_wide_controls(self):
        self.assertEqual(len(jobs()),12);self.assertEqual(len(jobs(True)),2)
        for variant in METHODS:
            c=Config(variant=variant,device='cpu');wide=initialize_projection(c,2,96);old=initialize_adapter(c,2)
            slim=initialize_projection(c,2,8)
            self.assertEqual(sum(p.numel() for p in wide.project.parameters()),80160)
            self.assertEqual(sum(p.numel() for p in slim.project.parameters()),6856)
            for k,v in wide.state_dict().items():
                torch.testing.assert_close(v,old.state_dict()[k],rtol=0,atol=0)
                if slim.state_dict()[k].shape==v.shape:torch.testing.assert_close(v,slim.state_dict()[k],rtol=0,atol=0)
            self.assertEqual(slim.project[0].weight.shape,(8,768))
            self.assertEqual(slim.project[2].weight.shape,(64,8))
            self.assertTrue(torch.equal(slim.text(torch.randn(1,768),False,'cpu'),torch.zeros(1,64)))
    def test_slim_preserves_routing_ablation_and_expert_feedback(self):
        c=Config(variant=METHODS[0],device='cpu');ad=initialize_projection(c,2,8).eval()
        a=[torch.randn(1,2,32),torch.randn(2,32),torch.randn(1,64),torch.randn(64),torch.randn(1,8)]
        y=ad(*a);gate=ad.gate.clone();a[3]+=30;z=ad(*a)
        torch.testing.assert_close(ad.gate,gate,rtol=0,atol=0)
        self.assertEqual(float(ad.auxiliary),0.);self.assertFalse(torch.equal(y,z))
    def test_slim_updates_without_changing_backbone_or_future_dependence(self):
        d=toy();other=copy.deepcopy(d);other.target[95:]+=999;other.embeddings[95:]+=999
        for variant in METHODS:
            c=Config(variant=variant,device='cpu');seed_all(7);model=Backbone(36,6,2,d_model=32).eval()
            frozen=copy.deepcopy(model.state_dict());ad=initialize_projection(c,2,8);initial=copy.deepcopy(ad.state_dict());replica=copy.deepcopy(ad)
            seed_all(12);a,ev,_=Stream(c,d,model,ad).run('test',origins=list(range(84,95)))
            seed_all(12);b,ev2,_=Stream(c,other,model,replica).run('test',origins=list(range(84,95)))
            np.testing.assert_array_equal(a['predictions'],b['predictions']);self.assertEqual(ev,ev2)
            self.assertFalse(torch.equal(initial['project.0.weight'],ad.state_dict()['project.0.weight']))
            for k,v in model.state_dict().items():torch.testing.assert_close(v,frozen[k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()

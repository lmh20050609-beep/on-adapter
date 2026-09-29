import copy,unittest
from dataclasses import replace
import numpy as np
import torch
from test_protocol import toy
from iadapter.config import Config
from iadapter.models import Backbone
from iadapter.engine import Stream
from iadapter.util import seed_all
from iadapter.review_sprint import truncate_development,initialize_adapter,DevelopmentObserver,matrix

class ReviewSprintTests(unittest.TestCase):
    def test_predeclared_matrix(self):
        self.assertEqual(len(matrix()),42);self.assertEqual(len(matrix(True)),2)
        self.assertEqual(len({(j['dataset'],j['seed']) for j in matrix()}),6)
    def test_development_excludes_old_test_and_uses_separate_selection(self):
        raw=toy();raw.train_end=int(.6*len(raw.x))
        d,b=truncate_development(raw)
        self.assertEqual(len(d.x),84);self.assertEqual(d.train_end,63);self.assertEqual(d.val_end,73)
        self.assertEqual(d.origins('val',36,6)[-1]+6,72)
        self.assertEqual(d.origins('test',36,6)[-1]+6,83)
        modified=copy.deepcopy(raw);modified.target[84:]=999;modified.embeddings[84:]=999
        other,_=truncate_development(modified)
        np.testing.assert_array_equal(d.target,other.target);np.testing.assert_array_equal(d.embeddings,other.embeddings)
    def test_factorial_shared_initialization_and_single_conditioner(self):
        base=initialize_adapter(Config(variant='full',device='cpu'),2)
        for name in ['no_route_text_gradient','no_decisiveness','no_route_no_decisiveness','single_mlp']:
            other=initialize_adapter(Config(variant=name,device='cpu'),2)
            for key,value in other.state_dict().items():
                if key in base.state_dict():torch.testing.assert_close(value,base.state_dict()[key],rtol=0,atol=0)
        self.assertTrue(torch.equal(base.adapter.experts[0].out.weight,torch.zeros_like(base.adapter.experts[0].out.weight)))
    def test_combined_variant_gate_ignores_text_gradient_and_entropy(self):
        ad=initialize_adapter(Config(variant='no_route_no_decisiveness',device='cpu'),2).eval()
        args=[torch.randn(1,2,32),torch.randn(2,32),torch.randn(1,64),torch.randn(64),torch.randn(1,8)]
        ad(*args);w=ad.gate.clone();self.assertEqual(float(ad.auxiliary),0.)
        args[3]+=50;ad(*args);torch.testing.assert_close(ad.gate,w,rtol=0,atol=0)
    def test_text_probe_does_not_change_reference_trajectory(self):
        d=toy();c=Config(variant='full',device='cpu');seed_all(7)
        model=Backbone(36,6,2,d_model=32).eval();a=initialize_adapter(c,2);b=copy.deepcopy(a)
        observer=DevelopmentObserver(d,c.seed,True,token_lengths=[20]*len(d.x))
        seed_all(12);x,events,_=Stream(c,d,model,a).run('test',origins=list(range(84,95)))
        seed_all(12);y,events2,_=Stream(c,d,model,b).run('test',origins=list(range(84,95)),observer=observer)
        for key in ['predictions','gate_weights','feedback_origins']:np.testing.assert_array_equal(x[key],y[key])
        self.assertEqual(events,events2)
        for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
        for p in observer.probes:self.assertLess(p['donor'],p['origin'])
        np.testing.assert_array_equal(np.stack([p['predictions'][0] for p in observer.probes]),y['predictions'])
    def test_text_mean_uses_training_prefix_only(self):
        a=toy();b=copy.deepcopy(a);b.embeddings[a.train_end:]+=1000
        x=DevelopmentObserver(a,2023,True,token_lengths=[20]*len(a.x));y=DevelopmentObserver(b,2023,True,token_lengths=[20]*len(b.x))
        np.testing.assert_array_equal(x.mean,y.mean)

if __name__=='__main__':unittest.main()

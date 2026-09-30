import unittest,copy
from dataclasses import replace
import numpy as np
import torch
from iadapter.config import Config
from iadapter.models import Backbone
from iadapter.revision_candidate import initialize_five,DonorBackend
from iadapter.shared_online import InternalBackend,OnlineRunner
from iadapter.util import seed_all
from test_protocol import toy

class CandidateTests(unittest.TestCase):
    def test_equal_five_dimensional_initializer_and_missing_text_zero(self):
        c=Config(variant='text_add',device='cpu');a=initialize_five(c,2);b=initialize_five(c,2)
        self.assertTrue(all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items()))
        self.assertEqual(a.project[0].in_features,5);a.eval()
        self.assertFalse(a.text(torch.ones(1,5),False,'cpu').any())
    def test_donor_probe_leaves_forecasts_updates_and_final_state_identical(self):
        data=toy();data=replace(data,embeddings=data.embeddings[:,:5].copy());c=Config(variant='text_add',device='cpu');seed_all(10);model=Backbone(36,6,2,d_model=32);a=initialize_five(c,2);b=copy.deepcopy(a);origins=list(range(84,99))
        seed_all(16);plain=OnlineRunner(c,data,InternalBackend(c,data,model,a)).run(origins)
        probe=DonorBackend(c,data,model,b,2023);seed_all(16);with_probe=OnlineRunner(c,data,probe).run(origins)
        np.testing.assert_array_equal(plain[0]['predictions'],with_probe[0]['predictions'])
        self.assertEqual(plain[1],with_probe[1]);self.assertTrue(all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items()))
        self.assertTrue(all(p['donor_origin']<data.train_end and p['donor_origin']<p['origin'] for p in probe.probes))
if __name__=='__main__':unittest.main()

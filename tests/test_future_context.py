"""Explicit full-model future-independence and unavailable-text checks."""
import copy,unittest
import numpy as np
import torch
from test_protocol import toy
from iadapter.config import Config
from iadapter.models import Backbone,Adaptation
from iadapter.engine import Stream
from iadapter.util import seed_all

class FutureContextTests(unittest.TestCase):
    def test_full_model_prefix_independent_of_future_labels_and_text(self):
        d=toy(); changed=copy.deepcopy(d); cut=48
        changed.target[cut+1:]+=500
        changed.embeddings[cut+1:]*=-100
        changed.texts[cut+1:]=['Different future text']*(len(d.texts)-cut-1)
        c=Config(variant='full',device='cpu')
        seed_all(71); b=Backbone(36,6,2,d_model=32).eval(); a=Adaptation(c,2)
        initial=copy.deepcopy(a.state_dict())
        origins=list(range(35,49))
        seed_all(17); out,events,_=Stream(c,d,b,a).run('train',origins)
        a2=Adaptation(c,2); a2.load_state_dict(initial)
        seed_all(17); altered,events2,_=Stream(c,changed,b,a2).run('train',origins)
        np.testing.assert_array_equal(out['predictions'],altered['predictions'])
        self.assertEqual(events,events2)
    def test_no_text_projection_is_zero_and_not_content_dependent(self):
        c=Config(variant='full',device='cpu'); a=Adaptation(c,2).eval()
        first=a.text(torch.randn(1,768),False,'cpu')
        second=a.text(torch.randn(1,768)*100,False,'cpu')
        torch.testing.assert_close(first,torch.zeros_like(first),rtol=0,atol=0)
        torch.testing.assert_close(first,second,rtol=0,atol=0)

if __name__=='__main__': unittest.main()

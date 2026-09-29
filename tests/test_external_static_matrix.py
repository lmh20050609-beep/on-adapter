import unittest
from iadapter.config import DOMAINS
from iadapter.external_static import pilot_config
from iadapter.external_static_matrix import groups,reused,verify_core_unchanged

class StaticMatrixTests(unittest.TestCase):
    def test_fixed_matrix_has_exact_pilot_reuse(self):
        jobs=groups();self.assertEqual(len(jobs),25);self.assertEqual(sum(reused(j) for j in jobs),1)
        self.assertEqual({(j['dataset'],j['seed']) for j in jobs},{(d,s) for d in DOMAINS for s in range(2023,2028)})
        for j in jobs:
            c=pilot_config(**j);self.assertEqual(c.seed,j['seed']);self.assertEqual(c.dataset,j['dataset'])
            self.assertEqual((c.horizon,c.d_model,c.e_layers,c.pretrain_batch),(6,32,2,16))
            self.assertEqual(c.lookback,48 if j['dataset']=='Energy' else 36)
    def test_pilot_training_functions_unchanged(self):
        self.assertTrue(verify_core_unchanged()['training_and_evaluation_ast_equal'])

if __name__=='__main__':unittest.main()

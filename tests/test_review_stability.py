import unittest
from iadapter.review_stability import jobs,METHODS,verify_training_unchanged
from iadapter.review_sprint import matrix

class StabilityExtensionTests(unittest.TestCase):
    def test_seed_extension_is_fixed_and_disjoint(self):
        new={(j['dataset'],j['seed'],j['variant']) for j in jobs()}
        old={(j['dataset'],j['seed'],j['variant']) for j in matrix()}
        self.assertEqual(len(new),20);self.assertFalse(new&old)
        self.assertEqual({s for _,s,_ in new},{2026,2027})
        self.assertEqual(len(METHODS),5)
    def test_same_training_implementation_as_original_three_seeds(self):
        self.assertEqual(verify_training_unchanged()['training_matches_commit'][:7],'e214667')

if __name__=='__main__':unittest.main()

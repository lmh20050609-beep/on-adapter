import unittest
import pandas as pd
from iadapter.paired import paired_effects

class PairedEvidenceTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(dataset='Example',horizon=6,group=1,seed=s,variant=v,mse=value+(s-2023)*.1,mae=value/2,valid_elements=12) for s in range(2023,2028) for v,value in [('numeric_add',3.),('text_add',2.),('full',2.5)]])
    def test_text_control_and_effect_direction(self):
        pairs,summary=paired_effects(self.frame(),grouped=True)
        self.assertEqual(len(pairs),15)
        effect=summary[(summary.candidate=='text_add')&(summary.control=='numeric_add')].iloc[0]
        self.assertAlmostEqual(effect.delta_mse_mean,-1.)
        self.assertEqual(effect.mse_improved_seeds,5)
        self.assertEqual(effect.valid_elements_per_seed,12)
    def test_missing_seed_is_not_silently_intersected(self):
        with self.assertRaisesRegex(ValueError,'Incomplete paired seeds'): paired_effects(self.frame().iloc[1:])
    def test_duplicate_runs_rejected(self):
        frame=self.frame()
        with self.assertRaisesRegex(ValueError,'Duplicate'): paired_effects(pd.concat([frame,frame.iloc[:1]]))
    def test_different_valid_counts_rejected(self):
        frame=self.frame(); frame.loc[0,'valid_elements']=11
        with self.assertRaisesRegex(ValueError,'different valid target counts'): paired_effects(frame)

if __name__=='__main__': unittest.main()

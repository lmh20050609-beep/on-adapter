import unittest
from iadapter.task_text_prior import parse_output,rule_label,cache_key
class TaskPriorTests(unittest.TestCase):
    def test_only_enum_outputs_accepted(self):
        self.assertEqual(parse_output(' decrease\n'),('DECREASE',True))
        self.assertEqual(parse_output('INCREASE because prices rise'),('UNKNOWN',False))
    def test_rule_respects_target_and_conflicting_direction(self):
        self.assertEqual(rule_label('Energy','Influenza cases increased sharply.'),'UNKNOWN')
        self.assertEqual(rule_label('Energy','Gasoline prices increased, then decreased.'),'MIXED')
        self.assertEqual(rule_label('SocialGood','The unemployment rate has remained stable.'),'STABLE')
    def test_cache_key_binds_exact_visible_input_and_domain(self):
        self.assertNotEqual(cache_key('Energy','same input'),cache_key('SocialGood','same input'))
        self.assertNotEqual(cache_key('Energy','same input'),cache_key('Energy','same input extra'))
if __name__=='__main__':unittest.main()

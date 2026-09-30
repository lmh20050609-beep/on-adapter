import unittest,json,tempfile,copy
from pathlib import Path
from unittest import mock
from iadapter.task_text_prior import generation_admission,SYSTEM,DOMAIN,LABELS
from iadapter.audit import AuditError
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
class AdmissionTests(unittest.TestCase):
    def test_exploratory_admission_preserves_strict_failure_and_frozen_contract(self):
        proof=dict(status='passed',code=dict(commit='old',dirty=False),prompt=SYSTEM,domain_context=DOMAIN,label_options=LABELS,greedy=True,max_new_tokens=8,repetition_penalty=1.,finite_generation_logits_checked=True,frozen_parameters=True,source={'model':'fixed'})
        quality={'status':'stopped_groundedness'}
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'SOURCE.json').write_text(json.dumps(proof['source']))
            with mock.patch('iadapter.task_text_prior.WEIGHTS',Path(tmp)),mock.patch('iadapter.task_text_prior.revision',return_value=dict(commit='new',dirty=False)):
                with self.assertRaises(AuditError):generation_admission(proof,quality)
                self.assertEqual(generation_admission(proof,quality,True),'user_revised_exploratory_imperfect_prior')
                for field,value in [('prompt','changed'),('max_new_tokens',9),('finite_generation_logits_checked',False),('source',{'model':'other'}),('status','stopped_invalid_format')]:
                    changed=copy.deepcopy(proof);changed[field]=value
                    with self.assertRaises(AuditError):generation_admission(changed,quality,True)
                self.assertEqual(quality['status'],'stopped_groundedness');self.assertEqual(proof['code']['commit'],'old')
if __name__=='__main__':unittest.main()

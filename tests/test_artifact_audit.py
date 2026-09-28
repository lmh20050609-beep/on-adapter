"""Integration checks mutate private copies of a completed real run, never sources."""
import copy,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from iadapter.config import ROOT,job_id
from iadapter.audit import audit_run,original_reference,AuditError
from iadapter.util import metrics
from iadapter import campaign

JOB={'stage':'B','dataset':'Economy','horizon':6,'seed':2023,'variant':'full'}
SOURCE=ROOT/'runs'/job_id(JOB)

@unittest.skipUnless(SOURCE.is_dir(),'Requires existing Economy evidence for mutation-based integration checks')
class EvidenceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.reference=original_reference('Economy')
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='iadapter_audit_test_')
        self.root=Path(self.temp.name); self.run=self.root/'runs'/job_id(JOB)
        shutil.copytree(SOURCE,self.run)
    def tearDown(self): self.temp.cleanup()
    def verify(self): return audit_run(self.run,JOB,self.reference,check_checkpoint=False)
    def change_arrays(self,mutation):
        path=self.run/'predictions.npz'
        with np.load(path) as z: arrays={k:z[k].copy() for k in z.files}
        mutation(arrays); np.savez_compressed(path,**arrays)
        # Make the falsified saved-array metrics self-consistent. A verifier
        # that only trusts its own saved targets would incorrectly accept this.
        p=self.run/'metrics.json'; result=json.loads(p.read_text())
        result.update(metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask']))
        p.write_text(json.dumps(result))
    def test_real_run_matches_original_csv(self):
        self.assertEqual(self.verify()['forecast_origins'],80)
    def test_changed_target_rejected_even_with_matching_metrics(self):
        self.change_arrays(lambda a:a['targets'].__setitem__((0,0,0),a['targets'][0,0,0]+1))
        with self.assertRaisesRegex(AuditError,'targets differ from original'): self.verify()
    def test_changed_mask_rejected_even_with_matching_metrics(self):
        self.change_arrays(lambda a:a['valid_mask'].__setitem__((0,0,0),False))
        with self.assertRaisesRegex(AuditError,'validity mask'): self.verify()
    def test_changed_origin_rejected(self):
        self.change_arrays(lambda a:a['origins'].__setitem__(0,a['origins'][0]+1))
        with self.assertRaisesRegex(AuditError,'Forecast origins'): self.verify()
    def test_missing_file_rejected(self):
        (self.run/'predictions.npz').unlink()
        with self.assertRaisesRegex(AuditError,'Missing required file'): self.verify()
    def test_config_seed_mismatch_rejected(self):
        p=self.run/'config.json'; d=json.loads(p.read_text()); d['seed']=2024; p.write_text(json.dumps(d))
        with self.assertRaisesRegex(AuditError,'configuration mismatch: seed'): self.verify()
    def test_future_feedback_rejected(self):
        self.change_arrays(lambda a:a['feedback_origins'].__setitem__((0,0),500))
        with self.assertRaisesRegex(AuditError,'Feedback ages'): self.verify()
    def state(self,status='completed'):
        d={'code':{'commit':'old','dirty':False},'finished_at':1234,'jobs':{job_id(JOB):dict(JOB,status=status)}}
        path=self.root/'queue_state.json'; path.write_text(json.dumps(d)); return path
    def test_completed_queue_different_head_is_read_only(self):
        path=self.state(); before=path.read_bytes()
        with patch.object(campaign,'ROOT',self.root),patch.object(campaign,'matrix',return_value=[JOB]),patch.object(campaign,'run_one') as launch,patch.object(campaign,'revision',return_value={'commit':'new','dirty':True}):
            result=campaign.main(); self.assertEqual(result['passed'],1); launch.assert_not_called()
        self.assertEqual(before,path.read_bytes())
    def test_incomplete_queue_different_head_still_rejected(self):
        self.state('pending')
        with patch.object(campaign,'ROOT',self.root),patch.object(campaign,'matrix',return_value=[JOB]),patch.object(campaign,'run_one') as launch,patch.object(campaign,'revision',return_value={'commit':'new','dirty':False}):
            with self.assertRaisesRegex(RuntimeError,'another code revision'): campaign.main()
            launch.assert_not_called()
    def test_status_detects_missing_artifact(self):
        path=self.state(); (self.run/'metrics.json').unlink()
        result=campaign.queue_status(json.loads(path.read_text()),[JOB],self.root)
        self.assertEqual(result['status'],'artifact_error')

class AnalysisGateTests(unittest.TestCase):
    def test_failed_source_audit_prevents_report_generation(self):
        from iadapter import analyze
        with patch.object(analyze,'audit_all',return_value={'status':'failed','failures':['changed raw target']}),patch.object(analyze,'write_json') as write:
            with self.assertRaisesRegex(AuditError,'reports were not regenerated'):
                analyze.main()
            write.assert_not_called()

if __name__=='__main__': unittest.main()

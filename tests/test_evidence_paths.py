import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from iadapter.evidence_paths import resolve_evidence_path

class EvidencePathTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'docs').mkdir();(self.root/'data').mkdir()
        self.target=self.root/'data/example.csv';self.target.write_bytes(b'x\n1\n')
        manifest={'file_mappings':{'/removed/project/data/example.csv':{'relative_path':'data/example.csv','sha256':hashlib.sha256(self.target.read_bytes()).hexdigest()}}}
        (self.root/'docs/PATH_MIGRATION.json').write_text(json.dumps(manifest))
    def tearDown(self):self.temp.cleanup()
    def test_verified_old_path_resolves_and_current_path_stays_current(self):
        self.assertEqual(resolve_evidence_path('/removed/project/data/example.csv',self.root),self.target)
        self.assertEqual(resolve_evidence_path(self.target,self.root),self.target)
    def test_changed_replacement_cannot_impersonate_original(self):
        self.target.write_bytes(b'x\n2\n')
        with self.assertRaisesRegex(ValueError,'differs from verified original'):
            resolve_evidence_path('/removed/project/data/example.csv',self.root)
    def test_missing_replacement_rejected(self):
        self.target.unlink()
        with self.assertRaisesRegex(FileNotFoundError,'Missing migrated evidence'):
            resolve_evidence_path('/removed/project/data/example.csv',self.root)
    def test_ready_bert_weights_do_not_require_deleted_archive(self):
        from iadapter import prepare
        target=self.root/'weights/bert-base-uncased';target.mkdir(parents=True)
        for name in ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt']:
            (target/name).write_bytes(b'presence-only-test')
        with patch.object(prepare,'ROOT',self.root),patch.object(prepare,'BERT_ARCHIVE',self.root/'missing.zip'),patch.object(prepare.zipfile,'ZipFile') as archive:
            self.assertEqual(prepare.weight_directory(),target)
            archive.assert_not_called()

if __name__=='__main__':unittest.main()

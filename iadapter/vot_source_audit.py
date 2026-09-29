"""Read-only inventory and executable missing-prior-path witness, no model import."""
import ast,json,os
from collections import Counter
from types import SimpleNamespace
from .config import ROOT
from .util import write_json

def main():
    source=ROOT/'external/vot';out=ROOT/'refine-logs/external_baselines'
    manifest=json.loads((source/'SOURCE.json').read_text())
    assert manifest['commit']=='f769621da9efde3e82a1975f478efdec83dcbd10'
    inventory=[]
    for path in sorted((source/'data/stat_predictions_rag').rglob('*.json')):
        values=json.loads(path.read_text());keys=sorted(map(int,values))
        inventory.append({'file':str(path.relative_to(source)),'records':len(keys),'min_id':min(keys),'max_id':max(keys),'contiguous_ids':keys==list(range(len(keys))),'record_fields':sorted({k for v in values.values() for k in v}),'prediction_length_counts':dict(Counter(len(v['Prediction']) for v in values.values()))})
    tree=ast.parse((source/'data_provider/data_loader.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Dataset_Custom')
    function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_text_pred')
    isolated=ast.Module(body=[function],type_ignores=[]);ns={'os':os,'json':json}
    exec(compile(isolated,'upstream_get_text_pred','exec'),ns)
    error=None
    assert not (source/'data/stat_predictions_ecnu_rag').exists()
    try:ns['get_text_pred'](SimpleNamespace(root_path=str(source/'data'),data_path='SocialGood.csv'))
    except UnboundLocalError as e:error=str(e)
    assert error and 'text_pred_json' in error,'Expected missing-path failure not reproduced'
    result={'upstream_commit':manifest['commit'],'status':'source_audit_complete_not_benchmark_ready','prior_files':inventory,'loader_expected_directory':'data/stat_predictions_ecnu_rag','published_directory':'data/stat_predictions_rag','missing_directory_witness':{'exception':'UnboundLocalError','message':error},'generation_inputs_referenced_by_data_add_present':{'integrated_predictions_stat/stage1_train_optimized':(source/'integrated_predictions_stat/stage1_train_optimized').exists(),'integrated_predictions_stat/stage2_final_predictions':(source/'integrated_predictions_stat/stage2_final_predictions').exists()},'prior_values_used_for_training_or_scoring':False,'limits':'Absent origin/time/generation metadata prevents availability verification; this does not prove target leakage or invalidity of the published method.'}
    write_json(out/'vot_prior_audit.json',result)
    print('VOT_PRIOR_AUDIT_COMPLETE',len(inventory),'files; original loader path failure reproduced')

if __name__=='__main__':main()

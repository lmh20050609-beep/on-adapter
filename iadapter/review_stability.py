"""Predeclared two-seed extension of five fixed development candidates."""
import argparse,ast,subprocess,json
from pathlib import Path
from .config import ROOT
from .review_sprint import campaign
from .review_sprint_analysis import analyze
SOURCE_COMMIT='e2146677640df0e85354722d5ca61a32f12520f3'
OUTPUT=ROOT/'experiments/review_stability_20260929'
METHODS=['numeric_add','text_add','full','no_decisiveness','single_mlp']

def jobs():return [dict(dataset=d,seed=s,variant=v) for d in ['Energy','SocialGood'] for s in [2026,2027] for v in METHODS]

def verify_training_unchanged():
    for name in ['config.py','models.py','engine.py','data.py','run.py','util.py']:
        file='iadapter/'+name
        original=subprocess.check_output(['git','show',SOURCE_COMMIT+':'+file],cwd=ROOT,text=True)
        if ast.dump(ast.parse(original))!=ast.dump(ast.parse((ROOT/file).read_text())):
            raise ValueError('Training module changed since initial seeds: '+file)
    original=subprocess.check_output(['git','show',SOURCE_COMMIT+':iadapter/review_sprint.py'],cwd=ROOT,text=True)
    def bodies(text):return {n.name:ast.dump(n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before,after=bodies(original),bodies((ROOT/'iadapter/review_sprint.py').read_text())
    for name in ['development_data','truncate_development','initialize_adapter','DevelopmentObserver','write_events','run_cell']:
        if before[name]!=after[name]:raise ValueError('Development training implementation changed: '+name)
    return {'training_matches_commit':SOURCE_COMMIT,'method':'AST equality of model/data/stream/training modules and development cell functions; queue/report changes allowed'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run_dir',type=Path,default=OUTPUT);p.add_argument('--analyze',action='store_true');a=p.parse_args()
    proof=verify_training_unchanged()
    if a.analyze:
        analyze(a.run_dir,jobs_override=jobs(),factorial=False)
        (a.run_dir/'reports/training_equivalence.json').write_text(json.dumps(proof,indent=2))
    else:campaign(a.run_dir,jobs_override=jobs())

"""Read-only derivation retaining all static controls alongside online extensions."""
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import pandas as pd
from .analyze import md_table

ROOT = Path(__file__).resolve().parents[1]

def main():
    source = ROOT / 'experiments/external_online_20260929/reports'
    out = ROOT / 'refine-logs/external_baselines/online_evidence'
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(source / 'runs.csv')
    proof = json.loads((source / 'validation.json').read_text())
    assert proof['status'] == 'passed' and proof['completed'] == 75
    assert len(runs) == 75 and not runs.duplicated(['dataset', 'seed', 'method']).any()
    assert set(runs.seed) == set(range(2023, 2028))
    all_rows = []
    for row in runs.to_dict('records'):
        for mode in ['online', 'static']:
            path = Path(row['source' if mode == 'online' else 'static_source'])
            from .util import metrics
            with np.load(path / 'predictions.npz') as z:
                scores = metrics(z['predictions'], z['targets'], z['valid_mask'])
            for metric in ['mse', 'mae']:
                assert abs(scores[metric] - row[metric if mode == 'online' else 'static_' + metric]) < 1e-12
            all_rows.append(dict(dataset=row['dataset'], seed=row['seed'], method=row['method' if mode == 'online' else 'static_method'], mode=mode, mse=scores['mse'], mae=scores['mae'], source=str(path)))
    all_runs = pd.DataFrame(all_rows)
    assert len(all_runs) == 150 and not all_runs.duplicated(['dataset', 'seed', 'method']).any()
    all_runs.to_csv(out / 'all_static_online_runs.csv', index=False)
    summary = all_runs.groupby(['dataset','method'], sort=False).agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std')).reset_index()
    summary.to_csv(out / 'all_static_online_summary.csv', index=False)
    paired=[]
    for (dataset, seed), block in all_runs.groupby(['dataset','seed']):
        indexed=block.set_index('method'); candidate=indexed.loc['iadapter_online']
        for control in indexed.index:
            if control == 'iadapter_online': continue
            reference=indexed.loc[control]
            paired.append(dict(dataset=dataset,seed=seed,candidate='iadapter_online',control=control,delta_mse=candidate.mse-reference.mse,delta_mae=candidate.mae-reference.mae))
    pairs=pd.DataFrame(paired);pairs.to_csv(out/'all_controls_paired_runs.csv',index=False)
    contrasts=pairs.groupby(['dataset','control'],sort=False).agg(n=('seed','count'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),delta_mae_mean=('delta_mae','mean'),delta_mae_sd=('delta_mae','std'),mse_improved_seeds=('delta_mse',lambda x:int((x<0).sum())),mae_improved_seeds=('delta_mae',lambda x:int((x<0).sum()))).reset_index()
    contrasts.to_csv(out/'all_controls_paired_summary.csv',index=False)
    hashes={}
    for p in source.iterdir():
        if p.is_file():
            shutil.copy2(p,out/p.name)
            hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    metadata=dict(status='passed',online_cells=75,static_cells=75,unique_cells=150,source_code=proof['source_code'],copied_source_sha256=hashes,scope='OT H6 development, five optimization seeds; descriptive, not independent-data confirmation',validation='Recomputed all 150 saved-prediction metrics; original CSV/maturity audit is separately preserved in validation.json')
    (out/'derivation_validation.json').write_text(json.dumps(metadata,indent=2))
    text='# Static and online results retained together\n\nAll six settings are shown; the degraded GPT4MTS online extension never replaces its static reference. Mean and sample SD over five seeds on the fixed development stream. No pooled cross-protocol win count or significance claim.\n\n'+md_table(summary.round(6))+'\n\n## iAdapter online minus each control\n\nNegative differences favor iAdapter. Cross-model seed labels align repetitions but do not imply shared parameters.\n\n'+md_table(contrasts.round(6))+'\n'
    (out/'ALL_CONTROLS.md').write_text(text)
    print(json.dumps(metadata,indent=2))

if __name__ == '__main__':
    main()

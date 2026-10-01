"""Post-results forecast geometry diagnostic; oracle quantities are not forecasts."""
from pathlib import Path
import sys,json,hashlib,traceback
import numpy as np
import pandas as pd

ROOT=Path('/home/wuwa/projects/research/iadapter')
sys.path.insert(0,str(ROOT))
from iadapter.audit import original_reference,same,require
from iadapter.util import metrics,revision
from iadapter.data import load_data

OUTPUT=Path('/mnt/c/Users/18452/Documents/ChatGPT/FedNoise/output/iadapter-innovation-first-principles-20261001/diagnostics_retry1')

def load(path):
    require(json.loads((path.parent/'status.json').read_text())['status']=='completed','Incomplete source')
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in ['origins','predictions','targets','valid_mask']}

def paths(dataset,seed):
    if dataset in ['Energy','SocialGood']:
        base=ROOT/f'experiments/replan_20260930/exploratory_fusion/{dataset}_s{seed}_LLM5'
        n=ROOT/f'experiments/external_online_20260929/groups/{dataset}_s{seed}/numeric_online/predictions.npz'
        return n,{f'text_{m}':base/m/'predictions.npz' for m in ['static','online']}
    base=ROOT/f'experiments/extension_20261001/groups/{dataset}_s{seed}'
    return base/'numeric_online/predictions.npz',{f'text_{m}':base/f'LLM5_{m}'/'predictions.npz' for m in ['static','online']}

def corr(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None

def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    (OUTPUT/'status.json').write_text(json.dumps({'status':'running'}))
    rows=[];lead=[];sources=[]
    for dataset in ['Energy','SocialGood','Agriculture','Health']:
        ref=original_reference(dataset);k=ref['columns'].index('OT')
        data=load_data(dataset)
        for seed in range(2023,2026 if dataset in ['Energy','SocialGood'] else 2028):
            numeric_path,text_paths=paths(dataset,seed);n=load(numeric_path)
            origins=n['origins'];h=n['predictions'].shape[1]
            truth=np.stack([ref['target'][t+1:t+h+1,k:k+1] for t in origins]);valid=np.stack([ref['valid'][t+1:t+h+1,k:k+1] for t in origins])
            same(n['targets'],truth,'Original CSV target');same(n['valid_mask'],valid,'Original CSV mask',exact=True)
            avail=np.asarray([bool(data.texts[t]) for t in origins])[:,None,None]
            for branch,path in text_paths.items():
                m=load(path)
                for key in ['origins','targets','valid_mask']:same(n[key],m[key],'Matched '+key,exact=True)
                a,b=n['predictions'].astype(np.float64),m['predictions'].astype(np.float64)
                delta=b-a;r=truth.astype(np.float64)-a
                ns,ts=metrics(a,truth,valid),metrics(b,truth,valid)
                # Static equal weights need no future target; no-text emits numeric only.
                equal=a+.5*delta*avail
                es=metrics(equal,truth,valid)
                observed=valid&avail
                A=float((r*delta)[observed].sum());B=float((delta**2)[observed].sum())
                oracle=float(np.clip(A/B,0,1)) if B else 0.
                os=metrics(a+oracle*delta*avail,truth,valid)
                lead_prediction=a.copy()
                for j in range(h):
                    mask=observed[:,j,:];rh=r[:,j,:][mask];dh=delta[:,j,:][mask]
                    aa=float(np.clip(np.dot(rh,dh)/np.dot(dh,dh),0,1)) if len(dh) and np.dot(dh,dh)>0 else 0.
                    lead_prediction[:,j,:]=a[:,j,:]+aa*delta[:,j,:]*avail[:,0,:]
                    vm=valid[:,j,:]
                    nlead=float(np.mean((truth[:,j,:]-a[:,j,:])[vm]**2))
                    mlead=float(np.mean((truth[:,j,:]-b[:,j,:])[vm]**2))
                    elead=float(np.mean((truth[:,j,:]-equal[:,j,:])[vm]**2))
                    lead.append(dict(dataset=dataset,seed=seed,branch=branch,horizon=j+1,numeric_mse=nlead,text_mse=mlead,equal_mse=elead,oracle_hindsight_alpha=aa,content_available_valid_cells=int(mask.sum())))
                ls=metrics(lead_prediction,truth,valid)
                rows.append(dict(dataset=dataset,seed=seed,branch=branch,numeric_mse=ns['mse'],text_mse=ts['mse'],equal_mse=es['mse'],numeric_mae=ns['mae'],text_mae=ts['mae'],equal_mae=es['mae'],equal_minus_numeric=es['mse']-ns['mse'],equal_minus_text=es['mse']-ts['mse'],oracle_hindsight_alpha=oracle,oracle_hindsight_mse=os['mse'],oracle_per_h_hindsight_mse=ls['mse'],error_correlation=corr(r[valid],(truth-b)[valid]),text_better_cell_fraction=float(np.mean(((truth-b)**2)[valid]<(r**2)[valid])),valid_cells=int(valid.sum()),available_origins=int(avail.sum()),forecast_origins=len(origins)))
                for p in [numeric_path,path]:sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    frame=pd.DataFrame(rows);frame.to_csv(OUTPUT/'runs.csv',index=False);pd.DataFrame(lead).to_csv(OUTPUT/'horizons.csv',index=False)
    summary=frame.groupby(['dataset','branch']).agg(n=('seed','count'),numeric_mse=('numeric_mse','mean'),text_mse=('text_mse','mean'),equal_mse=('equal_mse','mean'),equal_delta_numeric=('equal_minus_numeric','mean'),equal_delta_sd=('equal_minus_numeric','std'),equal_delta_text=('equal_minus_text','mean'),equal_improved_seeds=('equal_minus_numeric',lambda x:int((x<0).sum())),oracle_per_h_hindsight_mse=('oracle_per_h_hindsight_mse','mean')).reset_index()
    summary.to_csv(OUTPUT/'summary.csv',index=False)
    pd.DataFrame(sources).drop_duplicates().to_csv(OUTPUT/'sources.csv',index=False)
    proof=dict(status='passed',matched_sequences=len(frame),new_neural_evaluations=0,scope='Post-results diagnostic on original issued forecasts; uniform blend rule fixed before this script execution; no test hyperparameter selection',oracle_not_deployable=True,alpha_initialization_or_calibration_not_tested=True,no_text_rule='Numeric forecast only',original_target_masks_checked=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),code=revision())
    from iadapter.analyze import md_table
    report='# 文本路径互补性：第一性原理诊断\n\n已有结果之后的探索性分析。没有新神经模型训练。所有预测来自当时发出的原预测，原点、原 CSV 真值与掩码逐一核验。数值控制是 numeric-online，文本分支的 static/online 分开。无文本原点的等权组合直接输出数值预测。\n\n'+md_table(summary)+'\n\n等权为固定 0.5，不用真值决定当前预测。oracle_per_h_hindsight_mse 和 oracle_hindsight_alpha 用完整计分目标计算，仅描述可达到的混合空间，绝不是在线性能。其余系数算法与合法校准先验尚未执行。Energy/SocialGood 使用已经查看的开发片段；Agriculture/Health 使用刚完成的固定扩域片段，此分析本身仍是后见探索。\n'
    (OUTPUT/'RESULTS.md').write_text(report,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    horizon=pd.DataFrame(lead)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.8),layout='constrained')
    for ax,dataset in zip(axes.flat,['Energy','SocialGood','Agriculture','Health']):
        for branch,color in [('text_static','#0072B2'),('text_online','#D55E00')]:
            g=horizon[(horizon.dataset==dataset)&(horizon.branch==branch)].copy()
            g['change']=100*(g.equal_mse/g.numeric_mse-1)
            group=g.groupby('horizon')['change'];mean=group.mean();sd=group.std()
            ax.plot(mean.index,mean.values,color=color,label=branch.replace('_',' '),marker='o',markersize=3)
            ax.fill_between(mean.index,mean-sd,mean+sd,color=color,alpha=.12,linewidth=0)
        ax.axhline(0,color='#555555',lw=.8,linestyle='--');ax.set_title(dataset)
        ax.set_xlabel('Forecast lead (reported rows)');ax.set_ylabel('Equal blend vs numeric online (%)')
        ax.set_xticks(range(1,7));ax.grid(axis='y',alpha=.15)
    axes.flat[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Fixed 0.5 blend: exploratory issued-forecast replay; shade = seed SD',fontsize=11)
    fig.savefig(OUTPUT/'equal_blend_by_horizon.png',dpi=220);fig.savefig(OUTPUT/'equal_blend_by_horizon.pdf');plt.close(fig)
    (OUTPUT/'validation.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    (OUTPUT/'status.json').write_text(json.dumps({'status':'completed'}))
    print(summary.to_string(index=False))

if __name__=='__main__':
    try:main()
    except Exception:
        if OUTPUT.exists():(OUTPUT/'status.json').write_text(json.dumps({'status':'failed','error':traceback.format_exc()}))
        raise

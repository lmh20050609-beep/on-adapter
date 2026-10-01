"""Recompute all pilot scores from original CSV targets and issued predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .calibration_pilot import OUT,jobs,key,full_ot,audit_arrays,REPRESENTATIONS,SCENARIOS,pair_ledger
from .util import metrics,write_json
from .audit import require
from .analyze import md_table

def analyze(base=OUT):
    base=Path(base);out=base/'reports'
    state=json.loads((base/'assessment_queue_state.json').read_text());require(all(v=='completed' for v in state['jobs'].values()),'Incomplete assessment')
    out.mkdir(exist_ok=False);write_json(out/'status.json',dict(status='running'))
    rows=[];branches=[];pairs=[];content=[]
    for j in jobs():
        data=full_ot(j['dataset']);g=base/'groups'/key(j)/'assessment'
        for name in ['numeric_static','numeric_online','LLM5_static','LLM5_online','raw5_static','raw5_online','full_online']:
            d=g/name;require(json.loads((d/'status.json').read_text())['status']=='completed','Incomplete branch')
            saved=json.loads((d/'metrics.json').read_text())
            with np.load(d/'predictions.npz',allow_pickle=False) as z:
                arrays={k:z[k].copy() for k in z.files};score=audit_arrays(j,data,'assessment',arrays)
            for m in ['mse','mae']:require(score[m]==saved[m],'Branch saved score differs')
            branches.append(dict(**j,method=name,**score))
            p=d/'content_probes.npz'
            if p.exists():
                with np.load(p,allow_pickle=False) as z:
                    require((z['donor_origins']<data.train_end).all(),'Future donor')
                    for condition in ['historical_donor','zero_current_text']:
                        s=metrics(z[condition],arrays['targets'],arrays['valid_mask'])
                        content.append(dict(**j,method=name,condition=condition,delta_mse=s['mse']-score['mse'],delta_mae=s['mae']-score['mae'],**s))
        for rep in REPRESENTATIONS:
            for nm,tm in SCENARIOS:
                ledger=pair_ledger(j,'assessment',rep,nm,tm,base)
                control={name:metrics(ledger[k],ledger['targets'],ledger['valid_mask']) for name,k in [('numeric','numeric'),('text','text')]}
                root=base/'mixes'/key(j)/rep/f'numeric_{nm}_text_{tm}';local={}
                for family in ['equal','fixed','ls','shared_ls','endpoint_hedge','mixed_hedge']:
                    d=root/family;require(json.loads((d/'status.json').read_text())['status']=='completed','Incomplete mix')
                    saved=json.loads((d/'metrics.json').read_text())
                    with np.load(d/'predictions.npz',allow_pickle=False) as z:
                        arrays={k:z[k].copy() for k in z.files};score=audit_arrays(j,data,'assessment',arrays)
                        require(np.all((arrays['alpha']>=0)&(arrays['alpha']<=1)),'Invalid convex coefficient')
                    for m in ['mse','mae']:require(score[m]==saved[m],'Mix saved score differs')
                    events=[json.loads(line) for line in (d/'updates.jsonl').read_text().splitlines()]
                    require([e['origin'] for e in events]==ledger['origins'][:-6].tolist(),'Mix maturity consumption differs')
                    require(all(e['mature_at']==e['origin']+6==e['update_at'] for e in events),'Premature mix update')
                    local[family]=score;rows.append(dict(**j,representation=rep,numeric_mode=nm,text_mode=tm,family=family,**score))
                for family,score in local.items():
                    for name,c in {**control,'equal':local['equal'],'fixed':local['fixed'],'mixed_hedge':local['mixed_hedge']}.items():
                        if name==family:continue
                        pairs.append(dict(**j,representation=rep,numeric_mode=nm,text_mode=tm,family=family,control=name,
                            delta_mse=score['mse']-c['mse'],delta_mae=score['mae']-c['mae'],candidate_mse=score['mse'],control_mse=c['mse'],candidate_mae=score['mae'],control_mae=c['mae']))
    frame=pd.DataFrame(rows);bf=pd.DataFrame(branches);cf=pd.DataFrame(content);pf=pd.DataFrame(pairs)
    frame.to_csv(out/'runs.csv',index=False);bf.to_csv(out/'branches.csv',index=False);cf.to_csv(out/'content_runs.csv',index=False);pf.to_csv(out/'paired_runs.csv',index=False)
    summary=frame.groupby(['dataset','representation','numeric_mode','text_mode','family']).agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std')).reset_index();summary.to_csv(out/'summary.csv',index=False)
    bs=bf.groupby(['dataset','method']).agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std')).reset_index();bs.to_csv(out/'branch_summary.csv',index=False)
    ps=pf.groupby(['dataset','representation','numeric_mode','text_mode','family','control']).agg(n=('seed','count'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),delta_mae_mean=('delta_mae','mean'),delta_mae_sd=('delta_mae','std'),candidate_mse=('candidate_mse','mean'),control_mse=('control_mse','mean'),candidate_mae=('candidate_mae','mean'),control_mae=('control_mae','mean'),mse_improved_seeds=('delta_mse',lambda s:int((s<0).sum()))).reset_index()
    ps['mse_change_percent']=100*(ps.candidate_mse/ps.control_mse-1);ps['mae_change_percent']=100*(ps.candidate_mae/ps.control_mae-1);ps.to_csv(out/'paired_summary.csv',index=False)
    cf.groupby(['dataset','method','condition']).agg(n=('seed','count'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),delta_mae_mean=('delta_mae','mean'),delta_mae_sd=('delta_mae','std')).reset_index().to_csv(out/'content_summary.csv',index=False)
    primary=summary[(summary.representation=='LLM5')&(summary.numeric_mode=='online')&(summary.text_mode=='online')]
    report='# 成熟反馈校准：已见数据后的探索性结果\n\n60/10/5/5/20，两个域，三个seed，OT/H6。96条神经分支预测流（校准18、选择36、评估42），复用已有训练权重；组合器在同一账本上形成216条评估序列，不计作216次神经训练。所有分数由原CSV真值/掩码重算。规则只在选择片段锁定，之后才生成评估预测。已有时期被研究者查看，不声称未见数据确认。\n\n## 分支对照\n\n'+md_table(bs)+'\n\n## 主要组合场景：numeric-online + LLM5-online\n\n'+md_table(primary)+'\n\n全部raw5、静态文本和静态数值场景保存在summary.csv，不以最好场景替代主要场景。SD为相同时间流上的seed标准差；改进seed数不等于显著性。当前内容donor/zero为同一在线状态的预测探针，原文本学习路径不变，不是重新训练/部署政策，也不能证明语义因果。\n'
    (out/'RESULTS.md').write_text(report,encoding='utf-8')
    print(primary.to_string(index=False),flush=True)
    plots(base,out)
    write_json(out/'validation.json',dict(status='passed',neural_forecast_streams=96,assessment_branch_sequences=len(branches),mix_assessment_sequences=len(rows),original_csv_masks_checked=True,once_only_maturity_checked=True,all_failed_runs_excluded=True,scope='Exploratory known-period repartition, 3 optimization seeds; no significance/causal/general-SOTA claim'))
    write_json(out/'status.json',dict(status='completed'))

def plots(base,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for d in ['Energy','SocialGood']:
        j=dict(dataset=d,seed=2023);data=full_ot(d)
        ledger=pair_ledger(j,'assessment','LLM5','online','online',base)
        p=base/'mixes'/key(j)/'LLM5/numeric_online_text_online/ls/predictions.npz'
        with np.load(p,allow_pickle=False) as z:pred=z['predictions'].copy();alpha=z['alpha'].copy();origins=z['origins'].copy()
        observed=ledger['valid_mask'][:,0,0];truth=np.where(observed,ledger['targets'][:,0,0]*data.scale[0]+data.mean[0],np.nan)
        fig,axes=plt.subplots(2,1,figsize=(9,5.3),sharex=True,layout='constrained')
        axes[0].plot(origins,truth,color='#222222',lw=1.3,label='Observed target (lead 1)')
        axes[0].plot(origins,ledger['numeric'][:,0,0]*data.scale[0]+data.mean[0],color='#777777',lw=1,label='Numeric online')
        axes[0].plot(origins,pred[:,0,0]*data.scale[0]+data.mean[0],color='#0072B2',lw=1,label='LS mix')
        axes[0].set_ylabel('Original target units');axes[0].legend(frameon=False,ncol=3,fontsize=8)
        for h in range(6):axes[1].plot(origins,alpha[:,h],lw=1,label=f'Lead {h+1}')
        axes[1].set_ylim(-.02,1.02);axes[1].set_ylabel('Applied text weight');axes[1].set_xlabel('Forecast origin (reported row index)')
        axes[1].legend(frameon=False,ncol=6,fontsize=8)
        first=int(origins[0])+6
        for ax in axes:ax.axvline(first,color='#777777',lw=.8,ls='--');ax.grid(axis='y',alpha=.15)
        fig.suptitle(f'{d}: fixed seed 2023; forecast before mature update; exploratory',fontsize=11)
        fig.savefig(out/f'{d}_alpha_forecast.png',dpi=220);fig.savefig(out/f'{d}_alpha_forecast.pdf');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=OUT);a=p.parse_args();analyze(a.base)

"""Validate all planned raw predictions, summarize matched runs and make figures."""
import json,csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import ROOT,matrix,job_id,MAIN,DOMAINS
from .util import metrics,write_json
from .data import load_data

LABEL={'numeric_add':'Numeric adapter','text_add':'Text addition','full':'iAdapter','no_text_gradient':'No text gradients','no_route_text_gradient':'No routing text gradients','no_decisiveness':'No decisiveness','homogeneous':'Five MLP experts','no_text':'No text (same MoE)','no_numeric_gradient':'No numeric gradients','no_distribution':'No distribution descriptor','uniform':'Uniform fusion'}
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','savefig.dpi':300,'pdf.fonttype':42})

def md_table(frame):
    cols=list(frame.columns); lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for row in frame.itertuples(index=False,name=None): lines.append('| '+' | '.join(str(x) for x in row)+' |')
    return '\n'.join(lines)

def savefig(fig,path):
    fig.savefig(path.with_suffix('.png')); fig.savefig(path.with_suffix('.pdf')); plt.close(fig)

def main_figure(pairs,figs):
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,baseline,color in zip(axes,['numeric_add','text_add'],['#0072B2','#D55E00']):
        for i,dataset in enumerate(DOMAINS):
            values=pairs[(pairs.dataset==dataset)&(pairs.horizon==6)&(pairs.variant==baseline)].mse_reduction_percent.to_numpy()
            assert len(values)==5
            ax.scatter(values,np.full(5,i)+np.linspace(-.12,.12,5),color=color,alpha=.55,s=24)
            ax.errorbar(values.mean(),i,xerr=values.std(ddof=1),fmt='D',color='black',capsize=4,markersize=4)
        ax.axvline(0,color='.5',lw=.8); ax.set_xlabel(f'MSE reduction vs {LABEL[baseline]} (%)'); ax.set_yticks(range(5),DOMAINS); ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis(); fig.tight_layout(); savefig(fig,figs/'main_paired_effects')

def main():
    out=ROOT/'reports'; figs=out/'figures'; figs.mkdir(parents=True,exist_ok=True)
    rows=[]; refs={}; stream_refs={}; states=[]; routing=[]
    for job in matrix():
        directory=ROOT/'runs'/job_id(job)
        assert json.loads((directory/'status.json').read_text())['status']=='completed',job
        saved=json.loads((directory/'metrics.json').read_text())
        assert saved['config']['variant']==job['variant'] and not saved['code']['dirty']
        key=(job['dataset'],job['horizon'],job['seed'])
        with np.load(directory/'predictions.npz',allow_pickle=False) as z:
            pred=z['predictions']; y=z['targets']; mask=z['valid_mask']; origin=z['origins']
            recalculated=metrics(pred,y,mask)
            np.testing.assert_allclose([recalculated['mse'],recalculated['mae']],[saved['mse'],saved['mae']],rtol=0,atol=1e-12)
            if key in refs:
                r=refs[key]
                np.testing.assert_array_equal(origin,r[0]); np.testing.assert_array_equal(y,r[1]); np.testing.assert_array_equal(mask,r[2]); assert saved['backbone_checkpoint']==r[3]
            else: refs[key]=(origin.copy(),y.copy(),mask.copy(),saved['backbone_checkpoint'])
            stream_key=(job['dataset'],job['horizon'])
            if stream_key in stream_refs:
                for a,b in zip((origin,y,mask),stream_refs[stream_key]): np.testing.assert_array_equal(a,b)
            else: stream_refs[stream_key]=(origin.copy(),y.copy(),mask.copy())
            for group in range(3):
                selected=z['state_change_group']==group
                if selected.any() and mask[selected].any():
                    states.append(dict(job,group=group,**metrics(pred[selected],y[selected],mask[selected])))
            weights=z['gate_weights']
            if np.isfinite(weights).all():
                effective=1/(weights.astype(np.float64)**2).sum(-1)
                routing.append(dict(job,effective_experts_mean=float(effective.mean()),effective_experts_sd_over_origins_variables=float(effective.std()),**{f'expert_{i}_mean_weight':float(weights[:,:,i].mean()) for i in range(5)}))
        updates=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines()]
        assert [e['origin'] for e in updates]==[int(t) for t in origin if t+job['horizon']<=origin[-1]]
        assert all(e['origin']==e['text_origin']==e['state_origin'] and e['mature_at']<=e['update_at'] for e in updates)
        rows.append(dict(job,**recalculated,seconds=saved['seconds'],trainable_parameters=saved['trainable_parameters'],commit=saved['code']['commit'],run_dir=str(directory)))
    frame=pd.DataFrame(rows); frame.to_csv(out/'runs.csv',index=False)
    pd.DataFrame(routing).to_csv(out/'routing_summary.csv',index=False)
    assert len(frame)==190 and len(set(frame.commit))==1
    grouped=frame.groupby(['dataset','horizon','variant'],sort=False)
    summary=grouped.agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    paired=[]
    for (dataset,H),block in frame.groupby(['dataset','horizon']):
        full=block[block.variant=='full'].set_index('seed')
        for variant in block.variant.unique():
            if variant=='full': continue
            other=block[block.variant==variant].set_index('seed')
            for seed in full.index.intersection(other.index):
                paired.append(dict(dataset=dataset,horizon=int(H),variant=variant,seed=int(seed),delta_mse=float(full.loc[seed,'mse']-other.loc[seed,'mse']),delta_mae=float(full.loc[seed,'mae']-other.loc[seed,'mae']),mse_reduction_percent=float(100*(other.loc[seed,'mse']-full.loc[seed,'mse'])/other.loc[seed,'mse'])))
    pairs=pd.DataFrame(paired); pairs.to_csv(out/'paired_differences.csv',index=False)
    ps=pairs.groupby(['dataset','horizon','variant'],sort=False).agg(n=('seed','count'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),delta_mae_mean=('delta_mae','mean'),delta_mae_sd=('delta_mae','std'),mse_reduction_mean=('mse_reduction_percent','mean'),mse_reduction_sd=('mse_reduction_percent','std')).reset_index()
    ps.to_csv(out/'paired_summary.csv',index=False)
    state_frame=pd.DataFrame(states); state_frame.to_csv(out/'state_groups.csv',index=False)
    state_summary=state_frame[(state_frame.horizon==6)&state_frame.variant.isin(MAIN)].groupby(['dataset','group','variant']).agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'),valid_elements=('valid_elements','first')).reset_index()
    state_summary.to_csv(out/'state_group_summary.csv',index=False)
    main_table=summary[summary.variant.isin(MAIN)].copy()
    display=pd.DataFrame({'Dataset':main_table.dataset,'H':main_table.horizon,'Method':main_table.variant.map(LABEL),'MSE (mean ± SD)':[f'{m:.6f} ± {s:.6f}' for m,s in zip(main_table.mse_mean,main_table.mse_sd)],'MAE (mean ± SD)':[f'{m:.6f} ± {s:.6f}' for m,s in zip(main_table.mae_mean,main_table.mae_sd)],'n':main_table.n})
    (out/'main_table.md').write_text('# Main comparisons\n\n'+md_table(display)+'\n\nSD is across five seeds on a fixed chronological test stream.\n',encoding='utf-8')
    (out/'main_table.tex').write_text(display.to_latex(index=False,escape=True).replace('±',r'$\pm$'))
    ab=summary[(summary.horizon==6)&(~summary.variant.isin(['numeric_add','text_add']))].copy()
    ab.to_csv(out/'ablation_table.csv',index=False)
    ab.to_latex(out/'ablation_table.tex',index=False,float_format='%.6f',escape=True)
    # Matched seed effects, no confidence interval or significance stars.
    main_figure(pairs,figs)
    # Mechanism figures use fixed dataset/seed and the full stream, not best cases.
    captions=[]
    for dataset in ['Energy','SocialGood']:
        data=load_data(dataset); sensor=data.columns.index('OT') if 'OT' in data.columns else 0
        records={v:np.load(ROOT/'runs'/f'{dataset}_h6_s2023_{v}'/'predictions.npz') for v in MAIN}
        full=records['full']; origins=full['origins']; valid=full['valid_mask'][:,0,sensor]
        fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,gridspec_kw={'height_ratios':[2,1,1.5]})
        axes[0].plot(origins,np.where(valid,full['targets'][:,0,sensor],np.nan),color='black',lw=1.4,label='Observed target')
        for v,color,style in [('numeric_add','#999999',':'),('text_add','#0072B2','--'),('full','#D55E00','-')]: axes[0].plot(origins,records[v]['predictions'][:,0,sensor],color=color,ls=style,lw=1,label=LABEL[v])
        axes[0].set_ylabel('First-step target\n(standardized)'); axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.25),ncol=4,frameon=False,fontsize=8)
        axes[1].plot(origins,full['state_change_scores'],color='#009E73'); axes[1].set_ylabel('State-change\nscore')
        available=full['text_available']; axes[1].plot(origins[available],np.zeros(available.sum()),'|',color='#0072B2',alpha=.3,label='Text available (proxy)'); axes[1].legend(loc='upper left',fontsize=8,frameon=False)
        gate=full['gate_weights'].mean(1)
        axes[2].stackplot(origins,gate.T,labels=['Zero-init','Low-rank','Spectral','Fluctuation','TextFiLM'],alpha=.85)
        axes[2].set_ylabel('Mean expert weight'); axes[2].set_xlabel('Forecast origin (sorted row index)'); axes[2].set_ylim(0,1); axes[2].legend(loc='upper center',bbox_to_anchor=(.5,-.28),ncol=5,frameon=False,fontsize=8)
        for ax in axes: ax.axvline(origins[0]+6,color='.3',ls='--',lw=.8)
        fig.tight_layout(); savefig(fig,figs/f'mechanism_{dataset}')
        dominant=int(gate.mean(0).argmax()); names=['Zero-init','Low-rank','Spectral','Fluctuation','TextFiLM']
        captions.append(f'{dataset}: fixed seed 2023, H=6, complete test stream; plots the first horizon step of target {data.columns[sensor]}. Expert weights average over variables. The dominant {names[dominant]} branch has mean weight {gate.mean(0)[dominant]:.5f}; this near-single-expert example does not demonstrate temporal specialization. Dashed vertical line is the first test-label maturity. Text ticks indicate availability under end_date proxy, not verified publication times. Missing targets are omitted from the ground-truth curve. No favorable event selection.')
        for rec in records.values(): rec.close()
    captions.insert(0,'Paired H=6 MSE reductions: each percentage is 100 x (baseline MSE - iAdapter MSE) / baseline MSE. Dots are five matched seed/backbone runs per dataset; diamonds and whiskers show mean ± one sample SD of those paired percentages, not a confidence interval. Positive values favor iAdapter. Mean paired percentages need not have the same sign as the difference of mean raw MSEs. All predefined datasets and seeds are shown.')
    (out/'figure-catalog.md').write_text('# Figure catalog\n\n'+'\n\n'.join(captions)+'\n',encoding='utf-8')
    includes=[]
    for name,caption in zip(['main_paired_effects','mechanism_Energy','mechanism_SocialGood'],captions):
        includes.append('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.98\\textwidth]{figures/'+name+'.pdf}\n\\caption{'+caption.replace('_','\\_').replace('%','\\%').replace('±',r'\(\pm\)')+'}\n\\label{fig:'+name+'}\n\\end{figure*}\n')
    (figs/'latex_includes.tex').write_text('\n'.join(includes))
    write_json(out/'validation.json',{'planned_runs':190,'completed_runs':len(frame),'unique_backbones':len(refs),'all_saved_metrics_recomputed':True,'all_comparisons_same_origins_targets_masks_checkpoint':True,'all_mature_origins_once_only':True,'code_commit':frame.commit.iloc[0],'p_values_computed':False})
    (out/'stats-appendix.md').write_text('# Statistical scope\n\nFive seeds measure training/optimization variability conditional on each fixed dataset and chronological split. Comparisons share each seed-specific pretrained backbone and identical forecast origins and observed target masks. Means and sample SDs are descriptive. Overlapping windows, target variables and horizons are not counted as independent repeats. No significance claims, p-values or unstable five-seed bootstrap intervals are reported. Relative effects are computed within matched seeds; raw MSE values are not pooled across domains. All 190 planned runs are included. State-change groups use validation tertiles of historical half-window mean/scale differences; missing test groups are omitted, not imputed. Group labels are descriptive and their valid-element counts are reported.\n')
    print('ANALYSIS_VALIDATED 190/190',flush=True)

if __name__=='__main__': main()

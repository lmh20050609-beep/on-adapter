"""Render all matched seeds, no selection or simulated observations."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'evidence'
OUT=Path(__file__).resolve().parent
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'axes.linewidth':.6})
b1=pd.read_csv(ROOT/'b1/runs.csv');b2=pd.read_csv(ROOT/'b2/runs.csv')
assert len(b1)==60 and len(b2)==24
contrasts=[]
for d in ['Energy','SocialGood']:
    p=b1[b1.dataset==d].pivot(index='seed',columns='method',values='mse')
    q=b2[b2.dataset==d].pivot(index='seed',columns='probe',values='mse')
    for label,series in [('Full U1F1 - U1F0',p.iadapter_online-p.iadapter_U1F0),('Text online - numerical online',p.text_add_online-p.numeric_online),('Content P2 - metadata P1',q.P2-q.P1),('Donor P3 - correct P2',q.P3-q.P2)]:
        for seed,v in series.items():contrasts.append(dict(dataset=d,seed=seed,contrast=label,delta_mse=v))
paired=pd.DataFrame(contrasts);paired.to_csv(OUT/'paired_source_data.csv',index=False)
labels=paired.contrast.unique();colors=['#577C9D','#AB754F','#577C9D','#AB754F']
fig,axs=plt.subplots(1,2,figsize=(7.086614,3.779528),layout='constrained')
for ax,d,panel in zip(axs,['Energy','SocialGood'],['a','b']):
    for y,(label,color) in enumerate(zip(labels,colors)):
        v=paired[(paired.dataset==d)&(paired.contrast==label)].sort_values('seed').delta_mse.to_numpy()
        assert len(v)==3
        ax.scatter(v*1000,y+np.array([-.13,0,.13]),s=16,color=color,alpha=.55,zorder=3)
        ax.errorbar(v.mean()*1000,y,xerr=v.std(ddof=1)*1000,fmt='s',color='#222222',markersize=3.3,capsize=2,elinewidth=.8,zorder=4)
    ax.axvline(0,color='#888888',lw=.7,ls='--');ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_title(d,loc='left',fontweight='bold');ax.set_xlabel('Paired MSE change (x 0.001)');ax.text(-.09,1.07,panel,transform=ax.transAxes,fontweight='bold')
fig.suptitle('All matched seeds; square = mean, whisker = sample SD',fontsize=8)
fig.savefig(OUT/'paired_evidence.svg',bbox_inches='tight')
fig.savefig(OUT/'paired_evidence.pdf',bbox_inches='tight')
fig.savefig(OUT/'paired_evidence.tiff',dpi=600,bbox_inches='tight')
fig.savefig(OUT/'paired_evidence.png',dpi=600,bbox_inches='tight')
plt.close(fig)
methods=[('frozen_backbone','Bare frozen backbone'),('numeric_static','Numerical static'),('numeric_online','Numerical online'),('text_add_static','Additive text static'),('text_add_online','Additive text online'),('iadapter_static','Full U0F0'),('iadapter_U0F1','Full U0F1'),('iadapter_U1F0','Full U1F0'),('iadapter_online','Full U1F1'),('gpt4mts_static','Author-model GPT4MTS static'),('P0','P0 numerical readout'),('P1','P1 + metadata'),('P2','P2 + BERT-PCA16'),('P3','P3 historical donor')]
records=[]
fig,axs=plt.subplots(1,2,figsize=(7.086614,5.708661),layout='constrained')
for ax,d,panel in zip(axs,['Energy','SocialGood'],['a','b']):
    for y,(method,label) in enumerate(methods):
        rows=(b1[(b1.dataset==d)&(b1.method==method)] if y<10 else b2[(b2.dataset==d)&(b2.probe==method)]).sort_values('seed')
        assert len(rows)==3
        vals=rows.mse.to_numpy();color='#577C9D' if y<5 else ('#AB754F' if y<10 else '#848484')
        ax.scatter(vals,y+np.array([-.12,0,.12]),s=12,color=color,alpha=.6,zorder=3)
        ax.errorbar(vals.mean(),y,xerr=vals.std(ddof=1),fmt='s',color=color,markersize=3.1,capsize=1.8,elinewidth=.75,zorder=4)
        for _,row in rows.iterrows():records.append(dict(dataset=d,method=method,seed=int(row.seed),mse=row.mse,mae=row.mae))
    ax.set_yticks(range(len(methods)),[x[1] for x in methods]);ax.invert_yaxis();ax.set_xlim(left=0);ax.set_title(d,loc='left',fontweight='bold');ax.set_xlabel('Standardized MSE (lower is better)');ax.axhline(9.5,color='#aaaaaa',lw=.6);ax.text(-.09,1.04,panel,transform=ax.transAxes,fontweight='bold')
fig.suptitle('Strong controls retained; all seeds, mean and sample SD',fontsize=8)
fig.savefig(OUT/'strong_controls.svg',bbox_inches='tight')
fig.savefig(OUT/'strong_controls.pdf',bbox_inches='tight')
fig.savefig(OUT/'strong_controls.tiff',dpi=600,bbox_inches='tight')
fig.savefig(OUT/'strong_controls.png',dpi=600,bbox_inches='tight')
plt.close(fig)
pd.DataFrame(records).to_csv(OUT/'controls_source_data.csv',index=False)

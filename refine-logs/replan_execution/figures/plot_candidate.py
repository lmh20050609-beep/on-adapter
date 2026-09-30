from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.titlesize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
OUT=Path(__file__).resolve().parent
rows=pd.read_csv(OUT.parent/'evidence/E003/paired_runs.csv')
comparisons=[('LLM_online_minus_raw_online','LLM online - raw online'),('LLM_online_minus_numeric_online','LLM online - numerical online'),('LLM_online_minus_LLM_static','LLM online - LLM static')]
# The other12 rows are donor contrasts shown in the companion tables, not excluded observations.
selected=rows[rows.comparison.isin([x[0] for x in comparisons])].copy();assert len(rows)==30 and len(selected)==18
selected.to_csv(OUT/'E003_plot_source_data.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(7.086614,3.0),layout='constrained')
for ax,d,panel in zip(axs,['Energy','SocialGood'],['a','b']):
    for y,(name,label) in enumerate(comparisons):
        values=selected[(selected.dataset==d)&(selected.comparison==name)].sort_values('seed').delta_mse.to_numpy()*1000;assert len(values)==3
        ax.scatter(values,y+np.array([-.12,0,.12]),s=15,color='#577C9D',alpha=.55,zorder=3)
        ax.errorbar(values.mean(),y,xerr=values.std(ddof=1),fmt='s',markersize=3.3,color='#222222',capsize=2,zorder=4)
    ax.axvline(0,color='#888888',ls='--',lw=.7);ax.set_yticks(range(3),[x[1] for x in comparisons]);ax.invert_yaxis();ax.set_title(d,loc='left',fontweight='bold');ax.set_xlabel('Paired MSE change (x 0.001)');ax.text(-.08,1.07,panel,transform=ax.transAxes,fontweight='bold')
fig.suptitle('Simple five-dimensional fusion: all seeds, mean and sample SD',fontsize=8)
fig.savefig(OUT/'E003_progress.svg',bbox_inches='tight')
fig.savefig(OUT/'E003_progress.pdf',bbox_inches='tight')
fig.savefig(OUT/'E003_progress.tiff',dpi=600,bbox_inches='tight')
fig.savefig(OUT/'E003_progress.png',dpi=600,bbox_inches='tight')

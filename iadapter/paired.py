"""Matched main-method contrasts; seed is the repetition unit, never a window."""
import pandas as pd
from .config import MAIN

CONTRASTS=[('text_add','numeric_add'),('full','numeric_add'),('full','text_add')]

def paired_effects(frame,grouped=False):
    keys=['dataset','horizon']+(['group'] if grouped else [])
    selected=frame[frame.variant.isin(MAIN)].copy()
    if selected.duplicated(keys+['seed','variant']).any():
        raise ValueError('Duplicate method/seed in a comparison cell')
    rows=[]
    for cell,block in selected.groupby(keys,sort=True):
        cell=cell if isinstance(cell,tuple) else (cell,)
        for candidate,control in CONTRASTS:
            a=block[block.variant==candidate].set_index('seed')
            b=block[block.variant==control].set_index('seed')
            expected=set(range(2023,2028))
            if set(a.index)!=expected or set(b.index)!=expected:
                raise ValueError(f'Incomplete paired seeds: {cell}, {candidate}/{control}')
            for seed in sorted(expected):
                x,y=a.loc[seed],b.loc[seed]
                if int(x.valid_elements)!=int(y.valid_elements):
                    raise ValueError('Paired methods have different valid target counts')
                rows.append(dict(zip(keys,cell),seed=seed,candidate=candidate,control=control,valid_elements=int(x.valid_elements),delta_mse=float(x.mse-y.mse),delta_mae=float(x.mae-y.mae)))
    pairs=pd.DataFrame(rows)
    if pairs.empty: raise ValueError('No paired main-method observations')
    summary=pairs.groupby(keys+['candidate','control'],sort=True).agg(n=('seed','count'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),delta_mae_mean=('delta_mae','mean'),delta_mae_sd=('delta_mae','std'),mse_improved_seeds=('delta_mse',lambda x:int((x<0).sum())),mae_improved_seeds=('delta_mae',lambda x:int((x<0).sum())),valid_elements_per_seed=('valid_elements','first')).reset_index()
    counts=pairs.groupby(keys+['candidate','control']).valid_elements.nunique()
    if (counts!=1).any(): raise ValueError('Different seed target counts in a comparison cell')
    return pairs,summary

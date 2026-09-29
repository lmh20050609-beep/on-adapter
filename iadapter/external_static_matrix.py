"""Fixed five-domain/five-seed partial external matrix, preserving pilot evidence."""
import argparse,ast,json,subprocess,time,traceback
from pathlib import Path
import pandas as pd
from .config import ROOT,DOMAINS
from .external_static import OUTPUT as PILOT,METHODS,main as run_group,pilot_config
from .external_static_analysis import analyze as audit_group
from .util import revision,write_json
from .analyze import md_table

OUTPUT=ROOT/'experiments/external_static_matrix_20260929'
PILOT_COMMIT='6b93caf96fdb5f680fe026d182529af664cae904'

def groups():return [dict(dataset=d,seed=s) for d in DOMAINS for s in range(2023,2028)]
def group_id(job):return f'{job["dataset"]}_s{job["seed"]}'
def reused(job):return job==dict(dataset='Economy',seed=2023)
def group_path(base,job):return PILOT if reused(job) else Path(base)/'groups'/group_id(job)

def verify_core_unchanged():
    def source(file):return subprocess.check_output(['git','show',PILOT_COMMIT+':iadapter/'+file],cwd=ROOT,text=True)
    for file in ['config.py','models.py','data.py','engine.py','run.py','native_gpt4mts.py','review_sprint.py']:
        assert ast.dump(ast.parse(source(file)))==ast.dump(ast.parse((ROOT/'iadapter'/file).read_text())),'Core differs: '+file
    def bodies(text):return {n.name:ast.dump(n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    before,after=bodies(source('external_static.py')),bodies((ROOT/'iadapter/external_static.py').read_text())
    for name in ['ot_data','encode_cls','evaluate','train_native']:
        assert before[name]==after[name],'Training/evaluation function differs: '+name
    # The previous default Config remains identical; only dataset/seed are parameterized.
    original=next(n for n in ast.parse(source('external_static.py')).body if isinstance(n,ast.FunctionDef) and n.name=='pilot_config')
    from .config import Config
    ns={'Config':Config};exec(compile(ast.Module(body=[original],type_ignores=[]),'pilot_config_reference','exec'),ns)
    for variant in ['numeric_add','full']:
        assert ns['pilot_config'](variant).to_dict()==pilot_config(variant).to_dict()
    return dict(pilot_commit=PILOT_COMMIT,training_and_evaluation_ast_equal=True,default_pilot_config_equal=True)

def campaign(base=OUTPUT):
    base=Path(base);base.mkdir(parents=True,exist_ok=True);code=revision();proof=verify_core_unchanged()
    if code['dirty']:raise RuntimeError('Commit matrix source before launch')
    assert (PILOT/'gpt4mts_cls512.npz').is_file(),'Declared reusable Economy cache is missing'
    path=base/'queue_state.json';planned=groups()
    if path.exists():
        state=json.loads(path.read_text())
        if state['code']!=code or state['planned_groups']!=planned:raise RuntimeError('Matrix source/config mismatch')
    else:state=dict(code=code,planned_groups=planned,methods=METHODS,proof=proof,groups={group_id(j):{'status':'pending'} for j in planned},planned_cells=75,new_cells=72,reused_cells=3)
    for job in planned:
        key=group_id(job);entry=state['groups'][key];directory=group_path(base,job)
        if entry['status'] in ['completed','reused']:continue
        if entry['status']!='pending':raise RuntimeError('Unresolved matrix group: '+key)
        entry.update(status='running',source=str(directory),started_at=time.time());write_json(path,state)
        try:
            if not reused(job):
                cache=PILOT if job['dataset']=='Economy' else base/'text_cache'/job['dataset']
                run_group(directory,dataset=job['dataset'],seed=job['seed'],cache_dir=cache)
            validation=audit_group(directory,**job)
            if reused(job):assert validation['code']['commit']==PILOT_COMMIT
            else:assert validation['code']==code
            entry.update(status='reused' if reused(job) else 'completed',validated_cells=3,completed_at=time.time())
        except Exception:
            entry.update(status='failed',error=traceback.format_exc());write_json(path,state);raise
        write_json(path,state);print(f'MATRIX_GROUP_COMPLETE {key} {entry["status"]}',flush=True)
    print('STATIC_MATRIX_COMPLETE 75 / 75',flush=True)

def analyze(base=OUTPUT):
    base=Path(base);state=json.loads((base/'queue_state.json').read_text());verify_core_unchanged()
    assert state['planned_groups']==groups() and state['methods']==METHODS
    frames=[];proofs=[]
    for job in groups():
        entry=state['groups'][group_id(job)];directory=group_path(base,job)
        assert entry['status']==('reused' if reused(job) else 'completed') and Path(entry['source']).resolve()==directory.resolve()
        proof=audit_group(directory,**job);proofs.append(proof)
        assert proof['code']['commit']==(PILOT_COMMIT if reused(job) else state['code']['commit'])
        f=pd.read_csv(directory/'reports/summary.csv');f['reused']=reused(job);f['source']=str(directory);f['experiment_commit']=proof['code']['commit'];frames.append(f)
    runs=pd.concat(frames,ignore_index=True)
    expected={(j['dataset'],j['seed'],m) for j in groups() for m in METHODS}
    assert len(runs)==75 and not runs.duplicated(['dataset','seed','method']).any() and set(zip(runs.dataset,runs.seed,runs.method))==expected
    assert all(len(block)==15 and block.origins.nunique()==1 and block.valid_elements.nunique()==1 for _,block in runs.groupby('dataset'))
    out=base/'reports';out.mkdir(exist_ok=True);runs.to_csv(out/'runs.csv',index=False)
    summary=runs.groupby(['dataset','method'],sort=False).agg(n=('seed','size'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'),origins=('origins','first'),valid_elements=('valid_elements','first'),trainable_parameters=('trainable_parameters_before_static_freeze','first')).reset_index()
    assert (summary.n==5).all();summary.to_csv(out/'summary.csv',index=False)
    pairs=[]
    for (d,s),block in runs.groupby(['dataset','seed']):
        b=block.set_index('method')
        for candidate,control in [('iadapter_static','numeric_static'),('gpt4mts_static','numeric_static'),('iadapter_static','gpt4mts_static')]:
            pairs.append(dict(dataset=d,seed=s,candidate=candidate,control=control,delta_mse=b.loc[candidate,'mse']-b.loc[control,'mse'],delta_mae=b.loc[candidate,'mae']-b.loc[control,'mae']))
    pairs=pd.DataFrame(pairs);pairs.to_csv(out/'paired_runs.csv',index=False)
    ps=pairs.groupby(['dataset','candidate','control'],sort=False).agg(n=('seed','size'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),delta_mae_mean=('delta_mae','mean'),delta_mae_sd=('delta_mae','std'),mse_improved_seeds=('delta_mse',lambda x:int((x<0).sum())),mae_improved_seeds=('delta_mae',lambda x:int((x<0).sum()))).reset_index();ps.to_csv(out/'paired_summary.csv',index=False)
    decisions=[]
    for d,block in summary.groupby('dataset'):
        b=block.set_index('method')
        for control in ['numeric_static','gpt4mts_static']:
            row=ps[(ps.dataset==d)&(ps.candidate=='iadapter_static')&(ps.control==control)].iloc[0]
            decisions.append(dict(dataset=d,control=control,mse_change_percent=100*(b.loc['iadapter_static','mse_mean']/b.loc[control,'mse_mean']-1),mae_change_percent=100*(b.loc['iadapter_static','mae_mean']/b.loc[control,'mae_mean']-1),mse_improved_seeds=int(row.mse_improved_seeds),mae_improved_seeds=int(row.mae_improved_seeds)))
    decisions=pd.DataFrame(decisions);decisions.to_csv(out/'iadapter_comparisons.csv',index=False)
    display=summary[['dataset','method','n','origins','valid_elements']].copy()
    display['MSE mean ± SD']=[f'{m:.6f} ± {s:.6f}' for m,s in zip(summary.mse_mean,summary.mse_sd)]
    display['MAE mean ± SD']=[f'{m:.6f} ± {s:.6f}' for m,s in zip(summary.mae_mean,summary.mae_sd)]
    validation=dict(status='passed',completed_cells=75,new_cells=72,reused_cells=3,datasets=DOMAINS,seeds=list(range(2023,2028)),original_csv_audited_groups=25,static_evaluation_no_updates=True,text_consumption_identical=False,experiment_code=state['code'],group_proofs=proofs,scope='exploratory static OT development; VoT absent; not original Table2 reproduction or independent confirmation')
    write_json(out/'validation.json',validation)
    report=f'''# 五域五seed共同OT静态对照

75/75单元完成：72新增、3个Economy/2023 pilot复用。25组均从原CSV核验目标/掩码/尺度/文本ID、选模边界与评分无更新；缺失或失败单元不进入此表。H6；Energy L48，其余L36。前60%训练、60%–70%选模、70%–80%开发评分。

## 全部结果

{md_table(display)}

## iAdapter相对控制

百分比为平均误差之比减1，正值为退化。改善seed数均以5为分母。

{md_table(decisions.round(3))}

## 比较边界

- 各域三个方法在相同原点/目标/有效掩码上评分，开发期没有更新。内部两方法共享每seed的单变量骨干，预热后将反馈清零并保持零；GPT4MTS保留作者模型和冻结规则。
- 共同可见性使用end_date代理，真实发布时间没有验证。iAdapter读取当前文本mean/64，GPT4MTS消费历史文本CLS/512序列；文本信息量、表示与训练阶段不相同，所以不是结构单因素试验。
- seed只是同一固定流上的优化重复；内部方法共享骨干，而跨架构没有相同初始参数保证。差值按相同seed标签和评分流对齐，仅作描述性统计。重叠目标不能当独立重复，不给出显著性或独立泛化结论。
- 本表为静态OT任务，不能与原多变量在线表合并，不能由它推出在线机制的因果贡献。VoT没有可核验的生成/原点时间记录，仍未列入；75单元不是原四方法100单元的完成声明。
- 原始预测和全部训练历史仍在各source目录；runs.csv列明每个单元来源与提交号，paired_runs.csv保留全部逐seed差值。计时包含不同缓存命中和共享训练机会，仅记录执行成本，不能替代独立效率测量。
'''
    (out/'RESULTS.md').write_text(report,encoding='utf-8')
    print('STATIC_MATRIX_VALIDATED 75 cells / 25 groups',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--analyze',action='store_true');a=p.parse_args()
    if a.analyze:analyze(a.output)
    else:campaign(a.output)

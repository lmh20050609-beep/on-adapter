"""Remeasure all four E-01 settings in the same timing campaign."""
import argparse,json,random,traceback
from pathlib import Path
import torch
from .config import ROOT
from .review_sprint import DEFAULT_RUN
from .review_compression import OUTPUT,METHODS,initialize_projection
from .review_cost import profile_case,evidence_stamp
from .review_cost_report import main as report
from .util import revision,write_json

COST_OUTPUT=ROOT/'profile_output/review_compression_20260929'
SETTINGS={'E0':(METHODS[0],96),'E1':(METHODS[0],8),'E2':(METHODS[1],96),'E3':(METHODS[1],8)}

def main(output=COST_OUTPUT,sanity=False):
    proof=json.loads((OUTPUT/'reports/validation.json').read_text())
    assert proof['status']=='passed' and proof['combined_cells']==24
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);before={str(b):evidence_stamp(b) for b in [DEFAULT_RUN,OUTPUT]}
    queues={str(b):(b/'queue_state.json').read_bytes() for b in [DEFAULT_RUN,OUTPUT]}
    config=dict(datasets=['Energy'] if sanity else ['Energy','SocialGood'],variants=['E1','E3'] if sanity else list(SETTINGS),rounds=1 if sanity else 3,warmup=2 if sanity else 50,repeats=5 if sanity else 200,hardware=torch.cuda.get_device_name(),torch=str(torch.__version__),profile_code=revision(),scope='isolated microbenchmark, not accuracy; queue/disk excluded',setting_definitions='E0/E1: five experts with routing text feedback and entropy disabled, projection 96/8. E2/E3: single MLP, projection 96/8. Sanity timings are not formal costs.')
    if config['profile_code']['dirty']:raise RuntimeError('Commit profiling source first')
    write_json(output/'config.json',config);write_json(output/'status.json',{'status':'running'});rows=[]
    try:
        for round_id in range(config['rounds']):
            order=[(d,v) for d in config['datasets'] for v in config['variants']];random.Random(5050+round_id).shuffle(order)
            for d,setting in order:
                variant,width=SETTINGS[setting];base=DEFAULT_RUN if width==96 else OUTPUT
                directory=base/'runs'/f'{d}_h6_s2023_{variant}{"_p8" if width==8 else ""}'
                row=profile_case(base,d,variant,round_id,config['warmup'],config['repeats'],directory_override=directory,adapter_factory=lambda c,n:initialize_projection(c,n,width))
                row.update(variant=setting,base_variant=variant,projection_hidden=width,source=str(directory))
                rows.append(row);write_json(output/'raw.json',rows)
                print(f'COMPRESSION_COST {len(rows)} {d}/{setting}/round{round_id} pred={row["prediction_cached"]["median_ms"]:.3f}ms update={row["update_fixed_record"]["median_ms"]:.3f}ms',flush=True)
        assert before=={str(b):evidence_stamp(b) for b in [DEFAULT_RUN,OUTPUT]}
        assert queues=={str(b):(b/'queue_state.json').read_bytes() for b in [DEFAULT_RUN,OUTPUT]}
        write_json(output/'status.json',dict(status='completed',cases=len(rows),scientific_artifacts_unchanged=True,backbone_unchanged=True))
        report(output,reference='E1' if sanity else 'E0')
    except Exception:
        write_json(output/'status.json',dict(status='failed',completed_cases=len(rows),error=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sanity',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    main(a.output or (COST_OUTPUT.with_name(COST_OUTPUT.name+'_sanity') if a.sanity else COST_OUTPUT),a.sanity)

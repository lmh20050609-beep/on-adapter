from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data'
BERT_ARCHIVE = ROOT / 'weights/bert-base-uncased.zip'
DOMAINS = ['Economy','Energy','Security','SocialGood','Climate']
MAIN = ['numeric_add','text_add','full']
ABLATIONS = ['no_text_gradient','no_route_text_gradient','no_decisiveness','homogeneous','no_text','no_numeric_gradient','no_distribution','uniform']
VARIANTS = MAIN + ABLATIONS

@dataclass
class Config:
    dataset: str = 'Economy'
    horizon: int = 6
    seed: int = 2023
    variant: str = 'full'
    d_model: int = 32
    e_layers: int = 2
    hidden: int = 64
    text_dim: int = 64
    dropout: float = .1
    temperature: float = 1.
    entropy_weight: float = .02
    momentum: float = .9
    lr: float = .001
    weight_decay: float = .0001
    grad_clip: float = 1.
    pretrain_epochs: int = 30
    pretrain_batch: int = 32
    patience: int = 5
    protocol: str = 'origin_queue_v1'
    device: str = 'cuda'

    @property
    def lookback(self): return 48 if self.dataset == 'Energy' else 36
    @property
    def uses_text(self): return self.variant not in {'numeric_add','no_text'}
    @property
    def text_feedback(self): return self.variant not in {'numeric_add','text_add','no_text','no_text_gradient'}
    @property
    def numeric_feedback(self): return self.variant != 'no_numeric_gradient'
    def to_dict(self): return dict(asdict(self), lookback=self.lookback, z_loc=2, online_batch=1, text_encoder='bert-base-uncased contextual 12 layers', text_max_length=64)

def matrix():
    jobs=[]
    for d in DOMAINS:
        for seed in range(2023,2028):
            for v in MAIN: jobs.append(dict(stage='B',dataset=d,horizon=6,seed=seed,variant=v))
    for v in ABLATIONS:
        for d in ['Energy','SocialGood']:
            for seed in range(2023,2028): jobs.append(dict(stage='C',dataset=d,horizon=6,seed=seed,variant=v))
    for seed in range(2023,2028): jobs.append(dict(stage='C',dataset='Security',horizon=6,seed=seed,variant='no_route_text_gradient'))
    for d in ['Economy','Energy']:
        for seed in range(2023,2028):
            for v in MAIN: jobs.append(dict(stage='D',dataset=d,horizon=12,seed=seed,variant=v))
    assert len(jobs)==190
    return jobs

def job_id(job): return f"{job['dataset']}_h{job['horizon']}_s{job['seed']}_{job['variant']}"

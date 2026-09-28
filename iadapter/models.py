import torch
from torch import nn
from .legacy_core import ITransformerZ,IdentityExpert,TrendExpert,SeasonalityExpert,FluctuationExpert,TextFiLMExpert,AddFusionAdapter,compute_offset_features

class Backbone(ITransformerZ):
    def context(self,x):
        mu=x.mean(1,keepdim=True).detach()
        sigma=torch.sqrt(x.var(1,keepdim=True,unbiased=False)+1e-5).detach()
        h=self.embedding((x-mu)/sigma)
        for layer in self.layers: h=layer(h)
        return h,mu,sigma
    def head(self,h,mu,sigma):
        return self.projector(self.norm(h)).permute(0,2,1)*sigma[:,0,:].unsqueeze(1)+mu[:,0,:].unsqueeze(1)
    def forward(self,x):
        h,mu,sigma=self.context(x)
        return self.head(h,mu,sigma)

class Homogeneous(nn.Module):
    def __init__(self,hidden,width,dz):
        super().__init__(); self.net=nn.Sequential(nn.Linear(hidden,width),nn.GELU(),nn.Linear(width,dz))
    def forward(self,h,*args): return self.net(h)

class SoftMoE(nn.Module):
    def __init__(self,cfg,N):
        super().__init__(); self.cfg=cfg; D=cfg.d_model; h=cfg.hidden; T=cfg.text_dim
        self.lr=nn.Parameter(torch.ones(N,D)); self.txt_lr=nn.Parameter(torch.ones(T))
        self.f_proj=nn.Linear(D,h); self.g_proj=nn.Linear(D,h); self.t_proj=nn.Linear(T,h)
        self.trunk=nn.Linear(3*h,h); self.drop=nn.Dropout(cfg.dropout)
        self.experts=nn.ModuleList([IdentityExpert(h,D),TrendExpert(h,D,rank=2),SeasonalityExpert(h,D,num_modes=8),FluctuationExpert(h,D),TextFiLMExpert(h,D,T)])
        target=sum(p.numel() for p in self.experts.parameters())
        self.expert_matching={'heterogeneous_parameters':target}
        if cfg.variant=='homogeneous':
            # Per MLP: h*w+w + w*D+D. Total strictly monotone in width.
            width=min(range(1,4097),key=lambda w:(abs(5*(w*(h+D+1)+D)-target),w))
            self.experts=nn.ModuleList([Homogeneous(h,width,D) for _ in range(5)])
            self.expert_matching['homogeneous_width']=width
        self.expert_matching['actual_parameters']=sum(p.numel() for p in self.experts.parameters())
        self.offset_norm=nn.LayerNorm(8)
        self.gate_f=nn.Linear(D,h); self.gate_p=nn.Linear(D,h); self.gate_g=nn.Linear(D,h)
        self.gate_t=nn.Linear(T,h); self.gate_o=nn.Linear(8,h)
        self.gate_net=nn.Sequential(nn.Linear(5*h,h),nn.ReLU(),nn.Linear(h,5))
        self.grad_mod_z=nn.Linear(D,5); self.grad_mod_t=nn.Linear(T,5)
        self.alpha=nn.Parameter(torch.tensor(.1)); self.last_gate=None; self.aux_loss=None

    def forward(self,feature,hg,e,ge,offset):
        c=self.cfg; B,N,D=feature.shape
        if not c.uses_text: e=torch.zeros_like(e); ge=torch.zeros_like(ge)
        if not c.text_feedback: ge=torch.zeros_like(ge)
        if not c.numeric_feedback: hg=torch.zeros_like(hg)
        g=(hg*self.lr).unsqueeze(0).expand(B,-1,-1)
        f_h=torch.relu(self.f_proj(feature)); g_h=torch.relu(self.g_proj(g))
        txt=e*(1+self.txt_lr*ge/(ge.norm()+1e-6)).unsqueeze(0) if c.text_feedback else e
        t_h=torch.relu(self.t_proj(txt.unsqueeze(1).expand(-1,N,-1)))
        h=self.drop(torch.relu(self.trunk(torch.cat([f_h,g_h,t_h],-1))))
        outputs=torch.stack([expert(h,feature,g_h,txt,f_h) for expert in self.experts],2)
        if c.variant=='uniform':
            w=feature.new_full((B,N,5),.2)
        else:
            routing_text=e if c.variant=='no_route_text_gradient' else txt
            dist=torch.relu(self.gate_o(self.offset_norm(offset))).unsqueeze(1).expand(-1,N,-1)
            if c.variant=='no_distribution': dist=torch.zeros_like(dist)
            gh=torch.cat([torch.relu(self.gate_f(feature)),torch.relu(self.gate_p(feature.mean(1,keepdim=True).expand(-1,N,-1))),torch.relu(self.gate_g(g)),torch.relu(self.gate_t(routing_text.unsqueeze(1).expand(-1,N,-1))),dist],-1)
            bias=self.grad_mod_z(g.mean(1)) if c.numeric_feedback else feature.new_zeros((B,5))
            if c.text_feedback and c.variant!='no_route_text_gradient': bias=bias+self.grad_mod_t(ge.unsqueeze(0).expand(B,-1))
            w=torch.softmax((self.gate_net(gh)+self.alpha*bias.unsqueeze(1))/c.temperature,-1)
        self.last_gate=w.detach()
        self.aux_loss= -(w*(w+1e-9).log()).sum(-1).mean()*c.entropy_weight if c.variant not in {'uniform','no_decisiveness'} else feature.new_zeros(())
        return (outputs*w.unsqueeze(-1)).sum(2)

class Adaptation(nn.Module):
    def __init__(self,cfg,N):
        super().__init__(); self.cfg=cfg
        if cfg.uses_text:
            self.project=nn.Sequential(nn.Linear(768,96),nn.ReLU(),nn.Linear(96,cfg.text_dim),nn.ReLU(),nn.Dropout(cfg.dropout),nn.LayerNorm(cfg.text_dim))
        else: self.project=None
        if cfg.variant in {'numeric_add','text_add'}:
            self.adapter=AddFusionAdapter(cfg.d_model,(1,N,cfg.d_model),text_dim=cfg.text_dim,hidden=cfg.hidden,dropout=cfg.dropout,use_text=cfg.uses_text,use_text_grad=False)
        else: self.adapter=SoftMoE(cfg,N)
    def text(self,pooled,available,device):
        if self.project is None: return torch.zeros(1,self.cfg.text_dim,device=device)
        return self.project(pooled)*float(available)
    def forward(self,feature,hg,e,ge,offset): return self.adapter(feature,hg,e,ge,offset)
    @property
    def gate(self): return self.adapter.last_gate
    @property
    def auxiliary(self): return self.adapter.aux_loss

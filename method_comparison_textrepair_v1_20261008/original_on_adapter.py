"""Use the submitted IG-SMoE and VICReg classes in the common origin protocol."""
import ast
import importlib.util
from pathlib import Path
import torch
from torch import nn
from .config import ROOT

SOURCE = ROOT / 'original-paper-replay/uniform-l36-source'
names = {'IdentityExpert', 'TrendExpert', 'SeasonalityExpert', 'FluctuationExpert',
         'TextFiLMExpert', 'IGSoftMoEFusionAdapter', 'compute_offset_features'}
tree = ast.parse((SOURCE / 'adapt_z_multimodal.py').read_text('utf-8-sig'))
selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
assert len(selected) == len(names)
# This logging diagnostic is not read by the gate/loss. Keep it on-device.
changes = 0
for node in ast.walk(next(n for n in selected if n.name == 'IGSoftMoEFusionAdapter')):
    if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == 'eff_experts' and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == 'float'
            and isinstance(node.value.args[0], ast.Call)):
        node.value = node.value.args[0]
        changes += 1
assert changes == 1, 'Pinned diagnostic assignment changed'
namespace = {'torch': torch, 'nn': nn}
exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
             str(SOURCE / 'adapt_z_multimodal.py'), 'exec'), namespace)
compute_offset_features = namespace['compute_offset_features']
spec = importlib.util.spec_from_file_location('on_adapter_alignment', SOURCE / 'models/text_alignment.py')
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


class OriginalOnAdapter(nn.Module):
    def __init__(self, settings, channels):
        super().__init__()
        d = settings['d_model']
        self.proj = nn.Sequential(nn.Linear(768, 96), nn.ReLU(), nn.Linear(96, 64),
                                  nn.ReLU(), nn.Dropout(settings['dropout']))
        self.norm = nn.LayerNorm(64)
        self.adapter = namespace['IGSoftMoEFusionAdapter'](
            d_model=d, z_shape=(1, channels, d), text_dim=64, hidden=64,
            dropout=settings['dropout'], aux_weight=.02, device=settings['device'])
        self.align = alignment.VICRegAlign(d, 64, d_clip=64, lam=25., mu=25., nu=1.)
        self.auxiliary = None

    def project(self, pooled):
        return self.norm(self.proj(pooled.float()))

    def forward(self, feature, gradient, text, text_gradient, offset):
        residual = self.adapter(feature, gradient, text, text_gradient, offset)
        self.auxiliary = self.adapter.aux_loss
        if self.training and len(feature) > 1:
            variance = text.detach().var(0).mean().double()
            # Preserve former Python double arithmetic, casting at multiplication.
            weight = (.1 * (1. + 4. * (1. - variance / .3).clamp_min(0.))).to(text.dtype)
            confidence = (text.detach().abs().sum(-1, keepdim=True) > 0).float()
            self.auxiliary = self.auxiliary + weight * self.align(feature, text, conf=confidence)
        return residual

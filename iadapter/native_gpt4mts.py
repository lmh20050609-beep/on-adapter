"""Load pinned author GPT4MTS with two explicit runtime compatibility changes."""
import ast,contextlib,io,json,types
from types import SimpleNamespace
from .config import ROOT

SOURCE=ROOT/'external/gpt4mts'
COMMIT='c85e74b2fd7048f5ef601453e5bcb9f6073cd3b4'
WEIGHTS=ROOT/'weights/gpt2'

class Compatibility(ast.NodeTransformer):
    def __init__(self,fix_batch):self.fix_batch=fix_batch;self.loads=0;self.squeeze=0
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='GPT2Model' and node.func.attr=='from_pretrained':
            assert len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value=='gpt2'
            node.args=[ast.Constant(str(WEIGHTS))]
            node.keywords=[k for k in node.keywords if k.arg!='cache_dir']+[ast.keyword(arg='local_files_only',value=ast.Constant(True))]
            self.loads+=1
        if isinstance(node.func,ast.Attribute) and node.func.attr=='squeeze' and not node.args and not node.keywords:
            value=node.func.value
            if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and value.func.attr=='mean' and isinstance(value.func.value,ast.Name) and value.func.value.id=='summary':
                self.squeeze+=1
                if self.fix_batch:return ast.copy_location(value,node)
        return node

def model_class(fix_batch=True):
    manifest=json.loads((SOURCE/'SOURCE.json').read_text(encoding='utf-8-sig'))
    if manifest['commit']!=COMMIT:raise ValueError('Unexpected upstream GPT4MTS revision')
    # Isolated namespace avoids colliding with other repositories named utils/models.
    rev={};exec(compile((SOURCE/'utils/rev_in.py').read_text(),str(SOURCE/'utils/rev_in.py'),'exec'),rev)
    file=SOURCE/'models/GPT4MTS.py';tree=ast.parse(file.read_text())
    imports=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='utils.rev_in']
    assert len(imports)==1 and [x.name for x in imports[0].names]==['RevIn']
    tree.body.remove(imports[0]);patch=Compatibility(fix_batch);tree=patch.visit(tree)
    assert patch.loads==1 and patch.squeeze==1,'Pinned source no longer matches declared compatibility patch'
    ast.fix_missing_locations(tree);namespace={'__name__':'iadapter.native_upstream_gpt4mts','RevIn':rev['RevIn']}
    exec(compile(tree,str(file),'exec'),namespace)
    return namespace['GPT4MTS']

def native_config(lookback=36,horizon=6):
    return SimpleNamespace(is_gpt=1,revin=True,patch_size=8,pretrain=1,stride=4,seq_len=lookback,gpt_layers=6,freeze=1,d_model=768,pred_len=horizon)

def build(lookback=36,horizon=6,device='cuda'):
    if not (WEIGHTS/'model.safetensors').is_file():raise FileNotFoundError('Pinned local GPT2 weights are required')
    with contextlib.redirect_stdout(io.StringIO()):model=model_class()(native_config(lookback,horizon),device)
    return model.to(device)

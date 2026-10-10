"""Reuse the submission's non-informative-clause filter on fact text only."""
import ast
import re
from .config import ROOT

path = ROOT / 'original-paper-replay/uniform-l36-source/adapt_z_multimodal.py'
tree = ast.parse(path.read_text('utf-8-sig'))
variables = {'_DATE_SEG', '_DROP_PATTERNS', '_STRIP_PATTERNS', '_WS'}
functions = {'_clean_clause', 'split_date_segments', 'clean_timemmd_text'}
body = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id in variables for t in node.targets):
        body.append(node)
    elif isinstance(node,ast.FunctionDef) and node.name in functions:
        body.append(node)
namespace={'re':re}
exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),namespace)

def clean_fact(text):
    cleaned, kept, _ = namespace['clean_timemmd_text'](text,max_clauses=6,min_clause_chars=20)
    return ('Facts. '+cleaned)[-1200:] if kept else ''

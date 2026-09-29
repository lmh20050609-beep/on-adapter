"""Resolve verified historical paths without editing original experiment records."""
import hashlib
import json
from pathlib import Path
from .config import ROOT

def resolve_evidence_path(recorded_path,root=ROOT):
    root=Path(root).resolve()
    manifest=root/'docs/PATH_MIGRATION.json'
    mappings=json.loads(manifest.read_text())['file_mappings'] if manifest.is_file() else {}
    entry=mappings.get(str(recorded_path))
    if entry is None:
        return Path(recorded_path).resolve()
    target=(root/entry['relative_path']).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Evidence migration target escapes the current project')
    if not target.is_file():
        raise FileNotFoundError(f'Missing migrated evidence: {target}')
    if hashlib.sha256(target.read_bytes()).hexdigest()!=entry['sha256']:
        raise ValueError(f'Migrated evidence differs from verified original: {target}')
    return target

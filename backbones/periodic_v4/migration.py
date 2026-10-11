"""Explicit periodic v3 -> v4 recovery lineage. Original checkpoints are read only."""
import ast
import json
import shutil
from pathlib import Path
import torch
from .config import ROOT
from .resources import digest, identity, atomic_json
from .state_io import read_json
from .parallel_policy import job_key

COMPUTE_FILES = ('native.py', 'multimodal.py', 'original_on_adapter.py', 'online.py',
                 'stream_dsof.py', 'data.py', 'data_cache.py', 'batching.py',
                 'prepared_text.py', 'text.py', 'text_clean.py', 'feature_store.py',
                 'numeric_features.py', 'checkpoint.py')


from .provenance import protocol, parent_settings


def compatible_code():
    new = Path(__file__).parent
    fork = read_json(new / 'FORK.json')
    old = ROOT / fork['parent_package']
    for name, expected in fork['parent_source_sha256'].items():
        if digest(old / name) != expected:
            raise ValueError('Migration parent code changed: ' + name)
    for name in COMPUTE_FILES:
        if digest(old / name) != digest(new / name):
            raise ValueError('Migration compute module changed: ' + name)
    normalized_worker=(new/'worker.py').read_text('utf-8').replace(
        'method_comparison_periodic_v4_20261007', 'method_comparison_periodic_v3_20261006').replace(
        'method_comparison_periodic_v4','method_comparison_periodic_v3').replace(
        "'periodic-v4'", "'periodic-v3'")
    if normalized_worker!=(old/'worker.py').read_text('utf-8'):
        raise ValueError('Worker differs beyond execution version/default entrypoint')
    # The worker has execution/metadata edits. Its loss, validation, parameter
    # selection and full epoch loop must have identical syntax trees.
    trees = [ast.parse((base / 'worker.py').read_text('utf-8')) for base in (old, new)]
    for name in ('loss_ot', 'evaluate', 'checkpoint_keys'):
        nodes = [next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name) for t in trees]
        if ast.dump(nodes[0]) != ast.dump(nodes[1]):
            raise ValueError('Migration worker compute changed: ' + name)
    loops = [next(n for n in ast.walk(t) if isinstance(n, ast.While)
                  and 'progress' in ast.dump(n.test)) for t in trees]
    if ast.dump(loops[0]) != ast.dump(loops[1]):
        raise ValueError('Migration epoch loop changed')
    return {name: digest(old / name) for name in COMPUTE_FILES}


def verify_source(record, settings):
    source = Path(record['source_directory'])
    for name, expected in record['source_hashes'].items():
        if digest(source / name) != expected:
            raise ValueError('Migration source artifact changed: ' + name)
    meta = read_json(source / 'config.json')
    old = parent_settings(settings)
    if meta['settings_input'] != old or protocol(old) != protocol(settings):
        raise ValueError('Migration experiment settings changed')
    for relative, expected in meta['source_fingerprint'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Migration recorded parent fingerprint changed')
    if digest(meta['data']['source']) != meta['data_sha256']:
        raise ValueError('Migration data changed')
    return meta


def continuation_jobs(settings, known, approved):
    if not settings['execution'].get('resume_periodic_v3', False):
        return set()
    code = compatible_code()
    old = parent_settings(settings)
    root = Path(old['runs_root']) / old['campaign']
    jobs = set()
    for path in root.glob('*/*/*/*/resume.json'):
        source = path.parent
        if (source / 'result.json').exists():
            continue
        meta = read_json(source / 'config.json'); key = job_key(meta)
        if key in known:
            continue  # already completed; never run twice
        if key not in approved:
            raise ValueError('Migration outside approved matrix')
        manifest = read_json(path)
        checkpoint = 'resume_%d.pt' % manifest['slot']
        if digest(source / checkpoint) != manifest['sha256']:
            raise ValueError('Migration checkpoint integrity failure')
        destination = Path(settings['runs_root']) / settings['campaign'] / source.relative_to(root)
        destination.mkdir(parents=True, exist_ok=True)
        names = [name for name in ['config.json', 'resume.json', checkpoint, 'best.pt', 'history.json'] if (source / name).exists()]
        record = dict(format='verified-periodic-v3-to-v4-v1', source_directory=str(source),
                      source_hashes={name: digest(source / name) for name in names},
                      original_metadata_identity=identity(meta), manifest=manifest,
                      unchanged_compute_modules=code,
                      policy='Continue committed model, optimizers, RNG, sampling and online state; '
                             'separate periodic-v4 lineage, concurrent GPU trajectories not presumed identical')
        verify_source(record, settings)
        migration = destination / 'migration.json'
        if migration.exists() and read_json(migration) != record:
            raise ValueError('Existing migration identity differs')
        if not (destination / 'resume.json').exists():
            for name in ('best.pt', 'history.json'):
                if name not in record['source_hashes']: continue
                target = destination / name
                if target.exists() and digest(target) != record['source_hashes'][name]:
                    raise ValueError('Existing destination artifact differs')
                if not target.exists():
                    shutil.copyfile(source / name, target)
        atomic_json(migration, record)
        jobs.add(key)
    return jobs


def lineage(folder):
    path = Path(folder) / 'migration.json'
    if not path.exists():
        return None
    record = read_json(path)
    return dict(source_directory=record['source_directory'],
                source_checkpoint_sha256=record['manifest']['sha256'],
                original_metadata_identity=record['original_metadata_identity'],
                source_phase=record['manifest']['phase'], source_cursor=record['manifest']['cursor'],
                policy=record['policy'])


def load_migrated(folder, metadata):
    path = Path(folder) / 'migration.json'
    if not path.exists():
        return None
    record = read_json(path)
    compatible_code()
    previous = verify_source(record, metadata['settings_input'])
    excluded = {'settings_input', 'settings', 'resolved', 'source_fingerprint',
                'execution_version', 'trajectory_policy', 'resume_lineage'}
    if {k: v for k, v in previous.items() if k not in excluded} != {k: v for k, v in metadata.items() if k not in excluded}:
        raise ValueError('Migration model/data/features/environment differ')
    for name in ('settings', 'resolved'):
        if protocol(previous[name]) != protocol(metadata[name]):
            raise ValueError('Migration effective settings differ: ' + name)
    checkpoint = Path(record['source_directory']) / ('resume_%d.pt' % record['manifest']['slot'])
    value = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if value['provenance'] != record['original_metadata_identity']:
        raise ValueError('Migration checkpoint provenance mismatch')
    # This is an explicit, audited identity transition, not bypassed validation.
    value['provenance'] = identity(metadata)
    value['migration'] = lineage(folder)
    return value

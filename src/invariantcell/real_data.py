"""Bounded raw-count loader and immutable, metadata-only context split."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .io import digest, file_hash, load, utc_now, write_json


def strings(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def column(group, name):
    import h5py
    node = group[name]
    if isinstance(node, h5py.Dataset):
        return strings(node[()])
    categories = strings(node['categories'][()])
    codes = node['codes'][()]
    return [categories[i] if i >= 0 else '' for i in codes]


def metadata(path):
    import h5py
    with h5py.File(path, 'r') as f:
        if not isinstance(f['X'], h5py.Dataset):
            raise ValueError('This bounded loader supports dense X only')
        genes = column(f['var'], 'ensembl_id')
        symbols = column(f['var'], f['var'].attrs.get('_index', '_index'))
        labels = column(f['obs'], 'perturbation')
        if len(genes) != len(set(genes)) or any(not g.startswith('ENSG') for g in genes):
            raise ValueError('Require unique Ensembl gene IDs')
        return {'genes': genes, 'symbols': symbols, 'labels': labels,
                'shape': list(f['X'].shape), 'dtype': str(f['X'].dtype),
                'source_cell_line_labels': sorted(set(column(f['obs'], 'cell_line')))}


def choose(items, key, n):
    return sorted(items, key=lambda x: hashlib.sha256(f'17:{key}:{x}'.encode()).hexdigest())[:n]


def prepare(project, cfg):
    """Read IDs/controls only, never expression summaries or response values."""
    project = Path(project)
    if (project / 'configs/split_manifest_real.yaml').exists():
        raise FileExistsError('Frozen real split exists; reuse it, do not silently refreeze')
    archive = load(project / 'audit/scperturb_primary_metadata.json')
    paths = {
        'K562': Path(cfg['discovery']['recursive_root']) / 'non_linear_models/data/raw/ReplogleWeissman2022_K562_essential.h5ad',
        'RPE1': project / 'data/public/ReplogleWeissman2022_rpe1.h5ad',
    }
    sources, schemas = {}, {}
    for context, path in paths.items():
        entry = next(f for f in archive['files'] if f['key'] == path.name)
        md5, sha = hashlib.md5(), hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b''):
                md5.update(block); sha.update(block)
        if path.stat().st_size != entry['size'] or 'md5:' + md5.hexdigest() != entry['checksum']:
            raise ValueError('Archive identity check failed: ' + context)
        schemas[context] = metadata(path)
        sources[context] = {'path': str(path), 'sha256': sha.hexdigest(), 'md5': md5.hexdigest(),
            'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
            'context': context, 'context_basis': 'verified archive filename and original experiment accession',
            'source_cell_line_labels': schemas[context]['source_cell_line_labels'],
            'license': 'CC-BY-4.0', 'accession': archive['doi'], 'organism': 'Homo sapiens',
            'modality': 'CRISPRi', 'X': 'raw UMI counts stored float32; validate every accessed block',
            'shape': schemas[context]['shape'], 'event_time': None,
            'replicate_structure': 'not resolved; cells are not independent context replicates'}
    common = set(schemas['K562']['genes']) & set(schemas['RPE1']['genes'])
    genes = sorted(choose(common, 'genes', 512))
    groups = {}
    for ctx, schema in schemas.items():
        groups[ctx] = {}
        for index, label in enumerate(schema['labels']):
            groups[ctx].setdefault(label, []).append(index)
    targets = set(groups['K562']) & set(groups['RPE1']) - {'control', ''}
    targets = sorted(choose([t for t in targets if len(groups['K562'][t]) >= 96 and len(groups['RPE1'][t]) >= 64], 'targets', 8))
    if len(targets) < 8:
        raise ValueError('Too few adequately sampled shared targets')
    rows = {}
    for ctx in paths:
        if len(groups[ctx].get('control', [])) < 128:
            raise ValueError('Insufficient controls')
        rows[ctx] = {'control': sorted(choose(groups[ctx]['control'], ctx + ':control', 128)), 'fit': {}, 'development': {}, 'evaluation': {}}
        for target in targets:
            selected = choose(groups[ctx][target], ctx + ':' + target, 96 if ctx == 'K562' else 64)
            if ctx == 'K562':
                rows[ctx]['fit'][target] = sorted(selected[:64])
                rows[ctx]['development'][target] = sorted(selected[64:])
            else:
                rows[ctx]['evaluation'][target] = sorted(selected)
    total = sum(len(v) if isinstance(v, list) else sum(map(len, v.values())) for ctx in rows.values() for v in ctx.values())
    if total > cfg['pilot']['real']['max_cells_initial'] or len(targets) > cfg['pilot']['real']['max_targets_initial']:
        raise ValueError('Real pilot configured cell/target cap exceeded')
    split = {'schema_version': 1, 'created_at': utc_now(), 'status': 'FROZEN_BEFORE_RESPONSE_ACCESS',
        'seed': 17, 'training_contexts': ['K562'], 'evaluation_contexts': ['RPE1'],
        'development': 'disjoint K562 cells; no independent development context and no demonstrated calibration transport',
        'genes': genes, 'targets': targets, 'rows': rows, 'total_cells': total,
        'selection_basis': 'seeded SHA256 ordering of IDs, target support >=96 K562 and >=64 RPE1; no response statistics',
        'heldout_interface': 'RPE1 controls only until immutable predictions AND all policy decisions committed',
        'hyperparameters': {'B1_shrinkage': 0.5, 'B2_ridge_alpha': 1.0, 'prediction_cells': 64},
        'preregistered_incumbent': 'B1', 'promotion': 'none from this evaluated context; additional locked contexts required',
        'outcome': 'MSE of log1p group mean raw count on fixed 512 genes',
        'decision': {'reliability_max_error': 0.05, 'budget': 4, 'risk_limit': 0.5, 'minimum_calibration_support': 4},
        'limitations': ['One training context cannot identify context regression effects.', 'Cells are not context replicates.',
                       'Subset mean-expression scores are not official VCC scores.', 'Upstream corpus is not pristine; contamination ledger retained.']}
    manifest = {'schema_version': 1, 'synthetic': False, 'sources': sources,
        'source_rights_report': 'reports/data_source_rights.md', 'identifier_namespace': 'Ensembl gene ID',
        'gene_columns': {ctx: [schemas[ctx]['genes'].index(g) for g in genes] for ctx in paths},
        'metadata_conflict': 'RPE1 source annotation may say K562; experiment accession determines context',
        'nih_empirical_resource_eligibility': 'UNRESOLVED'}
    # JSON is valid YAML; immutable creation avoids a mutable split disguised as config.
    write_json(project, 'configs/data_manifest_real.yaml', manifest, overwrite=False)
    write_json(project, 'configs/split_manifest_real.yaml', split, overwrite=False)
    frozen = {name: file_hash(project / 'configs' / name) for name in ('data_manifest_real.yaml', 'split_manifest_real.yaml')}
    write_json(project, 'state/real_protocol_freeze.json', {'created_at': utc_now(), 'files': frozen, 'responses_read': False}, overwrite=False)
    return {'status': 'REAL_PROTOCOL_FROZEN', 'targets': targets, 'genes': len(genes), 'cells': total, 'hashes': frozen}


def frozen_inputs(project):
    project = Path(project)
    receipt = load(project / 'state/real_protocol_freeze.json')
    for name, expected in receipt['files'].items():
        if file_hash(project / 'configs' / name) != expected:
            raise ValueError('Frozen data/split manifest changed')
    return load(project / 'configs/data_manifest_real.yaml'), load(project / 'configs/split_manifest_real.yaml')


def counts(manifest, split, context, role, target=None, *, actor='predictor'):
    """Selectors never call this; predictor cannot access withheld responses."""
    import h5py
    import numpy as np
    if actor not in {'predictor', 'evaluator'} or role not in {'control', 'fit', 'development', 'evaluation'}:
        raise ValueError('Unknown data access role')
    if role == 'evaluation' and actor != 'evaluator':
        raise PermissionError('Withheld outcomes are evaluator-only')
    if context in split['evaluation_contexts'] and actor != 'evaluator' and role != 'control':
        raise PermissionError('Held-out context exposes only controls to predictor')
    source = manifest['sources'][context]
    path = Path(source['path'])
    if path.stat().st_size != source['bytes'] or path.stat().st_mtime_ns != source['mtime_ns']:
        raise ValueError('Frozen input file changed')
    indices = split['rows'][context][role]
    if target is not None:
        indices = indices[target]
    if not isinstance(indices, list) or not indices:
        raise ValueError('Requested group absent from locked split')
    with h5py.File(path, 'r') as f:
        block = f['X'][indices, :][:, manifest['gene_columns'][context]]
    if not np.isfinite(block).all() or (block < 0).any() or (block != np.floor(block)).any():
        raise ValueError('Documented raw count block is not nonnegative integral data')
    return block.astype(np.int64)



def evaluation_groups(manifest, split, context, *, actor):
    """One batched HDF5 read; evaluator-only, preserving locked row memberships."""
    import copy
    if actor != 'evaluator':
        raise PermissionError('Withheld grouped responses are evaluator-only')
    members = {'control': split['rows'][context]['control'], **split['rows'][context]['evaluation']}
    indices = sorted({i for rows in members.values() for i in rows})
    temporary = copy.deepcopy(split)
    temporary['rows'][context]['evaluation'] = indices
    values = counts(manifest, temporary, context, 'evaluation', actor=actor)
    locations = {index: position for position, index in enumerate(indices)}
    return {name: values[[locations[i] for i in rows]] for name, rows in members.items()}

if __name__ == '__main__':
    import sys
    from .io import configuration
    root = Path(__file__).resolve().parents[2]
    if sys.argv[1:] != ['prepare']:
        raise SystemExit('usage: python -m invariantcell.real_data prepare')
    print(json.dumps(prepare(root, configuration(root)), indent=2))

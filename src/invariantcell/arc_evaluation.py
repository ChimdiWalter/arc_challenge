"""Pinned official metric implementation on a local diagnostic subset only."""
from dataclasses import replace, asdict
from pathlib import Path
import json
import math
import sys

from .io import file_hash, load, write_json


def run(project, cfg, run_id, config):
    import numpy as np
    import pandas as pd
    import anndata as ad
    from cell_eval2 import EvalConfig, compute_metrics
    from importlib.metadata import version
    from .real_data import frozen_inputs, evaluation_groups, metadata
    project = Path(project)
    source_run = load(config)['source_run']
    if '/' in source_run or not source_run.startswith('real-transfer-'):
        raise ValueError('Expected a completed project real-transfer run ID')
    source = project/'runs'/source_run
    completed = load(source/'manifest.json')
    if completed['status'] != 'REAL_CPU_BASELINES_WORKING':
        raise ValueError('Real pilot is not complete')
    for name in ['predictions.json', 'decision_commitment.json', 'evaluation.json']:
        if file_hash(source/name) != completed['artifact_sha256'][name]:
            raise ValueError('Frozen pilot artifact changed')
    data, split = frozen_inputs(project)
    prefix = f'runs/{run_id}'
    write_json(project, f'{prefix}/source_receipt.json', {'source_run': source_run, 'scope': 'local official-metric diagnostic'}, overwrite=False)
    # Outcomes were already opened by the committed evaluator; no new tuning.
    options = load(config)
    cached = options.get('reference_cache')
    if cached:
        cached_path = (project / cached['path']).resolve()
        if not cached_path.is_relative_to(project / 'runs') or file_hash(cached_path) != cached['sha256']:
            raise ValueError('Untrusted evaluated-reference cache')
        reference = ad.read_h5ad(cached_path)
        if list(reference.var_names) != split['genes']:
            raise ValueError('Cached reference gene order differs from frozen split')
        observed = {t: np.asarray(reference.X[reference.obs['target_gene'] == t]) for t in ['control'] + split['targets']}
        if any(len(v) != (128 if t == 'control' else 64) for t,v in observed.items()):
            raise ValueError('Cached reference group dimensions mismatch')
    else:
        observed = evaluation_groups(data, split, 'RPE1', actor='evaluator')
    control = observed.pop('control')
    schema = metadata(data['sources']['RPE1']['path'])
    symbol_to_id = dict(zip(schema['symbols'], schema['genes']))
    target_map = {t: symbol_to_id[t] for t in split['targets'] if t in symbol_to_id}
    def matrix(groups):
        labels = [target for target, values in groups.items() for _ in values]
        values = np.vstack(list(groups.values())).astype(np.float32)
        return ad.AnnData(values, obs=pd.DataFrame({'target_gene': labels}, index=[f'cell_{i}' for i in range(len(labels))]),
            var=pd.DataFrame(index=split['genes']))
    real = matrix({'control': control, **observed})
    real.write_h5ad(project/prefix/'local_reference.h5ad')
    base = EvalConfig.from_preset('vcc2026')
    supported_metrics = ['pds_cosine', 'de_wilcoxon_direction_fidelity_yield_raw', 'de_wilcoxon_direction_reach_raw', 'de_wilcoxon_sig_jaccard', 'de_wilcoxon_lfc_nmae']
    settings = replace(base, metrics=supported_metrics, device='cpu', num_threads=1, gather_threads=1, pert_col='target_gene',
        control='control', de=replace(base.de, backend='scanpy'), target_gene_map=target_map,
        discrimination=replace(base.discrimination, exclude_target_gene=any(g in split['genes'] for g in target_map.values())))
    write_json(project, f'{prefix}/evaluator_config.json', asdict(settings), overwrite=False)
    results = {}
    for method, records in load(source/'predictions.json').items():
        # Controls belong only to local scoring files, never submitted predictions.
        pred = matrix({'control': control, **{r['perturbation']: np.asarray(r['counts']) for r in records}})
        pred.write_h5ad(project/prefix/f'{method}_local_scoring.h5ad')
        frame = compute_metrics(pred, real, config=settings)
        frame.write_csv(project/prefix/f'{method}_official_metric_panel.csv')
        rows = frame.to_dicts()
        for row in rows:
            for key, value in list(row.items()):
                if isinstance(value, float) and not math.isfinite(value):
                    row[key] = None
        results[method] = rows
    write_json(project, f'{prefix}/official_metric_results.json', {'source_run': source_run, 'results': results,
        'scope': 'official vcc2026 metric implementation; nonofficial local subset, no leaderboard/anchor-comparable score',
        'controls': 'reference controls included only in local scoring files',
        'unavailable_scored_member': {'expr_mse_unbiased_capped_norm': 'Official implementation requires target transcripts in the gene panel; none are in the pre-frozen subset. Not computed; no dummy gene or count inserted.'},
        'preset_override': 'Disable panel target exclusion only when every target transcript is absent from frozen 512-gene panel; no own transcript then exists to exclude',
        'subset': {'genes': len(split['genes']), 'cells_per_target': 64, 'targets': len(split['targets']), 'contexts': 1},
        'limitations': ['512-gene normalization differs from full-gene scoring.', 'No challenge control/target bundle supplied.',
            'Cell sampling is small; undefined or unstable metrics retained.', 'No model selected or tuned from this evaluated panel.']}, overwrite=False)
    return {'run_id': run_id, 'status': 'ARC_EVALUATOR_PARTIAL_DIAGNOSTIC_WORKING', 'synthetic': False,
        'source_run': source_run, 'evaluator_version': version('cell-eval2'), 'official_submission_ready': False,
        'command': 'PYTHONPATH=.deps/arc_eval:src python -B -m invariantcell pilot --suite arc-evaluation --config configs/arc_evaluation_pilot.yaml',
        'nih_resource_eligibility': 'UNRESOLVED',
        'artifact_sha256': {str(p.relative_to(project/prefix)): file_hash(p) for p in (project/prefix).rglob('*') if p.is_file()}}

"""One prespecified K562->RPE1 baseline/decision pilot, with delayed evaluation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import digest, file_hash, load, utc_now, write_json


def error(prediction, truth):
    import numpy as np
    return float(np.mean((np.log1p(np.mean(prediction, axis=0)) - np.log1p(np.mean(truth, axis=0)))**2))


def ridge_deltas(deltas, training_control, evaluation_control, alpha):
    """Explicit target + control-expression linear ridge; no hidden tuning.

    A single training context leaves centered context coefficients zero. This is
    an executed ridge baseline, not evidence for learned context conditioning.
    """
    import numpy as np
    n = len(deltas)
    train = np.column_stack((np.eye(n), np.tile(np.log1p(training_control), (n, 1))))
    test = np.column_stack((np.eye(n), np.tile(np.log1p(evaluation_control), (n, 1))))
    mx, my = train.mean(axis=0), deltas.mean(axis=0)
    x, y = train - mx, deltas - my
    coef = x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(n), y)
    return (test - mx) @ coef + my, {'alpha': alpha,
        'context_coefficient_norm': float(np.linalg.norm(coef[n:])),
        'context_effect_identifiable': False, 'training_context_count': 1}


def _record(name, context, target, genes, values, dependency):
    from .core import BiologicalTransition, DependencyRef
    return BiologicalTransition(record_id=f'{name}:{context}:{target}', context=context,
        perturbation=target, genes=tuple(genes), counts=tuple(tuple(int(v) for v in row) for row in values),
        state_kind='predicted', synthetic=False, event_time=None,
        dependencies=(DependencyRef('frozen_real_model_inputs', dependency, None),), method=name)


def run(project, cfg, run_id):
    import numpy as np
    from .real_data import frozen_inputs, counts
    from .core import canonical_hash
    from .decision import freeze_panel, fit_calibration, POLICIES
    from .temporal import snapshot_from_transitions, TemporalQuery, issue_certificate
    from .replay import verify_certificate
    project, prefix = Path(project), f'runs/{run_id}'
    data, split = frozen_inputs(project)
    write_json(project, f'{prefix}/configuration.json', cfg, overwrite=False)
    hp, targets = split['hyperparameters'], split['targets']
    protocol = {'objective': 'reliability_triage', 'objective_version': 'real-log1p-group-mean-MSE-v1',
        'budget': split['decision']['budget'], 'seed': split['seed'], 'knowledge_boundary': utc_now(),
        'strict_historical': False, 'reliability_max_error': split['decision']['reliability_max_error'],
        'calibration_bin_edges': [], 'minimum_calibration_support': split['decision']['minimum_calibration_support'],
        'fixed_rules': {'max_native_uncertainty': None, 'require_supported_evidence': False},
        'risk_limit': split['decision']['risk_limit']}
    write_json(project, f'{prefix}/decision_protocol.json', protocol, overwrite=False)
    controls = {ctx: counts(data, split, ctx, 'control') for ctx in ('K562', 'RPE1')}
    fit = {t: counts(data, split, 'K562', 'fit', t) for t in targets}
    # K562 development cells are disjoint from fit; only these labels calibrate.
    development = {t: counts(data, split, 'K562', 'development', t) for t in targets}
    control_mean = controls['K562'].mean(axis=0)
    deltas = np.stack([fit[t].mean(axis=0) - control_mean for t in targets])
    model_input_hash = digest({'data': data, 'split': split,
        'fit': {t: digest(fit[t].tolist()) for t in targets},
        'controls': {ctx: digest(x.tolist()) for ctx, x in controls.items()}})
    model = {'model_hash': digest({'method': 'B1', 'inputs': model_input_hash, 'shrinkage': hp['B1_shrinkage']}),
             'dependencies': [{'name': 'frozen_real_inputs', 'sha256': model_input_hash, 'available_at': None}]}
    test_predictions, dev_predictions, ridge_diagnostics = {}, {}, {}
    for context, store in [('K562', dev_predictions), ('RPE1', test_predictions)]:
        template = controls[context][:hp['prediction_cells']]
        ridge, diagnostic = ridge_deltas(deltas, control_mean, controls[context].mean(axis=0), hp['B2_ridge_alpha'])
        ridge_diagnostics[context] = diagnostic
        for method, effects in [('B0', np.zeros_like(deltas)), ('B1', hp['B1_shrinkage'] * deltas), ('B2', ridge)]:
            store[method] = [_record(method, context, target, split['genes'],
                np.rint(np.maximum(0, template + effects[i])).astype(np.int64), model_input_hash)
                for i, target in enumerate(targets)]
    # All three predictions are serialized and hashed before evaluator access.
    write_json(project, f'{prefix}/predictions.json', {m: [r.to_dict() for r in records] for m, records in test_predictions.items()}, overwrite=False)
    write_json(project, f'{prefix}/model.json', {'model': model, 'hyperparameters': hp, 'ridge_diagnostics': ridge_diagnostics,
        'fit_delta': deltas.tolist(), 'genes': split['genes'], 'targets': targets,
        'native_uncertainty': 'delta-method training sample-mean variance proxy; not uncertainty about context transfer'}, overwrite=False)
    candidates, dev_rows, bridge = [], [], []
    for i, record in enumerate(test_predictions['B1']):
        snapshot = snapshot_from_transitions((record,), run_id + ':' + record.perturbation,
            generated_at=utc_now(), ingested_at=utc_now())
        evidence = snapshot.evidence[0]
        now = utc_now()
        query = TemporalQuery(evidence.subject, evidence.predicate, evidence.context, now, now, now, allow_predictions=True)
        certificate = issue_certificate(snapshot, query)
        replay = verify_certificate(certificate, snapshot, canonical_hash(snapshot.to_dict()), query)
        if not replay.valid:
            raise ValueError('Real prediction certificate replay failed')
        artifact = {'snapshot': snapshot.to_dict(), 'query': query.to_dict(), 'certificate': certificate, 'receipt': replay.to_dict()}
        cert_path = write_json(project, f'{prefix}/certificates/{record.perturbation}.json', artifact, overwrite=False)
        bridge.append({'target': record.perturbation, 'path': str(cert_path), 'sha256': file_hash(cert_path),
                      'meaning': 'replayed provenance/abstention; missing validity/event times are not fabricated'})
        x = fit[record.perturbation]
        uncertainty = float(np.mean(x.var(axis=0, ddof=1) / len(x) / (1 + x.mean(axis=0))**2))
        candidates.append({'candidate_id': record.record_id, 'context_id': record.context, 'target_id': record.perturbation,
            'intervention_protocol': 'Replogle-CRISPRi', 'prediction_score': -uncertainty,
            'model_hash': model['model_hash'], 'prediction_hash': canonical_hash(record.to_dict()),
            'native_uncertainty': uncertainty, 'native_uncertainty_kind': 'training_log_mean_sampling_variance_proxy',
            'formal_status': 'passed', 'evidence_status': 'missing', 'supporting_evidence_refs': []})
        dev = dev_predictions['B1'][i]
        dev_rows.append({'candidate_id': dev.record_id, 'context_id': 'K562', 'split': 'development',
            'model_hash': model['model_hash'], 'native_uncertainty': uncertainty,
            'native_uncertainty_kind': 'training_log_mean_sampling_variance_proxy',
            'agreement_error': error(dev.counts, development[dev.perturbation]), 'available_at': protocol['knowledge_boundary']})
    panel = freeze_panel(candidates, protocol, model)
    calibration = fit_calibration(dev_rows, panel, protocol)
    for name, value in [('panel.json', panel), ('calibration.json', calibration), ('development_rows.json', dev_rows), ('bridge.json', bridge)]:
        write_json(project, f'{prefix}/{name}', value, overwrite=False)
    env = dict(os.environ, PYTHONPATH=str(project / 'src'), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    decision_files, timings = {}, {}
    for policy in POLICIES:
        output = f'{prefix}/decisions/{policy}.json'
        args = [sys.executable, '-B', '-m', 'invariantcell', 'decision-select', '--frozen-panel', str(project/prefix/'panel.json'),
            '--protocol', str(project/prefix/'decision_protocol.json'), '--calibration', str(project/prefix/'calibration.json'),
            '--policy', policy, '--output', output]
        start = time.monotonic()
        child = subprocess.run(args, cwd=project, env=env, capture_output=True, text=True, timeout=30)
        timings[policy] = time.monotonic() - start
        if child.returncode:
            raise RuntimeError(child.stderr[:1500])
        decision_files[output] = file_hash(project / output)
    commitment = {'at': utc_now(), 'panel_hash': panel['panel_hash'], 'files': decision_files,
        'predictions_sha256': file_hash(project/prefix/'predictions.json'), 'protocol_sha256': file_hash(project/prefix/'decision_protocol.json'),
        'boundary': 'separate selector and evaluator processes, no OS access-control sandbox',
        'data_freeze_sha256': file_hash(project/'state/real_protocol_freeze.json')}
    write_json(project, f'{prefix}/decision_commitment.json', commitment, overwrite=False)
    child = subprocess.run([sys.executable, '-B', '-m', 'invariantcell.real_pilot', 'evaluate', str(project), prefix],
        cwd=project, env=env, capture_output=True, text=True, timeout=300)
    if child.returncode:
        raise RuntimeError('Real evaluator failed: ' + child.stderr[:2000])
    write_json(project, f'{prefix}/decision_resources.json', {'selector_wall_seconds': timings,
        'external_tool_calls': 0, 'evidence_retrieval_calls': 0, 'calibration_transfer': 'not validated on independent contexts'}, overwrite=False)
    sources = {str(p.relative_to(project)): file_hash(p) for p in (project/'src/invariantcell').glob('*.py')}
    return {'run_id': run_id, 'synthetic': False, 'status': 'REAL_CPU_BASELINES_WORKING',
        'scope': 'one heldout context, eight targets, 512 genes; diagnostic only, not official Arc scoring',
        'command': 'PYTHONPATH=src python -B -m invariantcell pilot --suite real-transfer',
        'configuration_sha256': digest(cfg), 'code_sha256': digest(sources), 'code_files': sources,
        'environment': {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__},
        'seed': split['seed'], 'frozen_protocol': file_hash(project/'state/real_protocol_freeze.json'),
        'prediction_panel_hash': panel['panel_hash'], 'decision_protocol_hash': digest(protocol),
        'preregistered_incumbent': 'B1', 'test_based_promotion': False, 'nih_eligibility': 'UNRESOLVED',
        'artifact_sha256': {str(p.relative_to(project/prefix)): file_hash(p) for p in (project/prefix).rglob('*') if p.is_file()},
        'limitations': split['limitations'] + ['Calibration uses within-K562 development cells; context transport is unproven.',
           'Rounded additive count shifts do not model a calibrated single-cell distribution.',
           'Formal replay validates a declared computation; it does not confirm a biological prediction.']}


def evaluate_files(project, prefix):
    import numpy as np
    from .real_data import frozen_inputs, counts
    from .decision import evaluate, verify_decisions
    project, base = Path(project), Path(project)/prefix
    commitment = load(base/'decision_commitment.json')
    panel, protocol, calibration = (load(base/name) for name in ('panel.json', 'decision_protocol.json', 'calibration.json'))
    if file_hash(base/'predictions.json') != commitment['predictions_sha256'] or file_hash(base/'decision_protocol.json') != commitment['protocol_sha256']:
        raise ValueError('Committed prediction/protocol changed')
    if file_hash(project/'state/real_protocol_freeze.json') != commitment['data_freeze_sha256']:
        raise ValueError('Frozen real protocol receipt changed')
    decisions = {}
    for path, expected in commitment['files'].items():
        if file_hash(project/path) != expected:
            raise ValueError('Committed selection changed')
        d = load(project/path)
        verify_decisions(d, panel, protocol, calibration)
        decisions[d['policy']] = d
    data, split = frozen_inputs(project)
    write_json(project, f'{prefix}/locked_outcome_access.json', {'opened_at': utc_now(),
        'actor': 'evaluator_subprocess', 'context': 'RPE1', 'role': 'evaluation',
        'decision_commitment_sha256': file_hash(base/'decision_commitment.json')}, overwrite=False)
    truth = {t: counts(data, split, 'RPE1', 'evaluation', t, actor='evaluator') for t in split['targets']}
    predictions = load(base/'predictions.json')
    metrics, outcomes = {}, []
    for method, records in predictions.items():
        by_target = {}
        for record in records:
            observed = truth[record['perturbation']]
            pred = np.asarray(record['counts'])
            mse = error(pred, observed)
            by_target[record['perturbation']] = {'log1p_mean_mse': mse,
                'raw_mean_mse': float(np.mean((pred.mean(0)-observed.mean(0))**2)),
                'mean_absolute_error': float(np.mean(np.abs(pred.mean(0)-observed.mean(0))))}
            if method == 'B1':
                outcomes.append({'candidate_id': record['record_id'], 'agreement_error': mse})
        metrics[method] = {'mean_log1p_mean_mse': float(np.mean([v['log1p_mean_mse'] for v in by_target.values()])), 'per_target': by_target}
    report = evaluate(decisions, panel, protocol, outcomes, calibration)
    report.update(synthetic=False, outcome_definition=split['outcome'], baselines=metrics,
        evaluation_contexts=['RPE1'], no_test_based_model_selection=True,
        uncertainty_scope='one test context: no defensible across-context confidence interval',
        official_arc_evaluation='NOT_RUN; local metric proxy')
    write_json(project, f'{prefix}/evaluation.json', report, overwrite=False)
    write_json(project, f'{prefix}/evaluated_outcomes.json', {'records': outcomes,
        'purpose': 'evaluator-only diagnostic output; these units are now exposed and cannot become a fresh blind test'}, overwrite=False)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != 'evaluate':
        raise SystemExit('usage: python -m invariantcell.real_pilot evaluate PROJECT RUN_PREFIX')
    evaluate_files(sys.argv[2], sys.argv[3])

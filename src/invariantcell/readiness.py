"""Read-only checkpoint and Arc input readiness; never opens response matrices."""
from pathlib import Path

from .io import file_hash, load, output_path, utc_now


def arc_preflight(project, cfg):
    project = Path(project)
    value = cfg.get('arc', {}).get('official_bundle_dir')
    result = {'checked_at': utc_now(), 'bundle_dir': value, 'status': 'OFFICIAL_BUNDLE_NOT_CONFIGURED',
              'submission_ready': False, 'expression_values_read': False, 'files': [],
              'remaining': ['official bundle identity/phase verification', 'full-gene prediction and evaluation',
                            'official CLI prep receipt', 'artifact-specific submission approval']}
    if not value:
        return result
    supplied = Path(value)
    if not supplied.is_absolute():
        raise ValueError('official_bundle_dir must be an explicit absolute path')
    root = supplied.resolve()
    approved = [Path(p).resolve() for p in [cfg['workspace'], *cfg.get('approved_search_roots', []),
                *cfg.get('data_roots', [])] if p and not str(p).startswith('/your/')]
    if not any(root.is_relative_to(p) for p in approved):
        result['status'] = 'BUNDLE_OUTSIDE_APPROVED_INPUT_ROOTS'
        return result
    if not root.is_dir():
        result['status'] = 'CONFIGURED_BUNDLE_UNAVAILABLE'
        return result
    missing = []
    for name in ('gene_names.csv', 'pert_counts.csv'):
        p = root/name
        if not p.resolve().is_relative_to(root):
            missing.append(name + ': out-of-scope link')
        elif not p.is_file():
            missing.append(name)
        else:
            result['files'].append({'name': name, 'bytes': p.stat().st_size})
    controls = []
    for p in sorted(root.glob('context_*.h5ad')):
        if p.resolve().is_relative_to(root) and p.is_file():
            controls.append({'name': p.name, 'bytes': p.stat().st_size})
    result.update(files=result['files']+controls, missing=missing,
                  status='INPUT_FILES_PRESENT_NOT_VALIDATED' if not missing and controls else 'BUNDLE_FILES_INCOMPLETE')
    # Filenames cannot authenticate a phase, context identity, schema or counts.
    return result


def status(project, cfg, verify_artifacts=False):
    project = Path(project)
    state = load(project/'state/resume.json')
    budget_relative = cfg['compute'].get('experiment_budget_path', 'state/experiment_budget.json')
    budget = load(output_path(project, budget_relative))
    tests = load(project/'reports/engineering_tests.json')
    verified = {}
    if verify_artifacts:
        for field in ('synthetic_run_id', 'real_run_id', 'arc_evaluation_run_id'):
            run = output_path(project, 'runs/'+state[field])
            if not run.is_relative_to((project/'runs').resolve()):
                raise ValueError('Run identifier escapes run store')
            manifest = load(run/'manifest.json')
            for name, expected in manifest['artifact_sha256'].items():
                if file_hash(output_path(run, name)) != expected:
                    raise ValueError('Run artifact hash mismatch: '+name)
            verified[state[field]] = len(manifest['artifact_sha256'])
    cap = cfg['compute']['max_cumulative_experiment_wall_minutes']*60
    telemetry = cfg['compute'].get('cumulative_experiment_time_is_telemetry_only', False)
    return {'checked_at': utc_now(), 'status': state['status'], 'readiness': state['readiness'],
            'tests': {'passed': tests['passed'], 'executed': tests['executed'], 'receipt': 'reports/engineering_tests.json'},
            'experiments': {'ledger': budget_relative, 'used_seconds': budget['used_wall_seconds'], 'reserved_seconds': budget['reserved_wall_seconds'],
                'remaining_seconds': None if telemetry else max(0, cap-budget['used_wall_seconds']-budget['reserved_wall_seconds']),
                'cumulative_time_policy': 'telemetry_only' if telemetry else 'hard_cap',
                'recorded_cap_seconds': cap,
                'lock_present': (project/'state/experiment.lock').exists()},
            'artifact_verification': verified if verify_artifacts else 'NOT_REQUESTED',
            'arc_inputs': arc_preflight(project, cfg), 'next_action': state['next_action'],
            'research_results': 'Preserved prior results; no new fitting or outcome joins by this command',
            'nih_eligibility': cfg['compute'].get('nih_resource_eligibility', 'UNRESOLVED')}

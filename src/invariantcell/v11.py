"""Strict v11 namespace validation; scientific requirements are not completion claims."""
from pathlib import Path
import yaml
from .io import file_hash
from .post_confirmation import validate_namespace
CONFIG='docs/v11/post_confirmation_v11.yaml'
CONFIG_SHA256='b2fd7e9399c6025f2ee6c2558a2ae2fcd8497000cebe990dc1ff8e08364230ad'

def admit_task(project, task):
    """Reuse inherited finite resource checks without changing archived v10 code."""
    from .post_confirmation import admit_task as inherited_admission
    validate(project)
    if task.get('boundary') != 'post_confirmation_v11':
        raise ValueError('Explicit v11 task boundary required')
    receipt = inherited_admission(project, {**task, 'boundary': 'post_confirmation_v10'})
    return {**receipt, 'boundary': 'post_confirmation_v11'}

def validate(project):
    p=Path(project);path=p/CONFIG
    if file_hash(path)!=CONFIG_SHA256:raise ValueError('Reviewed v11 specification changed')
    expected=yaml.safe_load(path.read_text());actual=yaml.safe_load((p/'inputs.local.yaml').read_text()).get('post_confirmation_v11')
    if not isinstance(actual,dict):raise ValueError('v11 configuration missing')
    fields=validate_namespace(actual,expected)
    a=actual['arc']
    if a['contexts_per_round']*a['perturbations_per_round']*a['cells_per_perturbation']!=360000:raise ValueError('Official round cardinality differs')
    if set(a['required_metrics'])!={'pds','mse','jac','nmae','fid','reach'} or a['use_group_mean_mse_as_primary'] is not False:raise ValueError('Six-dimensional v11 evaluation required')
    requirements=[]
    for key in fields:
        if key.startswith('arc.'):
            consumer='scripts/arc_v10_population.py; reports/arc/six_metric_report.md; existing official input/format bridge'
            status='CONFIGURATION_VALIDATED; PUBLIC_SUBSET_EXECUTED; OFFICIAL_BUNDLE_COVERAGE_PENDING'
        elif key.startswith('boundaries.'):
            consumer='invariantcell.post_confirmation.validate; state/jurkat_response_access.json';status='ENFORCED_ARCHIVED_BOUNDARY'
        elif key.startswith('external_actions.'):
            consumer='Existing local-only worker allowlist and inherited inputs.local.yaml permissions';status='NO_EXTERNAL_ACTION_AUTHORIZED'
        elif key.startswith('nih.'):
            consumer='reports/nih/phase1_rubric_matrix.md; ontology/; state/post_confirmation_resource_attestation.json';status='REQUIREMENT_TRACKED; SUBMISSION_NOT_READY'
        elif key.startswith('mike.'):
            consumer='reports/decision/post_confirmation_protocol.md; reports/reasoning/biomedical_operator_transfer.md';status='MATCHED_PRIOR_RESULTS_PRESERVED; PROCEDURE_TRANSFER_PENDING'
        elif key.startswith('conformacell.'):
            consumer='reports/conformacell/observed_bridge_status.md; configs/conformacell_bridge_v11.json';status='SYNTHETIC_PILOT_EXECUTED; REAL_INDEPENDENT_ENDPOINT_PENDING'
        elif key.startswith('pathology.'):
            consumer='scripts/pathology_v10.py; reports/pathology/real_measurement_validation.md';status='REAL_REFERENCE_SCORED; GENUINE_REVISION_HISTORY_PENDING'
        else:consumer='state/post_confirmation_v11_boundary.json';status='DIRECTIVE_IDENTITY'
        requirements.append({'field':key,'consumer':consumer,'status':status})
    return {'status':'VALIDATED','config_sha256':CONFIG_SHA256,'namespace':'post_confirmation_v11','field_count':len(fields),'requirements':requirements,'flags_are_not_empirical_claims':True}

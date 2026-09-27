"""Immutable pre-confirmation commitments. No expression readers live here.

Hashes attest to supplied artifacts and declarations, not scientific adequacy or
OS-enforced isolation. A pending checklist is never a valid freeze receipt.
"""
from pathlib import Path
import json
from .io import digest,file_hash,load as load_yaml,output_path,write_json,utc_now

def load(path):
    # JSON exponent numbers must not be coerced to strings by YAML1.1.
    try:value=json.loads(Path(path).read_text())
    except json.JSONDecodeError:value=load_yaml(path)
    if not isinstance(value,dict):raise ValueError('Record mapping required')
    return value


ROLES = ('model','preprocessing','genes_order','targets','context_interface','normalization',
         'uncertainty','calibration','policies','reliability','decision_endpoint',
         'evaluation_code','external_evidence','development_comparisons')
CHECKS = ('development_only_selection','state_optimization_reviewed','public_baseline_feasibility_reviewed',
          'requested_challengers_resolved','all_three_development_contexts_scored',
          'uncertainty_and_calibration_fitted_without_confirmation','evidence_admissibility_reviewed',
          'evaluation_code_tested','confirmation_responses_unopened')
RECEIPT='state/pre_jurkat_freeze.json'
COMMITMENT='state/jurkat_prediction_decision_commitment.json'
ACCESS='state/jurkat_response_access.json'


def _files(project,paths):
    if not isinstance(paths,list) or not paths or len(set(paths))!=len(paths):
        raise ValueError('Each role requires distinct supplied artifact paths')
    result={}
    for relative in paths:
        if not isinstance(relative,str) or Path(relative).is_absolute():
            raise ValueError('Freeze artifacts must be workspace-relative')
        path=output_path(project,relative)
        if not path.is_file():raise FileNotFoundError(path)
        result[relative]=file_hash(path)
    return result


def _panel_contract(project):
    model=load(Path(project)/'configs/final_development_model.yaml')
    policy=load(Path(project)/'configs/final_decision_protocol.yaml')
    panel=model.get('confirmation_panel',{})
    if panel.get('context')!='Jurkat':raise ValueError('Explicit confirmation panel required')
    for key in ('targets','gene_ids'):
        values=panel.get(key)
        if not isinstance(values,list) or not values or any(not isinstance(v,str) or not v for v in values) or len(set(values))!=len(values):
            raise ValueError('Unique explicit panel targets and gene order required')
    if policy.get('candidate_count')!=len(panel['targets']) or type(policy.get('budget')) is not int or not 0<policy['budget']<=len(panel['targets']):
        raise ValueError('Policy budget and candidate universe must match frozen panel')
    return panel,policy


def freeze_bundle(project,roles,checks,notes):
    """Seal only a completed development selection; exclusive write preserves it."""
    project=Path(project)
    if (project/RECEIPT).exists() or (project/ACCESS).exists():
        raise FileExistsError('Existing freeze/access is immutable; use an explicitly separate exploratory branch')
    if set(roles)!=set(ROLES) or set(checks)!=set(CHECKS):
        raise ValueError('Incomplete or undeclared freeze roles/checks')
    if any(value is not True for value in checks.values()):
        raise ValueError('Incomplete development prerequisite')
    if not isinstance(notes,dict) or not notes.get('limitations'):
        raise ValueError('Explicit scope and limitations required')
    model=load(project/'configs/final_development_model.yaml')
    policy=load(project/'configs/final_decision_protocol.yaml')
    if model.get('status')!='READY_FOR_FREEZE' or policy.get('status')!='READY_FOR_FREEZE':
        raise ValueError('Draft configuration cannot authorize confirmation')
    if model.get('development_contexts')!=['K562','RPE1','HepG2'] or model.get('confirmation_contexts')!=['Jurkat']:
        raise ValueError('Unexpected context partition')
    if policy.get('policies')!=['P0','P1','P2','P3','P4'] or policy.get('primary_endpoint')!='selected_mean_error_at_fixed_budget':
        raise ValueError('Continuous matched-policy endpoint is required')
    if type(policy.get('budget')) is not int or policy['budget']<1:
        raise ValueError('Positive fixed budget required')
    if 'configs/final_development_model.yaml' not in roles['model'] or 'configs/final_decision_protocol.yaml' not in roles['policies']:
        raise ValueError('Final configurations must be sealed under their roles')
    _panel_contract(project)
    calibration=policy.get('calibration_path')
    code_paths=policy.get('selection_code_paths')
    if calibration not in roles['calibration'] or not isinstance(code_paths,list) or not code_paths:
        raise ValueError('Frozen calibration and selection code paths required')
    pinned={role:_files(project,roles[role]) for role in ROLES}
    all_pins={k:v for files in pinned.values() for k,v in files.items()}
    if any(path not in all_pins for path in code_paths):raise ValueError('Selection code must be frozen')
    if model.get('protocol') not in all_pins:raise ValueError('Referenced input protocol must be frozen')
    body={'schema_version':1,'status':'FROZEN_BEFORE_CONFIRMATION','created_at':utc_now(),
          'roles':pinned,'checks':checks,'notes':notes,'confirmation_responses_opened':False}
    receipt={**body,'freeze_hash':digest(body)}
    write_json(project,RECEIPT,receipt,overwrite=False)
    return receipt


def verify_freeze(project):
    project=Path(project);r=load(project/RECEIPT)
    body={k:v for k,v in r.items() if k!='freeze_hash'}
    if r.get('status')!='FROZEN_BEFORE_CONFIRMATION' or digest(body)!=r.get('freeze_hash'):
        raise ValueError('Invalid freeze receipt')
    if set(r.get('roles',{}))!=set(ROLES) or set(r.get('checks',{}))!=set(CHECKS) or any(v is not True for v in r['checks'].values()):
        raise ValueError('Incomplete freeze receipt')
    for role,files in r['roles'].items():
        if _files(project,list(files))!=files:raise ValueError('Frozen artifact changed: '+role)
    if r.get('confirmation_responses_opened') is not False:raise ValueError('Invalid pre-access declaration')
    _panel_contract(project)
    model=load(project/'configs/final_development_model.yaml');policy=load(project/'configs/final_decision_protocol.yaml')
    pins={k:v for files in r['roles'].values() for k,v in files.items()}
    if model.get('protocol') not in pins:raise ValueError('Referenced input protocol is not sealed')
    if model.get('status')!='READY_FOR_FREEZE' or policy.get('status')!='READY_FOR_FREEZE':raise ValueError('Draft configuration cannot authorize access')
    if policy.get('calibration_path') not in r['roles']['calibration']:raise ValueError('Unsealed calibration')
    if not policy.get('selection_code_paths') or any(k not in pins for k in policy['selection_code_paths']):raise ValueError('Unsealed selection code')
    return r


def _validate_panel_commitment(project,f,prediction_paths,decisions_path,selection_audit_path):
    """Validate actual finite predictions and replay every policy on the bound panel."""
    import numpy as np
    from .decision_v2 import select
    panel,policy=_panel_contract(project)
    prediction_files=_files(project,prediction_paths)
    manifests=[name for name in prediction_paths if name.endswith('.json')]
    if len(manifests)!=1 or len(prediction_paths)!=2:
        raise ValueError('Exactly one panel manifest and one NPY mean-prediction array required')
    manifest_path=manifests[0];manifest=load(output_path(project,manifest_path))
    array_path=manifest.get('array_path')
    if not isinstance(array_path,str) or not array_path.endswith('.npy') or set(prediction_paths)!={manifest_path,array_path}:
        raise ValueError('Prediction manifest must bind exactly the supplied array')
    if (manifest.get('status')!='COMPLETE_PREDICTION_PANEL' or manifest.get('freeze_hash')!=f['freeze_hash']
        or manifest.get('context')!=panel['context'] or manifest.get('targets')!=panel['targets']
        or manifest.get('gene_ids_sha256')!=digest(panel['gene_ids'])
        or manifest.get('array_sha256')!=prediction_files[array_path]
        or manifest.get('semantics')!='predicted_raw_group_mean_counts'):
        raise ValueError('Prediction panel identity/semantics differ from freeze')
    predictions=np.load(output_path(project,array_path),allow_pickle=False,mmap_mode='r')
    if (predictions.shape!=(len(panel['targets']),len(panel['gene_ids'])) or predictions.dtype.kind not in 'fiu'
        or not np.isfinite(predictions).all() or (predictions<0).any()):
        raise ValueError('Incomplete or invalid numerical prediction panel')
    decisions=load(output_path(project,decisions_path))
    if (decisions.get('freeze_hash')!=f['freeze_hash'] or decisions.get('prediction_manifest_path')!=manifest_path
        or decisions.get('prediction_manifest_sha256')!=prediction_files[manifest_path]
        or set(decisions.get('policies',{}))!={'P0','P1','P2','P3','P4'}):
        raise ValueError('Missing frozen policy decisions or prediction binding')
    calibrated_path=policy.get('calibration_path')
    if calibrated_path not in f['roles']['calibration']:raise ValueError('Calibration must be frozen under its role')
    calibration=load(output_path(project,calibrated_path))
    for field,key in [('base_calibrator','base'),('evidence_calibrator','evidence'),('protocol','protocol')]:
        if not isinstance(calibration.get(key),dict) or decisions.get(field)!=calibration[key]:
            raise ValueError('Decision calibration/protocol differs from freeze')
    protocol=decisions['protocol']
    if protocol.get('budget')!=policy['budget'] or protocol.get('primary_endpoint')!=policy['primary_endpoint']:
        raise ValueError('Decision endpoint differs from frozen configuration')
    for calibrator in (calibration['base'],calibration['evidence']):
        if calibrator.get('development_contexts')!=['HepG2','K562','RPE1']:
            raise ValueError('Final calibrator must be development-only across all three contexts')
    candidates=decisions.get('candidates')
    if not isinstance(candidates,list) or len(candidates)!=len(panel['targets']):raise ValueError('Incomplete candidate universe')
    for i,(target,candidate) in enumerate(zip(panel['targets'],candidates)):
        if (candidate.get('context')!='Jurkat' or candidate.get('target')!=target
            or candidate.get('candidate_id')!='Jurkat:'+target
            or candidate.get('prediction_hash')!=digest(predictions[i].astype(float).tolist())):
            raise ValueError('Candidate identity/prediction differs from committed panel')
    snapshots=decisions.get('evidence_snapshots')
    if not isinstance(snapshots,dict) or set(snapshots)!={c['candidate_id'] for c in candidates}:
        raise ValueError('Every candidate must resolve to an actual evidence snapshot')
    model=load(Path(project)/'configs/final_development_model.yaml')
    source_protocol=load(output_path(project,model['protocol']))
    expected_source=source_protocol.get('sources',{}).get('Jurkat',{}).get('sha256')
    allowed_controls=source_protocol.get('rows',{}).get('Jurkat',{}).get('control')
    if not isinstance(expected_source,str) or len(expected_source)!=64 or not isinstance(allowed_controls,list) or not allowed_controls:
        raise ValueError('Frozen protocol must identify allowed control-source rows')
    for candidate in candidates:
        snapshot=snapshots[candidate['candidate_id']]
        assessment={k:v for k,v in candidate['evidence'].items() if k!='snapshot_hash'}
        if (not isinstance(snapshot,dict) or digest(snapshot)!=candidate['evidence']['snapshot_hash']
            or snapshot.get('assessment')!=assessment or snapshot.get('candidate_id')!=candidate['candidate_id']
            or snapshot.get('context')!='Jurkat' or snapshot.get('target')!=candidate['target']
            or snapshot.get('kind')!='observed_control_measurement' or snapshot.get('source_sha256')!=expected_source
            or snapshot.get('source_control_rows')!=allowed_controls):
            raise ValueError('Evidence snapshot identity, assessment or control-source binding differs')
        wanted={key:candidate['features'][key] for key in ('target_detection_fraction','target_control_log_mean')}
        if snapshot.get('observed_control_features')!=wanted:
            raise ValueError('Evidence features differ from measured-control snapshot')
    for name,decision in decisions['policies'].items():
        if decision!=select(candidates,protocol,calibration['base'],calibration['evidence'],name):
            raise ValueError('Frozen policy replay failed')
    audit=load(output_path(project,selection_audit_path))
    if audit.get('freeze_hash')!=f['freeze_hash'] or audit.get('response_store_opens')!=[] or audit.get('completed') is not True:
        raise ValueError('Selection process access audit incomplete or exposed')
    if not audit.get('mechanism') or not audit.get('limitations'):
        raise ValueError('Access audit scope must be explicit')
    artifacts=_files(project,prediction_paths+[decisions_path])
    if audit.get('artifact_sha256')!=artifacts:raise ValueError('Access audit does not bind exact predictions and decisions')
    code_paths=policy.get('selection_code_paths')
    if not isinstance(code_paths,list) or not code_paths:raise ValueError('No frozen selection code')
    codes=_files(project,code_paths);pins={k:v for files in f['roles'].values() for k,v in files.items()}
    if any(pins.get(k)!=v for k,v in codes.items()) or audit.get('code_sha256')!=codes:
        raise ValueError('Audit does not bind frozen selection code')
    auxiliary=audit.get('auxiliary_artifact_sha256',{})
    if not isinstance(auxiliary,dict):raise ValueError('Invalid auxiliary audit artifact map')
    if auxiliary and _files(project,list(auxiliary))!=auxiliary:raise ValueError('Auxiliary audit artifact changed')
    return _files(project,prediction_paths+[decisions_path,selection_audit_path]+list(auxiliary))


def commit_predictions(project,prediction_paths,decisions_path,selection_audit_path):
    """Commit a complete checked panel and replayed P0-P4 decisions before access.

    The bound audit still is not proof of OS isolation or biological adequacy.
    """
    project=Path(project);f=verify_freeze(project)
    if (project/ACCESS).exists():raise PermissionError('Confirmation already accessed')
    files=_validate_panel_commitment(project,f,prediction_paths,decisions_path,selection_audit_path)
    body={'schema_version':2,'created_at':utc_now(),'freeze_hash':f['freeze_hash'],'files':files,
          'prediction_paths':prediction_paths,'decisions_path':decisions_path,'selection_audit_path':selection_audit_path,
          'status':'PREDICTIONS_AND_DECISIONS_COMMITTED_BEFORE_RESPONSES'}
    committed={**body,'commitment_hash':digest(body)}
    write_json(project,COMMITMENT,committed,overwrite=False)
    return committed


def reserve_single_evaluation(project):
    """Consume the one-time boundary before an evaluator opens any responses.

    A crash remains a consumed/failed attempt, never a silent fresh retry. This
    function performs no data access; the separately reviewed evaluator calls it.
    """
    project=Path(project);f=verify_freeze(project);c=load(project/COMMITMENT)
    if c.get('status')!='PREDICTIONS_AND_DECISIONS_COMMITTED_BEFORE_RESPONSES' or c.get('freeze_hash')!=f['freeze_hash']:
        raise ValueError('Prediction commitment does not match freeze')
    if digest({k:v for k,v in c.items() if k!='commitment_hash'})!=c.get('commitment_hash') or _files(project,list(c['files']))!=c['files']:
        raise ValueError('Prediction/decision commitment changed')
    validated=_validate_panel_commitment(project,f,c['prediction_paths'],c['decisions_path'],c['selection_audit_path'])
    if c.get('schema_version')!=2 or validated!=c['files']:raise ValueError('Incomplete commitment cannot authorize access')
    record={'status':'EVALUATION_ACCESS_RESERVED_ONCE','reserved_at':utc_now(),'freeze_hash':f['freeze_hash'],
            'commitment_hash':c['commitment_hash'],'attempts':1,
            'interpretation':'Any following response access consumes untouched-confirmation status. Failure does not restore it.'}
    write_json(project,ACCESS,record,overwrite=False)
    return record

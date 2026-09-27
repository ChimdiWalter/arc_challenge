"""Strict v10 configuration and finite-task admission; archived science is immutable."""
from pathlib import Path
import json
import math
import os
import platform
import shutil
import yaml
from .io import file_hash,utc_now
from .prefreeze import verify_freeze,COMMITMENT,ACCESS

PATCH='docs/v10/post_confirmation_config_patch_v10.yaml'
PATCH_SHA256='5a4a7460387b9492a6139e1dae5929239e8e584bbb73e310932f20e0dcb4bb00'
BOUNDARY='post_confirmation_v10'

def read(path):return json.loads(Path(path).read_text())
def leaves(obj,prefix=''):
    out={}
    for key,value in obj.items():
        name=prefix+'.'+key if prefix else key
        if isinstance(value,dict):out.update(leaves(value,name))
        else:out[name]=value
    return out

def validate_namespace(actual,expected):
    a,b=leaves(actual),leaves(expected)
    if set(a)!=set(b):raise ValueError('Unknown/missing v10 fields: '+repr(sorted(set(a)^set(b))))
    for key,value in a.items():
        if type(value)is not type(b[key]) or value!=b[key]:raise ValueError('Unsupported v10 setting requires reviewed implementation: '+key)
    return a

def validate(project,purpose='engineering'):
    project=Path(project).resolve();patch=project/PATCH
    if file_hash(patch)!=PATCH_SHA256:raise ValueError('Directive patch digest differs from verified bundle')
    expected=yaml.safe_load(patch.read_text())['post_confirmation'];config=yaml.safe_load((project/'inputs.local.yaml').read_text())
    fields=validate_namespace(config.get('post_confirmation',{}),expected)
    if Path(expected['workspace']).resolve()!=project:raise ValueError('Wrong v10 workspace')
    if config['permissions'].get('use_paid_services') is not False or config['compute'].get('max_paid_service_usd')!=0:raise ValueError('No new paid authority')
    freeze=verify_freeze(project);commit=read(project/COMMITMENT);access=read(project/ACCESS)
    if access['attempts']!=1 or access['freeze_hash']!=freeze['freeze_hash'] or access['commitment_hash']!=commit['commitment_hash']:raise ValueError('Archived confirmation boundary differs')
    for name,sha in commit['files'].items():
        if file_hash(project/name)!=sha:raise ValueError('Archived commitment artifact changed')
    if purpose not in ('engineering','arc','decision','conformacell','pathology','nih','discovery'):raise ValueError('Unknown task purpose')
    resource=None
    if purpose=='nih':
        resource=read(project/'state/post_confirmation_resource_attestation.json')
        if resource.get('nonfederal_basis')!='USER_ATTESTED_DOCUMENTATION_PENDING' or resource.get('known_prohibited_federal_support') is not False:
            raise PermissionError('NIH-directed development requires a nonprohibited resource basis; no relabel/rerun shortcut')
    return {'boundary':BOUNDARY,'fields':fields,'configuration':config,'archive_freeze_hash':freeze['freeze_hash'],
        'nih_resource_basis':resource,'host':platform.node(),'python':platform.python_version(),
        'effective_permissions':'Inherited; namespaced scientific requirements do not grant new authority'}

def protected_files(project):
    project=Path(project);freeze=verify_freeze(project);commit=read(project/COMMITMENT)
    return set(commit['files'])|{p for role in freeze['roles'].values() for p in role}|{'state/pre_jurkat_freeze.json',COMMITMENT,ACCESS}

def admit_task(project,task):
    project=Path(project).resolve();checked=validate(project,task['purpose']);config=checked['configuration']
    required={'name','boundary','purpose','script','args','input_sha256','outputs','completion_test','resources','status'}
    if set(task)!=required:raise ValueError('Unknown/incomplete task fields')
    if task['boundary']!=BOUNDARY or task['status']!='READY' or not task['completion_test']:raise ValueError('Task not explicitly ready')
    if not isinstance(task['args'],list) or not all(isinstance(a,str) for a in task['args']):raise ValueError('Invalid argv')
    resources=task['resources'];required_resources={'cpu_threads','ram_gib','wall_seconds','download_bytes','workspace_growth_bytes','gpu_hours','paid_usd'}
    if set(resources)!=required_resources or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in resources.values()):raise ValueError('Finite nonnegative task resources required')
    ram_cap=8
    if task['script']=='scripts/arc_native_prep_v12.py':
        approval=read(project/'state/local_packaging_resource_approval_v12.json')
        if approval.get('user_approved') is not True or config.get('finalization_v13',{}).get('schema_version')!=13:raise PermissionError('Native prep cap change is not authorized')
        ram_cap=min(32,config['compute'].get('max_ram_gib',8),approval['ram_gib'])
    if resources['cpu_threads']!=1 or not 0<resources['ram_gib']<=ram_cap or not 0<resources['wall_seconds']<=min(1800,60*config['compute']['max_single_process_wall_minutes']):raise ValueError('Task exceeds inherited process limits')
    if resources['gpu_hours']!=0 or resources['paid_usd']!=0:raise ValueError('First v10 batch is explicitly CPU/no-charge only')
    budget=read(project/'state/competition_development_budget.json')
    if resources['download_bytes']>budget['caps']['download_bytes']-budget['download_bytes_used']-budget.get('download_bytes_reserved',0):raise ValueError('Download budget unavailable')
    if resources['workspace_growth_bytes']>min(shutil.disk_usage(project).free,budget['caps']['workspace_growth_bytes']-budget['workspace_bytes_measured']):raise ValueError('Storage headroom unavailable')
    for name,sha in task['input_sha256'].items():
        path=(project/name).resolve()
        if not path.is_relative_to(project) or not path.is_file() or file_hash(path)!=sha:raise ValueError('Input absent, outside workspace, or changed: '+name)
    script=(project/task['script']).resolve()
    if not script.is_relative_to(project/'scripts') or task['script'] not in task['input_sha256']:raise ValueError('Reviewed script digest required')
    protected=protected_files(project)
    if not task['outputs']:raise ValueError('Concrete new outputs required')
    for name in task['outputs']:
        path=(project/name).resolve()
        if not path.is_relative_to(project) or name in protected or path.exists():raise ValueError('Output is not new unprotected workspace artifact: '+name)
    return {'status':'ADMITTED','boundary':BOUNDARY,'task':task['name'],'host':checked['host'],'at':utc_now()}

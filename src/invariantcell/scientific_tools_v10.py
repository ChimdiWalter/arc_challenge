"""Bounded candidate/check/replay interface; no arbitrary code or outcome access."""
from dataclasses import dataclass,asdict
from .io import digest
from .core import time_value

FIELDS={'subject_id','context_id','specimen_id','construct_id','quantity','unit','assay','temperature_C','sodium_phosphate_mM','annotation_version'}
TOOLS={'compare_field','available_as_of','independent_endpoint'}
SCOPE='Supplied metadata consistency; not biological truth or complete source provenance'

@dataclass(frozen=True)
class ToolCall:
    tool:str
    record_ids:tuple[str,...]
    field:str|None=None
    cutoff:str|None=None
    def __post_init__(self):
        if self.tool not in TOOLS or not isinstance(self.record_ids,tuple) or not self.record_ids or len(set(self.record_ids))!=len(self.record_ids):raise ValueError('Unsupported typed tool/record IDs')
        if any(not isinstance(i,str) or not i for i in self.record_ids):raise ValueError('Record IDs required')
        if self.tool=='compare_field':
            if len(self.record_ids)!=2 or self.field not in FIELDS or self.cutoff is not None:raise ValueError('Explicit allowed comparison field required')
        elif self.tool=='available_as_of':
            if len(self.record_ids)!=1 or self.field is not None or not self.cutoff:raise ValueError('Explicit time cutoff required')
            time_value(self.cutoff)
        elif len(self.record_ids)!=1 or self.field is not None or self.cutoff is not None:raise ValueError('Endpoint tool takes one record')
    def to_dict(self):
        result=asdict(self);result['record_ids']=list(self.record_ids);return result

def _records(call,records):
    result=[]
    for key in call.record_ids:
        r=records[key]
        if r.get('record_id')!=key or r.get('kind') not in ('observed','inferred','predicted','synthetic'):raise ValueError('Typed evidence identity required')
        source=r.get('source_sha256')
        if not isinstance(source,str) or len(source)!=64 or any(c not in '0123456789abcdef' for c in source):raise ValueError('Source hash required')
        result.append(r)
    return result

def candidate(call,records):
    values=_records(call,records)
    if call.tool=='compare_field':
        left,right=(r.get(call.field) for r in values)
        outcome='UNKNOWN' if left is None or right is None else ('MATCH' if left==right else 'MISMATCH')
        detail={'field':call.field,'left':left,'right':right}
    elif call.tool=='available_as_of':
        available=values[0].get('public_available_at')
        outcome='UNKNOWN' if available is None else ('ADMISSIBLE' if time_value(available)<=time_value(call.cutoff) else 'FUTURE')
        detail={'available_at':available,'cutoff':call.cutoff}
    else:
        uses=values[0].get('used_for');known=values[0].get('lineage_complete')
        outcome='UNKNOWN'
        if isinstance(uses,list):
            if any(x in ('sampling','fitting','model_selection','hyperparameters') for x in uses):outcome='NOT_INDEPENDENT'
            elif known is True:outcome='INDEPENDENT_WITHIN_DECLARED_LINEAGE'
        detail={'used_for':uses,'lineage_complete':known}
    body={'call':call.to_dict(),'record_hashes':{k:digest(records[k]) for k in call.record_ids},'outcome':outcome,'detail':detail,
          'scope':SCOPE}
    return {**body,'certificate_hash':digest(body)}

def verify(certificate,expected_call,trusted_records,expected_hashes):
    """Independent branch logic; cannot trust a producer's self-supplied anchors."""
    _records(expected_call,trusted_records)
    keys={'call','record_hashes','outcome','detail','scope','certificate_hash'}
    if set(certificate)!=keys or certificate['call']!=expected_call.to_dict() or certificate['scope']!=SCOPE:raise ValueError('Certificate schema/query differs')
    wanted={k:digest(trusted_records[k]) for k in expected_call.record_ids}
    if expected_hashes!=wanted or certificate['record_hashes']!=wanted:raise ValueError('External evidence anchor differs')
    if certificate['certificate_hash']!=digest({k:v for k,v in certificate.items() if k!='certificate_hash'}):raise ValueError('Certificate digest differs')
    if expected_call.tool=='compare_field':
        a=trusted_records[expected_call.record_ids[0]].get(expected_call.field);b=trusted_records[expected_call.record_ids[1]].get(expected_call.field)
        expected='MATCH'
        if a is None or b is None:expected='UNKNOWN'
        elif a!=b:expected='MISMATCH'
        details={'field':expected_call.field,'left':a,'right':b}
    elif expected_call.tool=='available_as_of':
        a=trusted_records[expected_call.record_ids[0]].get('public_available_at');expected='UNKNOWN'
        if a is not None:expected='FUTURE' if time_value(a)>time_value(expected_call.cutoff) else 'ADMISSIBLE'
        details={'available_at':a,'cutoff':expected_call.cutoff}
    else:
        record=trusted_records[expected_call.record_ids[0]];a=record.get('used_for');complete=record.get('lineage_complete');expected='UNKNOWN'
        if isinstance(a,list):
            if complete is True:expected='INDEPENDENT_WITHIN_DECLARED_LINEAGE'
            for purpose in a:
                if purpose in ('sampling','fitting','model_selection','hyperparameters'):expected='NOT_INDEPENDENT'
        details={'used_for':a,'lineage_complete':complete}
    if certificate['outcome']!=expected or certificate['detail']!=details:raise ValueError('Witness conclusion not supported')
    return {'valid':True,'outcome':expected,'checked_against_external_anchors':True,'biological_truth_certified':False}

def run_workflow(calls,records,max_tools=12):
    if not 0<len(calls)<=max_tools or max_tools>32:raise ValueError('Finite tool budget exceeded')
    witnesses=[]
    for call in calls:
        cert=candidate(call,records);anchors={k:digest(records[k]) for k in call.record_ids}
        verify(cert,call,records,anchors);witnesses.append(cert)
    blocked=any(c['outcome'] in ('MISMATCH','FUTURE','NOT_INDEPENDENT') for c in witnesses)
    unknown=any(c['outcome']=='UNKNOWN' for c in witnesses)
    answer='NOT_COMPARABLE' if blocked else ('INSUFFICIENT_EVIDENCE' if unknown else 'COMPARABLE_WITHIN_CHECKED_FIELDS')
    return {'decision':answer,'tool_calls':len(calls),'witnesses':witnesses,'record_hashes':{k:digest(v) for k,v in records.items()},
        'guarantee':'Only explicit checks are replayed; omitted scientific conditions are not inferred satisfied.'}

def explain_decision_change(old,new,old_records,new_records,change_kind):
    if change_kind not in ('new_measurement','annotation_revision','identifier_correction','new_evidence','model_change','metadata_revision'):raise ValueError('Explicit change classification required')
    for result,records in ((old,old_records),(new,new_records)):
        if result.get('record_hashes')!={k:digest(v) for k,v in records.items()}:raise ValueError('Explanation evidence differs from checked workflow')
    changed=[];unchanged=[]
    for key in sorted(set(old_records)|set(new_records)):
        if digest(old_records.get(key))==digest(new_records.get(key)):unchanged.append(key)
        else:
            a=old_records.get(key,{}) or {};b=new_records.get(key,{}) or {}
            changed.append({'record_id':key,'fields':{f:{'old':a.get(f),'new':b.get(f)} for f in sorted(set(a)|set(b)) if a.get(f)!=b.get(f)}})
    return {'change_kind':change_kind,'changed_evidence':changed,'unchanged_records':unchanged,'old_decision':old['decision'],'new_decision':new['decision'],
        'decision_changed':old['decision']!=new['decision'],'biological_progression_inferred':False}

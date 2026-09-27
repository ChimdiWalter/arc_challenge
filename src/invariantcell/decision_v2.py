"""Continuous development calibration and evidence-aware selection.

P4 adds actual target-detection measurements from allowed controls to calibration,
then applies deterministic provenance checks. Any gain cannot be attributed to
formal verification alone. No policy accepts outcome labels.
"""
import math
import random
from statistics import mean,median
from .io import digest

BASE_FEATURES=('native_uncertainty','control_distance','effect_size')
EVIDENCE_FEATURES=('target_detection_fraction','target_control_log_mean')
FEATURES=BASE_FEATURES+EVIDENCE_FEATURES


def vector(row,features):
    if set(row)!=set(FEATURES):raise ValueError('Undeclared/missing features')
    result=[]
    for key in FEATURES:
        x=row[key]
        if type(x) not in (int,float) or not math.isfinite(x) or x<0:raise ValueError('Invalid feature')
        if key=='target_detection_fraction' and x>1:raise ValueError('Invalid detection fraction')
    for key in features:result.append(math.log1p(row[key]))
    return result


def fit_calibrator(rows,features=BASE_FEATURES,forbidden_contexts=('Jurkat',),alpha=1.0):
    import numpy as np
    if tuple(features) not in (BASE_FEATURES,FEATURES) or alpha!=1.0:raise ValueError('Unregistered calibration recipe')
    seen=set()
    for r in rows:
        if set(r)!={'candidate_id','context','features','error'} or r['context'] not in ('K562','RPE1','HepG2') or r['context'] in forbidden_contexts or r['candidate_id'] in seen:
            raise ValueError('Calibration boundary or schema violation')
        if not math.isfinite(r['error']) or r['error']<0:raise ValueError('Invalid calibration error')
        seen.add(r['candidate_id'])
    contexts=sorted({r['context'] for r in rows})
    if len(contexts)<2 or len(rows)<8:raise ValueError('Insufficient development support')
    x=np.array([vector(r['features'],features) for r in rows]);y=np.log(np.array([r['error'] for r in rows])+1e-8)
    center=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-8);z=(x-center)/scale
    intercept=float(y.mean());coef=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-intercept))
    body={'version':2,'features':list(features),'center':center.tolist(),'scale':scale.tolist(),'coef':coef.tolist(),
          'log_error_intercept':intercept,'alpha':alpha,'development_contexts':contexts,'row_hash':digest(rows),
          'target':'log(error+1e-8)','calibration_scope':'Development-only ridge error prediction; not conformal coverage or a probability.'}
    return {**body,'calibrator_hash':digest(body)}


def estimate_error(features,calibrator):
    body={k:v for k,v in calibrator.items() if k!='calibrator_hash'}
    if digest(body)!=calibrator.get('calibrator_hash'):raise ValueError('Calibrator tampering')
    if tuple(calibrator.get('features',[])) not in (BASE_FEATURES,FEATURES):raise ValueError('Invalid calibrated feature schema')
    width=len(calibrator['features'])
    for key in ('center','scale','coef'):
        values=calibrator.get(key)
        if not isinstance(values,list) or len(values)!=width or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
            raise ValueError('Calibrator parameter shape/value invalid')
    if any(v<=0 for v in calibrator['scale']) or not math.isfinite(calibrator.get('log_error_intercept',float('nan'))):raise ValueError('Invalid calibrator scale/intercept')
    if not set(calibrator.get('development_contexts',[]))<=set(('K562','RPE1','HepG2')) or len(calibrator.get('development_contexts',[]))<2:
        raise ValueError('Calibrator context boundary invalid')
    values=vector(features,calibrator['features'])
    score=calibrator['log_error_intercept']+sum((v-m)/s*b for v,m,s,b in zip(values,calibrator['center'],calibrator['scale'],calibrator['coef']))
    return max(0.0,math.exp(max(-40,min(40,score)))-1e-8)


def select(candidates,protocol,base_calibrator,evidence_calibrator,policy):
    if policy not in ('P0','P1','P2','P3','P4'):raise ValueError('Unknown policy')
    budget=protocol['budget']
    if type(budget)is not int or not 0<budget<=len(candidates):raise ValueError('Invalid fixed budget')
    seen=set();rows=[]
    for c in candidates:
        if set(c)!={'candidate_id','context','target','features','evidence','prediction_hash'}:raise ValueError('Selection schema rejects outcomes or undeclared fields')
        if c['candidate_id'] in seen:raise ValueError('Duplicate candidate')
        seen.add(c['candidate_id']);vector(c['features'],FEATURES)
        e=c['evidence']
        if set(e)!={'provenance_complete','context_compatible','formal_valid','conflict','temporal_status','snapshot_hash'}:raise ValueError('Invalid evidence schema')
        if any(type(e[k])is not bool for k in ('provenance_complete','context_compatible','formal_valid','conflict')) or e['temporal_status'] not in ('admissible','unknown','inadmissible'):
            raise ValueError('Invalid evidence states')
        if not isinstance(e['snapshot_hash'],str) or len(e['snapshot_hash'])!=64:raise ValueError('Missing evidence snapshot commitment')
        reasons=[]
        if policy in ('P3','P4'):
            for key in ('provenance_complete','context_compatible','formal_valid'):
                if not e[key]:reasons.append(key.upper()+'_FAILED')
            if e['conflict']:reasons.append('KNOWN_UNRESOLVED_CONFLICT')
            if e['temporal_status']=='inadmissible':reasons.append('TEMPORALLY_INADMISSIBLE')
        if policy=='P3' and c['features']['native_uncertainty']>protocol['fixed_native_uncertainty_limit']:
            reasons.append('FIXED_UNCERTAINTY_RULE')
        if policy in ('P1','P3'):score=c['features']['native_uncertainty']
        elif policy=='P2':score=estimate_error(c['features'],base_calibrator)
        elif policy=='P4':score=estimate_error(c['features'],evidence_calibrator)
        else:score=0.0
        rows.append({'candidate_id':c['candidate_id'],'score':score,'excluded_reasons':reasons})
    if policy=='P0':
        ordered=sorted(rows,key=lambda r:r['candidate_id']);random.Random(protocol['seed']).shuffle(ordered)
        rank={r['candidate_id']:i for i,r in enumerate(ordered)}
        for r in rows:r['score']=rank[r['candidate_id']]
    ranked=sorted((r for r in rows if not r['excluded_reasons']),key=lambda r:(r['score'],r['candidate_id']))
    body={'policy':policy,'protocol_hash':digest(protocol),'candidate_hash':digest(candidates),'budget':budget,
          'selected_ids':[r['candidate_id'] for r in ranked[:budget]],'rows':rows,
          'base_calibrator_hash':base_calibrator['calibrator_hash'],'evidence_calibrator_hash':evidence_calibrator['calibrator_hash']}
    return {**body,'decision_hash':digest(body)}


def evaluate(decisions,candidates,protocol,outcomes,base_calibrator,evidence_calibrator):
    # Replay before the evaluation-only outcome join.
    if set(decisions)!={'P0','P1','P2','P3','P4'}:raise ValueError('All five frozen policies required')
    for policy,d in decisions.items():
        if d!=select(candidates,protocol,base_calibrator,evidence_calibrator,policy):raise ValueError('Decision replay failed')
    if set(outcomes)!={c['candidate_id'] for c in candidates} or any(not math.isfinite(x) or x<0 for x in outcomes.values()):raise ValueError('Outcome universe mismatch')
    budget=protocol['budget'];oracle=sorted(outcomes.values());result={}
    for policy,d in decisions.items():
        selected=d['selected_ids'];errors=[outcomes[c] for c in selected];n=len(errors)
        good=sum(x<=protocol['reliability_max_error'] for x in errors)
        curve=[]
        for k in range(1,n+1):curve.append({'selected_count':k,'coverage':k/len(candidates),'selected_mean_error':mean(errors[:k]),'oracle_regret_at_matched_count':mean(errors[:k])-mean(oracle[:k])})
        result[policy]={'selected_count':n,'unused_budget':budget-n,'coverage':n/len(candidates),
            'primary_selected_mean_error_at_fixed_budget':mean(errors) if n==budget else None,
            'selected_mean_error':mean(errors) if n else None,'selected_median_error':median(errors) if n else None,
            'regret_at_matched_count':mean(errors)-mean(oracle[:n]) if n else None,
            'reliable_yield_at_budget':good/budget,'precision':good/n if n else None,'quality_coverage_curve':curve,
            'primary_comparable':n==budget}
    return {'policies':result,'outcomes_hash':digest(outcomes),'biological_benefit':None,
            'scope':'Prediction reliability; no biological-benefit endpoint. Reduced coverage is not counted as a fixed-budget gain.',
            'p4_attribution':'Evidence-feature calibration plus checks; formal verification alone requires an additional ablation.'}

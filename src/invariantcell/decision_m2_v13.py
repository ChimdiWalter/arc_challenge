"""Selection of fixed documentary predictions; gold enters fitting/evaluation only."""
from __future__ import annotations
from collections import Counter
import hashlib,json,math

POLICIES=('P0_random','P1_native_confidence_proxy','P2_development_calibrated','P3_temporal_rules','P4_verified_evidence')
SEED=17
BIN_EDGES=(0.,1/3,2/3,1.)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def query_key(query):
    return digest({'id':query['id'],'query':query['query']})


def check_input(corpus,expected_split):
    if expected_split not in ('development','validation'):raise ValueError('M2 final-internal-test access forbidden')
    queries=corpus['queries'];records=corpus['records']
    if not queries or len({q['id']for q in queries})!=len(queries):raise ValueError('Empty or duplicate query panel')
    if any(q.get('split')!=expected_split for q in queries):raise ValueError('Mixed or forbidden partition')
    if any('gold'in q or'outcome'in q or'label'in q for q in queries):raise ValueError('Label field in query input')
    if len({r['evidence_id']for r in records})!=len(records):raise ValueError('Duplicate evidence identity')
    return queries,records


def representative_panel(corpus,split):
    """One hash-selected query per revision case, before any predictions/labels."""
    queries,_=check_input(corpus,split);cases={c['id']:c for c in corpus['cases']}
    if len(cases)!=len(corpus['cases']):raise ValueError('Duplicate revision case')
    groups={}
    for q in queries:
        if q['case_id']not in cases:raise ValueError('Unknown query parent case')
        groups.setdefault(q['case_id'],[]).append(q)
    if set(groups)!=set(cases):raise ValueError('Revision case without query')
    return {**corpus,'queries':[min(groups[c],key=query_key)for c in sorted(groups)]}


def native_confidence(records,prediction,query):
    """Unfitted consensus proxy; solve() does not supply model confidence."""
    by_id={r['evidence_id']:r for r in records};used=prediction['used_ids']
    if len(used)!=len(set(used))or any(i not in by_id for i in used):raise ValueError('Invalid native provenance')
    relevant=[]
    for ident in used:
        r=by_id[ident]
        if (r['subject'],r['predicate'],r['context'])!=(query['query']['subject'],query['query']['predicate'],query['query']['context']):raise ValueError('Native used inadmissible unrelated evidence')
        relevant.append(r)
    if not relevant:return 0.
    if query['query']['predicate']=='accepted_replacement_value':
        states=[str(r['value'])if r['polarity']=='supports'else'REFUTES'for r in relevant]
    else:states=[r['polarity']for r in relevant]
    return max(Counter(states).values())/len(states)


def bin_index(value):
    if not isinstance(value,(int,float))or not math.isfinite(value)or not 0<=value<=1:raise ValueError('Invalid native confidence')
    return min(2,int(value*3))


def fit_calibrator(development_panel,gold):
    if any(x['split']!='development'for x in development_panel):raise ValueError('Calibration is development only')
    if set(gold)!={x['id']for x in development_panel}:raise ValueError('Development gold/panel mismatch')
    n=[0]*3;correct=[0]*3
    for row in development_panel:
        b=bin_index(row['native_confidence']);n[b]+=1;correct[b]+=row['native']['outcome']==gold[row['id']]['outcome']
    return {'method':'three_fixed_equal_width_bins_Laplace1','bin_edges':list(BIN_EDGES),'n':n,'correct':correct,
        'probability_correct':[(k+1)/(count+2)for k,count in zip(correct,n)],'development_panel_sha256':digest(development_panel),
        'development_gold_sha256':digest(gold),'empty_bin_policy':'prior0.5; never fit validation or final-internal-test',
        'selection_direction':'higher estimated correctness; lower calibrated uncertainty'}


def build_panel(corpus,split,solve):
    queries,records=check_input(corpus,split);rows=[]
    # Same retrospective corpus and one checker-call allowance per query for each policy.
    for query in sorted(queries,key=query_key):
        native=solve(records,query,'static_kg')
        rules=solve(records,query,'simple_temporal_rules')if split=='validation'else None
        verified=solve(records,query,'verified_temporal')if split=='validation'else None
        if verified is not None and verified['certificate_replay_valid']is not True:raise ValueError('Verified checker replay failed')
        case=next(c for c in corpus['cases']if c['id']==query['case_id'])
        rows.append({'id':query['id'],'split':split,'query_sha256':query_key(query),
            'case_id':query['case_id'],'family':case['family'],'evidence_item_id':case['evidence_item_id'],
            'native':native,'native_confidence':native_confidence(records,native,query),
            'rules':rules,'verified':verified,'records_available':len(records),
            'matched_records':sum((r['subject'],r['predicate'],r['context'])==(query['query']['subject'],query['query']['predicate'],query['query']['context'])for r in records)})
    return rows


def rank_and_select(panel,calibrator):
    if not panel or any(r['split']!='validation'for r in panel):raise ValueError('Selection requires validation prediction panel')
    if len({r['id']for r in panel})!=len(panel):raise ValueError('Duplicate fixed prediction')
    n=len(panel);budget=max(1,(n+4)//5);result={}
    for policy in POLICIES:
        def score(row):
            if policy=='P0_random':return int(hashlib.sha256(f'{SEED}:{row["query_sha256"]}'.encode()).hexdigest(),16)
            if policy=='P1_native_confidence_proxy':return row['native_confidence']
            if policy=='P2_development_calibrated':return calibrator['probability_correct'][bin_index(row['native_confidence'])]
            checker=row['rules']if policy=='P3_temporal_rules'else row['verified']
            return int(checker['outcome']==row['native']['outcome'])
        ordered=sorted(panel,key=lambda row:(-score(row),row['query_sha256']))
        result[policy]={'selected_ids':[r['id']for r in ordered[:budget]],'ranked_ids':[r['id']for r in ordered],
            'scores':{r['id']:str(score(r))if policy=='P0_random'else score(r)for r in ordered},
            'budget':budget,'budget_fraction_requested':.2,'budget_rule':'ceil(0.2*n), minimum1',
            'native_prediction_sha256':digest({r['id']:r['native']for r in panel}),
            'tool_allowance':{'source_corpus_access':'identical current retrospective corpus','shared_native_calls':n,'maximum_extra_checker_calls':n,'maximum_replay_calls':n},
            'actual_tools':{'shared_native_calls':n,'extra_checker_calls':n if policy in ('P3_temporal_rules','P4_verified_evidence')else 0,
                'certificate_replays':n if policy=='P4_verified_evidence'else 0,'online_retrieval_calls':0}}
    return result


def evaluate(panel,selections,gold):
    """Evaluator-only: caller must persist native/selection commitments first."""
    if set(gold)!={r['id']for r in panel}:raise ValueError('Evaluation gold/panel mismatch')
    if set(selections)!=set(POLICIES):raise ValueError('Five matched policies required')
    rows={r['id']:r for r in panel};n=len(rows);correct={i:rows[i]['native']['outcome']==gold[i]['outcome']for i in rows}
    native_hash=digest({r['id']:r['native']for r in panel});summary={}
    for policy,s in selections.items():
        ids=s['selected_ids'];k=s['budget']
        if k!=max(1,(n+4)//5)or len(ids)!=k or len(set(ids))!=k or not set(ids)<=set(rows):raise ValueError('Fixed selection budget/identity violated')
        if s['native_prediction_sha256']!=native_hash:raise ValueError('Policy changed fixed native predictions')
        hits=sum(correct[i]for i in ids);oracle=min(k,sum(correct.values()));abstentions=sum(rows[i]['native']['outcome']=='ABSTAIN'for i in ids)
        summary[policy]={'selected_queries':k,'correct_selected':hits,'selected_accuracy':hits/k,'selected_error_rate':1-hits/k,
            'yield_correct_at_budget':hits,'selection_coverage':k/n,'regret_correct_count_vs_evaluation_only_oracle':oracle-hits,
            'regret_accuracy_vs_evaluation_only_oracle':(oracle-hits)/k,'native_abstentions_selected':abstentions,
            'native_abstention_fraction_selected':abstentions/k,'correct_native_abstentions_selected':sum(correct[i]and rows[i]['native']['outcome']=='ABSTAIN'for i in ids),
            'tool_cost':s['actual_tools'],'selected_ids':ids,
            'selected_revision_cases':len({rows[i]['case_id']for i in ids}),
            'selected_families':sorted({str(rows[i]['family'])for i in ids}),
            'selected_evidence_items':len({rows[i]['evidence_item_id']for i in ids})}
    p3=summary['P3_temporal_rules'];p4=summary['P4_verified_evidence']
    return {'status':'DOCUMENTARY_M2_EVALUATED','queries':n,'full_native_panel_correct':sum(correct.values()),'full_native_panel_accuracy':sum(correct.values())/n,
        'full_native_panel_abstentions':sum(r['native']['outcome']=='ABSTAIN'for r in panel),'policies':summary,
        'review_units':'One predetermined query per revision case; cases may share evidence item/family and are not IID.',
        'revision_cases':len({r['case_id']for r in panel}),'families':dict(Counter(str(r['family'])for r in panel)),
        'evidence_items':len({r['evidence_item_id']for r in panel}),
        'P4_minus_P3_selected_accuracy':p4['selected_accuracy']-p3['selected_accuracy'],'P4_P3_same_selection':p4['selected_ids']==p3['selected_ids'],
        'biological_M2_complete':False,'wet_lab_experiments_saved':None,'historically_available_knowledge_established':False,
        'scope':'Current-source retrospective editorial revision-query correctness; no biomedical outcome or biological decision utility.'}

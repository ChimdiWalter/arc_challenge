"""Audit saved temporal certificates against external frozen inputs.

This module does not call the answer-generating temporal evaluator or issuer.
Synthetic witnesses belong in tests. Real-panel contradictions may be absent.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from invariantcell.core import canonical_hash, time_value
from invariantcell.temporal import EvidenceRecord, Snapshot, TemporalQuery
from invariantcell.replay import verify_certificate

TAMPER_KINDS = ('result_rehashed', 'query_rehashed', 'source_rehashed', 'digest_corrupted')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rehash(certificate):
    certificate['snapshot_sha256'] = canonical_hash(certificate['snapshot'])
    certificate['dependencies_sha256'] = canonical_hash(certificate['snapshot']['dependencies'])
    certificate['query_sha256'] = canonical_hash(certificate['query'])
    certificate['certificate_sha256'] = canonical_hash({k: v for k, v in certificate.items() if k != 'certificate_sha256'})
    return certificate


def tamper(certificate, kind):
    changed = copy.deepcopy(certificate)
    if kind == 'result_rehashed':
        changed['result']['outcome'] = 'forged_outcome'
    elif kind == 'query_rehashed':
        changed['query']['valid_at'] = '1900-01-01T00:00:00Z'
    elif kind == 'source_rehashed':
        changed['snapshot']['evidence'][0]['source_sha256'] = '0' * 64
    elif kind == 'digest_corrupted':
        changed['certificate_sha256'] = '0' * 64
        return changed
    else:
        raise ValueError('unknown tamper task')
    return _rehash(changed)


def saved_answer(certificate, predicate):
    """Decode the saved checked result; do not regenerate a temporal answer."""
    answers = certificate['result']['answers']
    if predicate == 'accepted_replacement_value':
        values = sorted({a['value'] for a in answers if a['support_ids']})
        return 'ABSTAIN' if not values else 'VALUE:' + values[0] if len(values) == 1 else 'MULTIPLE_VALUES'
    support = any(a['support_ids'] for a in answers)
    refute = any(a['refute_ids'] for a in answers)
    return 'CONFLICT' if support and refute else 'ACCEPTED' if support else 'NOT_ACCEPTED' if refute else 'ABSTAIN'


def typed_reason(reason, record):
    if reason == 'public_evidence_after_cutoff_or_unknown':
        return 'HISTORICAL_PUBLIC_AVAILABILITY_NOT_ESTABLISHED'
    if reason == 'before_valid_interval':
        return 'BEFORE_DOCUMENTED_EDITORIAL_INTERVAL'
    if reason == 'after_valid_interval':
        return 'AFTER_DOCUMENTED_EDITORIAL_INTERVAL'
    if reason == 'event_after_cutoff_or_unknown':
        return 'CURATION_EVENT_TIME_UNKNOWN' if record.event_time is None else 'CURATION_EVENT_AFTER_REQUESTED_CUTOFF'
    return {'unknown_validity': 'VALIDITY_UNKNOWN', 'ingestion_after_cutoff_or_unknown': 'INGESTION_CUTOFF_UNMET',
            'prediction_not_observation': 'PREDICTION_NOT_ALLOWED', 'dependency_after_cutoff_or_unknown': 'DEPENDENCY_AVAILABILITY_UNESTABLISHED'}.get(reason, reason.upper())


def contradiction_witnesses(certificate):
    return [{'value': a['value'], 'support_ids': a['support_ids'], 'refute_ids': a['refute_ids']}
            for a in certificate['result']['answers'] if a['support_ids'] and a['refute_ids']]


def audit_query(records, query_record, prediction, gold, anchors, anchor_valid):
    """Consumed-panel audit. Query/snapshot trust anchors are external to proof."""
    query = TemporalQuery(**query_record['query'])
    snapshot = Snapshot('v13:' + query_record['id'], tuple(EvidenceRecord(**r) for r in records))
    certificate = prediction['certificate']
    digest = canonical_hash(snapshot.to_dict())
    first = verify_certificate(certificate, snapshot, digest, query)
    second = verify_certificate(certificate, snapshot, digest, query)
    declared = sorted(i for a in certificate['result']['answers'] for i in (*a['support_ids'], *a['refute_ids']))
    binding = declared == sorted(prediction['used_ids']) and saved_answer(certificate, query.predicate) == prediction['outcome']
    excluded = {x['evidence_id']: x['reasons'] for x in certificate['result']['excluded']}
    relevant = [r for r in snapshot.evidence if (r.subject, r.predicate, r.context) == (query.subject, query.predicate, query.context)]
    used = [r for r in relevant if r.evidence_id in prediction['used_ids']]
    paths = []
    for record in relevant:
        reasons = excluded.get(record.evidence_id, [])
        a = anchors.get(record.evidence_id, {})
        paths.append({'evidence_id': record.evidence_id, 'polarity': record.polarity,
              'source_sha256': record.source_sha256, 'source_path': a.get('source_path'),
              'source_json_pointer': a.get('json_pointer'), 'source_integrity_verified': anchor_valid.get(record.evidence_id, False),
              'observed_public_by': record.public_available_at, 'first_public_available_at': a.get('first_public_available_at'),
              'ingested_at': record.ingested_at, 'valid_from': record.valid_from, 'valid_to': record.valid_to,
              'curation_event_time': record.event_time, 'biological_event_time': a.get('biological_event_time'),
              'disposition': 'EXCLUDED' if reasons else 'INCLUDED', 'failed_rules': reasons,
              'rule_path': ['match proposition/context', 'source observed-by cutoff', 'ingestion cutoff',
                            'explicit editorial-validity interval', 'optional curation-event cutoff'],
              'typed_reasons': [typed_reason(reason, record) for reason in reasons]})
    reasons = sorted({reason for p in paths for reason in p['typed_reasons']}) if prediction['outcome'] == 'ABSTAIN' else []
    if prediction['outcome'] == 'ABSTAIN' and not relevant:
        reasons = ['NO_MATCHING_ASSERTION']
    tampered = {}
    for kind in TAMPER_KINDS:
        result = verify_certificate(tamper(certificate, kind), snapshot, digest, query)
        tampered[kind] = {'rejected': not result.valid, 'errors': list(result.errors), 'internal_hashes_recomputed': kind != 'digest_corrupted'}
    violations = {'observed_availability': sum(r.public_available_at is None or time_value(r.public_available_at) > time_value(query.evidence_cutoff) for r in used),
                  'interval': sum(r.validity_kind != 'interval' or (r.valid_from is not None and time_value(r.valid_from) > time_value(query.valid_at)) or (r.valid_to is not None and time_value(r.valid_to) <= time_value(query.valid_at)) for r in used),
                  'curation_event': sum(query.event_cutoff is not None and (r.event_time is None or time_value(r.event_time) > time_value(query.event_cutoff)) for r in used)}
    return {'query_id': query_record['id'], 'case_id': query_record['case_id'], 'query': query.to_dict(),
         'answer': prediction['outcome'], 'answer_correct_against_consumed_source_gold': prediction['outcome'] == gold['outcome'],
         'correctness_scope': 'recheck archived coding-agent labels; not new clinical truth or fresh test',
         'known_as_of': query.evidence_cutoff, 'valid_at': query.valid_at, 'evidence_ids': prediction['used_ids'],
         'proof_path': paths, 'typed_abstention_reasons': reasons,
         'certificate_hash': certificate['certificate_sha256'], 'external_snapshot_hash': digest,
         'replay_result': first.to_dict(), 'repeat_replay_result': second.to_dict(),
         'replay_reproducible': first.to_dict() == second.to_dict(), 'saved_answer_bound_to_certificate': binding,
         'provenance_complete': bool(paths) and all(p['source_integrity_verified'] for p in paths),
         'measured_constraint_violations': violations,
         'future_publication_leakage': {'status': 'NOT_IDENTIFIABLE_FIRST_PUBLIC_TIME_UNKNOWN', 'count': None},
         'contradiction_witnesses': contradiction_witnesses(certificate), 'tamper_results': tampered,
         'ontology_entailment_trace': {'status': 'NOT_PRESENT_IN_THIS_TEMPORAL_CERTIFICATE', 'trace': None}}


def summary(rows):
    n = len(rows)
    abstain = [r for r in rows if r['answer'] == 'ABSTAIN']
    return {'queries': n, 'answer_correct': sum(r['answer_correct_against_consumed_source_gold'] for r in rows),
       'proof_valid': sum(r['replay_result']['valid'] and r['saved_answer_bound_to_certificate'] for r in rows),
       'provenance_complete': sum(r['provenance_complete'] for r in rows),
       'replay_reproducible': sum(r['replay_reproducible'] for r in rows),
       'abstentions': len(abstain), 'correct_abstentions': sum(r['answer_correct_against_consumed_source_gold'] for r in abstain),
       'typed_abstentions': sum(bool(r['typed_abstention_reasons']) for r in abstain),
       'measured_constraint_violations': {k: sum(r['measured_constraint_violations'][k] for r in rows) for k in ('observed_availability', 'interval', 'curation_event')},
       'tamper_tasks': n * len(TAMPER_KINDS), 'tamper_rejections': sum(t['rejected'] for r in rows for t in r['tamper_results'].values()),
       'real_contradiction_witnesses': sum(len(r['contradiction_witnesses']) for r in rows),
       'real_contradiction_detection_accuracy': None,
       'contradiction_metric_status': 'NOT_ESTIMABLE_NO_INDEPENDENT_REAL_CONTRADICTION_LABELS',
       'future_publication_leakage': None, 'future_publication_leakage_status': 'NOT_IDENTIFIABLE_FIRST_PUBLIC_TIME_UNKNOWN',
       'ontology_entailment_trace_status': 'NOT_PRESENT_IN_THIS_TEMPORAL_CERTIFICATE',
       'incremental_answer_accuracy_claim': False, 'source_gold_is_clinical_expert_gold': False}

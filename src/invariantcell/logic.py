"""Finite, positive Horn logic over triples, with independently replayable steps.

No negation-as-failure, existential rules, OWL completeness, or biological truth
claim. Temporal admissibility must be established before facts enter this layer.
"""
from itertools import product
from .io import digest


def binding(patterns, facts):
    env = {}
    for pattern, fact in zip(patterns, facts):
        if len(pattern) != 3 or len(fact) != 3:
            raise ValueError('Facts/rule atoms must be triples')
        for symbol, value in zip(pattern, fact):
            if symbol.startswith('?'):
                if symbol in env and env[symbol] != value:
                    return None
                env[symbol] = value
            elif symbol != value:
                return None
    return env


def validate(ontology, facts):
    if ontology['logic'] != 'finite-positive-horn-triples-v1':
        raise ValueError('Unsupported logic profile')
    for fact in facts:
        if len(fact) != 3 or any(not isinstance(v, str) or not v or v.startswith('?') for v in fact):
            raise ValueError('Expected grounded, nonempty string triples')
    ids = set()
    for rule in ontology['rules']:
        if rule['id'] in ids or not 1 <= len(rule['body']) <= 2:
            raise ValueError('Duplicate rule ID or unsupported arity')
        ids.add(rule['id'])
        variables = {v for p in rule['body'] for v in p if v.startswith('?')}
        if any(v.startswith('?') and v not in variables for v in rule['head']):
            raise ValueError('Rule is not range restricted')


def contradictions(ontology, facts):
    facts = set(map(tuple, facts))
    conflicts = []
    for left, right in ontology.get('disjoint_types', []):
        for subject, predicate, value in sorted(facts):
            if predicate == 'rdf:type' and value == left and (subject, predicate, right) in facts:
                conflicts.append({'subject': subject, 'kind': 'disjoint_types', 'types': [left, right]})
    for subject, predicate, value in sorted(facts):
        if predicate in ontology.get('irreflexive', []) and subject == value:
            conflicts.append({'subject': subject, 'kind': 'irreflexive_relation', 'predicate': predicate})
    return conflicts


def reason(ontology, facts, max_facts=1000):
    validate(ontology, facts)
    known = set(map(tuple, facts))
    steps = []
    while True:
        old = len(known)
        for rule in ontology['rules']:
            pools = [[f for f in sorted(known) if p[1].startswith('?') or f[1] == p[1]] for p in rule['body']]
            for premises in product(*pools):
                env = binding(rule['body'], premises)
                if env is None:
                    continue
                conclusion = tuple(env.get(v, v) for v in rule['head'])
                if conclusion in known:
                    continue
                known.add(conclusion)
                if len(known) > max_facts:
                    raise ValueError('Finite closure budget exceeded')
                steps.append({'rule': rule['id'], 'premises': [list(f) for f in premises], 'conclusion': list(conclusion)})
        if len(known) == old:
            break
    body = {'ontology_hash': digest(ontology), 'input_hash': digest(sorted(map(list, facts))),
        'steps': steps, 'facts': sorted(map(list, known)), 'conflicts': contradictions(ontology, known),
        'claim': 'finite Horn consequences of supplied admissible premises; not biological truth'}
    return {**body, 'certificate_hash': digest(body)}


def verify(certificate, ontology, trusted_facts):
    """Check each derivation locally; never invoke the forward-chain engine."""
    validate(ontology, trusted_facts)
    body = {k: v for k, v in certificate.items() if k != 'certificate_hash'}
    if digest(body) != certificate['certificate_hash'] or certificate['ontology_hash'] != digest(ontology) or certificate['input_hash'] != digest(sorted(map(list, trusted_facts))):
        raise ValueError('Certificate/trusted input mismatch')
    known, rules = set(map(tuple, trusted_facts)), {r['id']: r for r in ontology['rules']}
    for step in certificate['steps']:
        if step['rule'] not in rules:
            raise ValueError('Untrusted rule')
        rule, premises = rules[step['rule']], step['premises']
        if len(premises) != len(rule['body']) or any(tuple(p) not in known for p in premises):
            raise ValueError('Missing or invented premise')
        # Independent unification implementation; no trust in producer binding.
        values = {}
        for pattern, premise in zip(rule['body'], premises):
            for symbol, value in zip(pattern, premise):
                if symbol.startswith('?'):
                    if symbol in values and values[symbol] != value:
                        raise ValueError('Inconsistent rule variable')
                    values[symbol] = value
                elif symbol != value:
                    raise ValueError('Rule constant mismatch')
        expected = [values[v] if v.startswith('?') else v for v in rule['head']]
        if expected != step['conclusion']:
            raise ValueError('Unsupported conclusion')
        known.add(tuple(expected))
    if sorted(map(list, known)) != certificate['facts'] or contradictions(ontology, known) != certificate['conflicts']:
        raise ValueError('Declared closure/conflicts differ from replay')
    return {'valid': True, 'consistent': not certificate['conflicts'], 'steps_replayed': len(certificate['steps']),
            'completeness': 'not certified; this checker proves only listed derivations'}

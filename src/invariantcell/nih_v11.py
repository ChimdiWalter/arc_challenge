"""Local v11 interoperability: separate structural and bounded logical checks.

No network, prediction, gold scoring or full-OWL claims. Optional RDF packages
must already exist in the project-local ontology dependency directory.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from invariantcell.ontology_validation import check_owlrl, dependency_status

IC_URI = 'urn:invariantcell:nih:v11#'
BL_URI = 'https://w3id.org/biolink/vocab/'
TIME_AXES = {
    'event_time': 'eventTime', 'valid_from': 'validFrom', 'valid_until': 'validUntil',
    'evidence_available_at': 'evidenceAvailableAt', 'source_publication_time': 'sourcePublicationTime',
    'annotation_version_time': 'annotationVersionTime', 'ingested_at': 'ingestedAt',
    'model_generated_at': 'modelGeneratedAt',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_artifacts(project):
    from rdflib import Graph
    from rdflib.namespace import OWL
    project = Path(project)
    ontology = Graph().parse(data=(project / 'ontology/nih_v11.owl').read_text(), format='xml')
    shapes = Graph().parse(data=(project / 'ontology/nih_v11_shapes.ttl').read_text(), format='turtle')
    if list(ontology.triples((None, OWL.imports, None))):
        raise ValueError('Remote imports are outside this bounded local validation profile')
    return ontology, shapes


def validate_mapping(project, mapping=None):
    """Check identifiers against pinned primary schema, rather than guessed CURIEs."""
    import yaml
    project = Path(project)
    mapping = mapping or yaml.safe_load((project / 'ontology/biolink_mapping_v11.yaml').read_text())
    path = project / mapping['schema_path']
    path.resolve().relative_to(project.resolve())
    if sha256(path) != mapping['schema_sha256']:
        raise ValueError('Pinned Biolink schema hash mismatch')
    schema = yaml.safe_load(path.read_text())
    if str(schema['version']) != mapping['schema_version']:
        raise ValueError('Biolink version mismatch')
    prefixes = {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'biolink': BL_URI}
    for section in ('classes', 'slots'):
        for item in mapping[section].values():
            key = item['schema_key']
            if key not in schema[section]:
                raise ValueError(f'Unsupported Biolink identifier: {key}')
            definition = schema[section][key]
            explicit = definition.get('class_uri' if section == 'classes' else 'slot_uri')
            if explicit:
                prefix, local = explicit.split(':', 1)
                expected = prefixes[prefix] + local
            else:
                local = ''.join(word.capitalize() for word in key.split()) if section == 'classes' else key.replace(' ', '_')
                expected = BL_URI + local
            if item['uri'] != expected:
                raise ValueError(f'Incorrect identifier URI for {key}')
    for item in mapping['external_exact_mappings']:
        if item['target'] not in schema[item['kind']][item['schema_key']].get('exact_mappings', []):
            raise ValueError('External mapping is not exact in the pinned schema')
    return {'status': 'IDENTIFIERS_VERIFIED', 'schema_version': mapping['schema_version'],
            'schema_sha256': mapping['schema_sha256'], 'class_entries': len(mapping['classes']),
            'slot_entries': len(mapping['slots']), 'external_entries': len(mapping['external_exact_mappings']),
            'full_biolink_instance_validation': 'NOT_EVALUATED'}


def _add_times(graph, node, values):
    from rdflib import Literal, Namespace
    from rdflib.namespace import XSD
    ic = Namespace(IC_URI)
    for axis, prop in TIME_AXES.items():
        if values.get(axis) is None:
            graph.add((node, ic.unknownTimeAxis, Literal(axis, datatype=XSD.string)))
        else:
            graph.add((node, ic[prop], Literal(str(values[axis]), datatype=XSD.dateTime)))


def documentary_graph(corpus):
    """Reify source assertions and retain legacy documentary validity verbatim.

    Copied intervals are declared conservative v10 documentary projections,
    never new biological timestamps or adjudicated precise historical events.
    """
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, RDFS, XSD
    ic, bl, graph = Namespace(IC_URI), Namespace(BL_URI), Graph()
    def uri(kind, value):
        return URIRef(IC_URI + kind + '/' + quote(value, safe=''))
    sources = {s['id']: s for s in corpus['sources']}
    for row in corpus['records']:
        e = row['evidence']
        node, source = uri('assertion', e['evidence_id']), uri('source', row['source_id'])
        if sources[row['source_id']]['snapshot_sha256'] != e['source_sha256']:
            raise ValueError('Documentary assertion/source hash mismatch')
        graph.add((source, RDF.type, ic.EvidenceSource))
        graph.add((source, ic.sourceHash, Literal(e['source_sha256'], datatype=XSD.string)))
        graph.add((source, RDFS.seeAlso, URIRef(sources[row['source_id']]['url'])))
        graph.add((node, RDF.type, ic.AttributedAssertion))
        graph.add((node, RDF.type, ic.DocumentaryAssertion))
        graph.add((node, RDF.subject, uri('documented-entity', e['subject'])))
        graph.add((node, RDF.predicate, bl.related_to))
        graph.add((node, RDF.object, uri('documentary-value', e['value'])))
        graph.add((node, bl.negated, Literal(e['polarity'] == 'refutes')))
        graph.add((node, ic.hasSource, source))
        for prop, value in [('context', e['context']), ('originalPredicate', e['predicate']),
                            ('mappingStatus', 'broad_only'), ('evidenceState', 'attributed'),
                            ('validityKind', e['validity_kind'])]:
            graph.add((node, ic[prop], Literal(value, datatype=XSD.string)))
        graph.add((node, RDFS.comment, Literal('Retained v10 conservative documentary validity projection; not biological time.')))
        _add_times(graph, node, {'event_time': e.get('event_time'), 'valid_from': e.get('valid_from'),
            'valid_until': e.get('valid_to'), 'evidence_available_at': e['public_available_at'],
            'source_publication_time': row.get('first_public_available_at'), 'ingested_at': e['ingested_at']})
        if row.get('corrects'):
            graph.add((node, ic.corrects, uri('assertion', row['corrects'])))
    return graph


def adapt_go_graph(graph):
    """Keep original triples, add local semantics and explicitly unknown times."""
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, XSD
    result, ic, bl = Graph(), Namespace(IC_URI), Namespace(BL_URI)
    old = Namespace('urn:invariantcell:ontology-mapping:')
    for triple in graph:
        result.add(triple)
    for node in graph.subjects(RDF.type, bl.Association):
        result.add((node, RDF.type, ic.AttributedAssertion))
        result.add((node, RDF.type, ic.GOAnnotation))
        for prop, value in [('context', 'GAF:unscoped-no-specimen'), ('evidenceState', 'attributed'), ('validityKind', 'unknown')]:
            result.add((node, ic[prop], Literal(value, datatype=XSD.string)))
        for value in graph.objects(node, old.gaf_qualifier):
            result.add((node, ic.gafQualifier, value))
        result.add((node, ic.annotationDate, graph.value(node, old.annotation_date)))
        for field in ('source_sha256', 'ontology_sha256'):
            digest = graph.value(node, old[field])
            source = URIRef(IC_URI + 'source/sha256/' + str(digest))
            result.add((source, RDF.type, ic.EvidenceSource))
            result.add((source, ic.sourceHash, digest))
            result.add((node, ic.hasSource, source))
        if graph.value(node, RDF.predicate) is None:
            result.add((node, ic.mappingStatus, Literal('unmapped', datatype=XSD.string)))
        _add_times(result, node, {'evidence_available_at': graph.value(node, old.public_available_at),
                                'ingested_at': graph.value(node, old.ingested_at)})
    return result


def validate_graph(graph, ontology, shapes):
    from pyshacl import validate
    from rdflib import Graph, Namespace
    from rdflib.namespace import RDF, SH
    combined = Graph()
    for source in (ontology, graph):
        for triple in source:
            combined.add(triple)
    conforms, report, _ = validate(graph, shacl_graph=shapes, ont_graph=ontology,
        inference='none', advanced=False, js=False, do_owl_imports=False,
        meta_shacl=True, inplace=False, abort_on_first=False)
    violations = [{'focus': str(report.value(n, SH.focusNode)), 'path': str(report.value(n, SH.resultPath)),
                   'component': str(report.value(n, SH.sourceConstraintComponent))}
                  for n in report.subjects(RDF.type, SH.ValidationResult)]
    logic, expanded = check_owlrl(combined, max_seconds=10)
    ic = Namespace(IC_URI)
    return {'rdf_parse': {'valid': True, 'data_triples': len(graph), 'ontology_triples': len(ontology)},
            'shacl': {'conforms': bool(conforms), 'assertion_targets': len(set(graph.subjects(RDF.type, ic.AttributedAssertion))),
                      'violations': sorted(violations, key=lambda v: (v['focus'], v['path'], v['component'])),
                      'inference': 'none', 'meta_shacl': True, 'imports_loaded': False}, 'owlrl': logic}, expanded


def ontology_metrics(graph, ontology, logic):
    """Provisional graph-scoped definitions; these are not official NIH scores."""
    from rdflib import URIRef
    from rdflib.namespace import RDF, RDFS, OWL
    classes = {c for c in ontology.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)}
    objects = set(ontology.subjects(RDF.type, OWL.ObjectProperty))
    data_props = set(ontology.subjects(RDF.type, OWL.DatatypeProperty))
    props = objects | data_props
    instantiated = {c for c in classes if (None, RDF.type, c) in graph}
    used = {p for p in props if (None, p, None) in graph}
    linked = {s for s, p, o in graph if p in objects}
    subjects = set(graph.subjects())
    attrs = {(c, p) for s, p, _ in graph if p in data_props for c in graph.objects(s, RDF.type) if c in classes}
    hierarchy = len(set(ontology.triples((None, RDFS.subClassOf, None))))
    def ratio(n, d): return None if not d else n / d
    return {'formula_version': 'provisional_internal_v1', 'official_formulas': 'NOT_FULLY_SPECIFIED',
            'scope': 'asserted documentary graph; explicitly declared named ontology classes/properties; no closure in denominators',
            'inconsistency': {'bounded_owlrl_rule_errors': len(logic['rule_errors']), 'completed': logic['completed'],
                             'unsatisfiable_classes': 'NOT_EVALUATED', 'full_owl_consistency': 'NOT_EVALUATED'},
            'triple_count': {'formula': '|asserted data triples|', 'value': len(graph)},
            'relationship_richness': {'formula': '|declared object properties| / (|declared object properties| + |subClassOf triples|)',
                                     'numerator': len(objects), 'denominator': len(objects)+hierarchy,
                                     'value': ratio(len(objects), len(objects)+hierarchy)},
            'attribute_richness': {'formula': '|observed (declared class, datatype property) pairs| / |declared named classes|',
                                  'numerator': len(attrs), 'denominator': len(classes), 'value': ratio(len(attrs), len(classes))},
            'instantiation': {'formula': 'classes or properties used in asserted data / explicitly declared named classes or properties',
                              'classes': [len(instantiated), len(classes)], 'properties': [len(used), len(props)],
                              'class_value': ratio(len(instantiated), len(classes)), 'property_value': ratio(len(used), len(props))},
            'interlinking': {'formula': '|data subjects with a declared object-property edge| / |data subjects|',
                            'numerator': len(linked), 'denominator': len(subjects), 'value': ratio(len(linked), len(subjects)),
                            'interpretation': 'internal graph linking; not external KG linkage coverage'}}


def build_report(project):
    project = Path(project)
    ontology, shapes = load_artifacts(project)
    mapping = validate_mapping(project)
    corpus_path = project / 'data/post_confirmation_v10/pathology/dated_cases/corpus.json'
    corpus = json.loads(corpus_path.read_text())
    for source in corpus['sources']:
        path = (project / source['snapshot_path']).resolve()
        path.relative_to(project.resolve())
        if sha256(path) != source['snapshot_sha256']:
            raise ValueError('Documentary snapshot bytes/hash mismatch')
    graph = documentary_graph(corpus)
    from rdflib import Graph
    from rdflib.compare import isomorphic
    parsed = Graph().parse(data=graph.serialize(format='nt'), format='nt')
    if not isomorphic(graph, parsed):
        raise ValueError('Export/parser roundtrip mismatch')
    graph = parsed
    result, _ = validate_graph(graph, ontology, shapes)
    paths = ['ontology/nih_v11.owl', 'ontology/nih_v11_shapes.ttl', 'ontology/biolink_mapping_v11.yaml',
             'src/invariantcell/nih_v11.py', 'scripts/nih_v11_validate.py', 'src/invariantcell/ontology_validation.py',
             str(corpus_path.relative_to(project))] + [source['snapshot_path'] for source in corpus['sources']]
    return {'version': 'nih-v11.1', 'purpose': 'formal metadata interoperability; no empirical benchmark rerun',
            'resource_eligibility': 'DOCUMENTATION_AND_DEPENDENCY_REVIEW_PENDING', 'dependencies': dependency_status(),
            'artifact_sha256': {p: sha256(project / p) for p in paths}, 'mapping': mapping, 'ntriples_roundtrip_isomorphic': True, **result,
            'metrics': ontology_metrics(graph, ontology, result['owlrl'])}, graph

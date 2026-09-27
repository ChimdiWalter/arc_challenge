"""Optional independent RDF parser, SHACL shapes and bounded OWL-RL checks.

Install only the project-local optional wheels documented in the status report.
No user-provided shapes, imports, URLs or executable rules are fetched. This
module does not claim full OWL consistency/completeness or biological truth.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import time

PROFILE = "invariantcell-attributed-go-validation-v1"
SHAPES = '''
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix biolink: <https://w3id.org/biolink/vocab/> .
@prefix ic: <urn:invariantcell:ontology-mapping:> .
ic:AssociationShape a sh:NodeShape ;
    sh:targetClass biolink:Association ; sh:targetSubjectsOf rdf:subject, ic:raw_gaf_row ;
    sh:class biolink:Association ; sh:nodeKind sh:IRI ;
    sh:property [ sh:path rdf:subject ; sh:minCount 1 ; sh:maxCount 1 ; sh:nodeKind sh:IRI ] ;
    sh:property [ sh:path rdf:object ; sh:minCount 1 ; sh:maxCount 1 ; sh:nodeKind sh:IRI ] ;
    sh:property [ sh:path rdf:predicate ; sh:maxCount 1 ; sh:nodeKind sh:IRI ] ;
    sh:or ( [ sh:property [ sh:path rdf:predicate ; sh:minCount 1 ] ]
            [ sh:property [ sh:path ic:mapping_issue ; sh:minCount 1 ] ] ) ;
    sh:property [ sh:path biolink:negated ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:boolean ] ;
    sh:property [ sh:path ic:annotation_date ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:date ] ;
    sh:property [ sh:path ic:raw_gaf_row ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:string ] ;
    sh:property [ sh:path ic:source_sha256 ; sh:minCount 1 ; sh:maxCount 1 ; sh:pattern "^[0-9a-f]{64}$" ; sh:datatype xsd:string ] ;
    sh:property [ sh:path ic:ontology_sha256 ; sh:minCount 1 ; sh:maxCount 1 ; sh:pattern "^[0-9a-f]{64}$" ; sh:datatype xsd:string ] ;
    sh:property [ sh:path ic:public_available_at ; sh:maxCount 1 ; sh:datatype xsd:dateTime ] ;
    sh:property [ sh:path ic:ingested_at ; sh:maxCount 1 ; sh:datatype xsd:dateTime ] ;
    sh:or ( [ sh:property [ sh:path biolink:negated ; sh:hasValue true ] ;
              sh:property [ sh:path ic:gaf_qualifier ; sh:hasValue "NOT"^^xsd:string ] ]
            [ sh:property [ sh:path biolink:negated ; sh:hasValue false ] ;
              sh:not [ sh:property [ sh:path ic:gaf_qualifier ; sh:hasValue "NOT"^^xsd:string ] ] ] ) .
ic:GOTermShape a sh:NodeShape ; sh:targetClass owl:Class ;
    sh:property [ sh:path ic:go_namespace ; sh:minCount 1 ; sh:maxCount 1 ;
                  sh:in ("molecular_function"^^xsd:string "biological_process"^^xsd:string "cellular_component"^^xsd:string) ] ;
    sh:property [ sh:path owl:deprecated ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:boolean ] ;
    sh:property [ sh:path biolink:category ; sh:minCount 1 ; sh:maxCount 1 ; sh:nodeKind sh:IRI ] .
'''


def dependency_status() -> dict:
    versions, missing = {}, []
    for package in ("rdflib", "pyshacl", "owlrl"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            missing.append(package)
    return {"available": not missing, "versions": versions, "missing": missing}


def _require():
    if not dependency_status()["available"]:
        raise RuntimeError("Optional ontology dependencies unavailable; use PYTHONPATH=.deps/ontology/site-packages:src")


def parse_ntriples(text: str, *, max_bytes: int = 1_000_000, max_triples: int = 2000):
    """RDFLib parses literal text in memory; no location or automatic format input."""
    _require()
    from rdflib import Graph
    if not isinstance(text, str) or len(text.encode("utf-8")) > max_bytes:
        raise ValueError("RDF text exceeds byte budget or is not text")
    graph = Graph().parse(data=text, format="nt")
    if len(graph) > max_triples:
        raise ValueError("RDF graph exceeds triple budget")
    return graph


def _copy(graph):
    from rdflib import Graph
    copied = Graph()
    for triple in graph:
        copied.add(triple)
    return copied


def validate_shapes(graph) -> dict:
    """Execute fixed local SHACL Core shapes, without inferencing or imports."""
    _require()
    from rdflib import Graph, Namespace
    from rdflib.namespace import RDF, SH
    from pyshacl import validate
    biolink = Namespace("https://w3id.org/biolink/vocab/")
    ic = Namespace("urn:invariantcell:ontology-mapping:")
    associations = set(graph.subjects(RDF.type, biolink.Association)) | set(graph.subjects(RDF.subject)) | set(graph.subjects(ic.raw_gaf_row))
    conforms, report_graph, _ = validate(
        _copy(graph), shacl_graph=Graph().parse(data=SHAPES, format="turtle"),
        inference="none", advanced=False, js=False, do_owl_imports=False,
        meta_shacl=True, inplace=False, abort_on_first=False)
    violations = []
    for result in report_graph.subjects(RDF.type, SH.ValidationResult):
        violations.append({"focus": str(report_graph.value(result, SH.focusNode)),
                           "path": str(report_graph.value(result, SH.resultPath)),
                           "component": str(report_graph.value(result, SH.sourceConstraintComponent)),
                           "messages": sorted(str(m) for m in report_graph.objects(result, SH.resultMessage))})
    return {"status": "NO_ASSOCIATION_TARGETS" if not associations else "CONFORMS" if conforms else "VIOLATES_SHAPES",
            "conforms": bool(conforms), "association_targets": len(associations),
            "violations": sorted(violations, key=lambda v: (v["focus"], v["path"], v["component"])),
            "shapes_sha256": hashlib.sha256(SHAPES.encode()).hexdigest(),
            "scope": "fixed local attributed-GO SHACL Core shapes; not complete Biolink conformance",
            "inference": "none", "imports_loaded": False}


class _BudgetExceeded(RuntimeError):
    pass


def check_owlrl(graph, *, max_triples: int = 10000, max_seconds: float = 5.0, max_cycles: int = 50):
    """Return (report, expanded copy) using OWL-RL's actual rule-error interface.

    NO_CONTRADICTION_DETECTED means no implemented OWL-RL rule reported an
    error in this bounded run. It does not establish full OWL consistency.
    """
    _require()
    from owlrl import OWLRL_Semantics
    from rdflib.namespace import OWL
    if max_triples <= 0 or max_seconds <= 0 or max_cycles <= 0:
        raise ValueError("OWL-RL budgets must be positive")
    expanded, started = _copy(graph), time.monotonic()
    ignored_imports = sorted(str(v) for v in graph.objects(None, OWL.imports))

    class BoundedSemantics(OWLRL_Semantics):
        def store_triple(self, triple):
            super().store_triple(triple)
            if len(self.graph) + len(self.added_triples) > max_triples:
                raise _BudgetExceeded("OWL-RL triple budget exceeded")

        def rules(self, triple, cycle_num):
            if time.monotonic() - started > max_seconds or cycle_num > max_cycles:
                raise _BudgetExceeded("OWL-RL time/cycle budget exceeded")
            return super().rules(triple, cycle_num)

    engine = BoundedSemantics(expanded, axioms=False, daxioms=False, rdfs=False)
    complete, failure = True, None
    try:
        if len(expanded) > max_triples:
            raise _BudgetExceeded("OWL-RL initial triple budget exceeded")
        engine.closure()
        if len(expanded) > max_triples:
            raise _BudgetExceeded("OWL-RL final triple budget exceeded")
    except _BudgetExceeded as error:
        complete, failure = False, str(error)
    errors = sorted(engine.error_messages)
    return {"status": "INCOMPLETE_BUDGET" if not complete else "CONTRADICTION_DETECTED" if errors else "NO_CONTRADICTION_DETECTED",
            "completed": complete, "contradictions_detected": bool(errors) if complete else None,
            "rule_errors": errors, "budget_failure": failure, "input_triples": len(graph),
            "expanded_triples": len(expanded), "imports_not_loaded": ignored_imports,
            "scope": "OWL-RL library rule checks; no full OWL consistency/completeness claim",
            "full_owl_consistency": "NOT_EVALUATED", "unsatisfiable_classes": "NOT_EVALUATED",
            "datatype_axioms": False, "rdfs_closure": False}, expanded


def validate_ntriples(text: str) -> dict:
    """Report parse, SHACL and logical results separately; shape failure is not inconsistency."""
    _require()
    from rdflib.exceptions import ParserError
    report = {"profile": PROFILE, "dependencies": dependency_status(),
              "source_sha256": hashlib.sha256(text.encode()).hexdigest()}
    try:
        graph = parse_ntriples(text)
    except (ValueError, SyntaxError, ParserError) as error:
        return {**report, "rdf_parse": {"valid": False, "error": str(error)},
                "shacl": {"status": "NOT_RUN_PARSE_FAILED"}, "owlrl": {"status": "NOT_RUN_PARSE_FAILED"}}
    return {**report, "rdf_parse": {"valid": True, "triples": len(graph)},
            "shacl": validate_shapes(graph), "owlrl": check_owlrl(graph)[0]}

"""Bounded GO GAF/OBO intake, attributed RDF export, and existing-engine bridges.

This is a local shape contract plus finite positive Horn profile, not an OWL or
SHACL validator. Uninterpreted content is retained, never executed or imported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import re
from urllib.parse import quote

from .core import DependencyRef, canonical_hash, normalized_time, time_value
from .logic import reason, verify
from .temporal import EvidenceRecord, Snapshot

BIOLINK_VERSION = "4.3.6"
BIOLINK_SCHEMA_URL = "https://raw.githubusercontent.com/biolink/biolink-model/v4.3.6/biolink-model.yaml"
BIOLINK_SCHEMA_SHA256 = "37146d876aee83c1be58baf7da5a5bc97d6f3a5f5d5017484cf3b6cba0ece305"
PROFILE = "invariantcell-go-mapping-v1"
GO_ID = re.compile(r"GO:[0-9]{7}\Z")
CURIE = re.compile(r"[^\s:|,]+:[^\s|,]+\Z")
NAMESPACES = {"F": "molecular_function", "P": "biological_process", "C": "cellular_component"}
CATEGORIES = {"molecular_function": "biolink:MolecularActivity", "biological_process": "biolink:BiologicalProcess",
              "cellular_component": "biolink:CellularComponent"}
# Only exact RO/BFO mappings confirmed in the pinned primary Biolink schema.
RELATIONS = {"enables": ("biolink:enables", "RO:0002327", "F"),
             "contributes_to": ("biolink:contributes_to", "RO:0002326", "F"),
             "involved_in": ("biolink:actively_involved_in", "RO:0002331", "P"),
             "located_in": ("biolink:located_in", "RO:0001025", "C"),
             "active_in": ("biolink:active_in", "RO:0002432", "C"),
             "part_of": ("biolink:part_of", "BFO:0000050", "C")}
SUBJECT_CATEGORIES = {"protein": "biolink:Protein", "gene": "biolink:Gene"}


def mapping_contract() -> dict:
    return {"profile": PROFILE, "biolink_version": BIOLINK_VERSION,
            "biolink_schema_url": BIOLINK_SCHEMA_URL, "biolink_schema_sha256": BIOLINK_SCHEMA_SHA256,
            "relations": RELATIONS, "go_categories": CATEGORIES, "subject_categories": SUBJECT_CATEGORIES,
            "legacy_missing_relation": "unmapped", "obsolete_policy": "retain_and_exclude",
            "temporal_evidence_state": "requires_explicit_per_annotation_declaration",
            "mapped_ingestion": "max_of_source_ontology_mapping_or_unknown",
            "extension_policy": "retain_and_exclude_from_temporal_bridge"}


@dataclass(frozen=True)
class DocumentSource:
    source_id: str
    sha256: str
    public_available_at: str | None
    ingested_at: str | None
    synthetic: bool

    @classmethod
    def from_text(cls, text: str, source_id: str, *, public_available_at: str | None = None,
                  ingested_at: str | None = None, synthetic: bool = False) -> DocumentSource:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        ref = DependencyRef(source_id, digest, public_available_at, "source")
        if type(synthetic) is not bool:
            raise ValueError("synthetic must be boolean")
        return cls(source_id, digest, ref.available_at, normalized_time(ingested_at), synthetic)


@dataclass(frozen=True)
class GOTerm:
    id: str
    name: str
    namespace: str
    alt_ids: tuple[str, ...]
    is_a: tuple[str, ...]
    obsolete: bool
    replaced_by: tuple[str, ...]
    consider: tuple[str, ...]
    raw_tags: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GOOntology:
    source: DocumentSource
    headers: tuple[tuple[str, str], ...]
    terms: tuple[GOTerm, ...]
    unmodeled_stanzas: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]

    @property
    def version(self) -> str | None:
        return next((v for k, v in self.headers if k == "data-version"), None)

    def index(self) -> dict[str, GOTerm]:
        return {identifier: term for term in self.terms for identifier in (term.id, *term.alt_ids)}


def _obo_value(raw: str) -> str:
    """Remove comments only outside quotes and escaped characters."""
    quoted = escaped = False
    for pos, char in enumerate(raw):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == "!" and not quoted:
            return raw[:pos].strip()
    return raw.strip()


def parse_obo(text: str, source_id: str, **source_times) -> GOOntology:
    """Parse GO term metadata and plain is_a; preserve other tags without inference.

    Imports are not fetched. Structural qualifiers/modifiers on modeled ID tags
    are rejected; unsupported axioms remain in raw_tags and profile disclosures.
    """
    source = DocumentSource.from_text(text, source_id, **source_times)
    headers: list[tuple[str, str]] = []
    stanzas: list[tuple[str, list[tuple[str, str]]]] = []
    current = headers
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("["):
            if not re.fullmatch(r"\[[A-Za-z]+\]", line):
                raise ValueError(f"OBO line {number}: malformed stanza")
            current = []
            stanzas.append((line[1:-1], current))
        else:
            key, sep, value = line.partition(":")
            if not sep or not key or " " in key:
                raise ValueError(f"OBO line {number}: expected tag: value")
            current.append((key, value.strip()))
    versions = [_obo_value(v) for k, v in headers if k == "format-version"]
    if len(versions) != 1 or versions[0] not in {"1.2", "1.4"}:
        raise ValueError("OBO requires one supported format-version (1.2 or 1.4)")
    if sum(k == "data-version" for k, _ in headers) > 1:
        raise ValueError("duplicate OBO data-version")
    terms, other = [], []
    for kind, tags in stanzas:
        if kind != "Term":
            other.append((kind, tuple(tags)))
            continue
        values: dict[str, list[str]] = {}
        for key, value in tags:
            values.setdefault(key, []).append(_obo_value(value))
        for key in ("id", "name", "namespace"):
            if len(values.get(key, [])) != 1 or not values[key][0]:
                raise ValueError(f"GO term requires exactly one {key}")
        if values["namespace"][0] not in CATEGORIES:
            raise ValueError("unsupported GO namespace")
        for key in ("id", "alt_id", "is_a", "replaced_by", "consider"):
            if any(not GO_ID.fullmatch(v) for v in values.get(key, [])):
                raise ValueError(f"invalid or unsupported qualified GO identifier in {key}")
        obsolete_values = values.get("is_obsolete", ["false"])
        if len(obsolete_values) != 1 or obsolete_values[0] not in {"true", "false"}:
            raise ValueError("is_obsolete must be a single boolean")
        obsolete = obsolete_values[0] == "true"
        if not obsolete and (values.get("replaced_by") or values.get("consider")):
            raise ValueError("replacement/consider requires obsolete term")
        if obsolete and any(values.get(k) for k in ("is_a", "relationship", "intersection_of", "union_of", "disjoint_from")):
            raise ValueError("obsolete term cannot retain structural axioms")
        terms.append(GOTerm(values["id"][0], values["name"][0], values["namespace"][0],
                            tuple(values.get("alt_id", [])), tuple(values.get("is_a", [])), obsolete,
                            tuple(values.get("replaced_by", [])), tuple(values.get("consider", [])), tuple(tags)))
    identifiers = [i for t in terms for i in (t.id, *t.alt_ids)]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate or ambiguous GO primary/alternative identifier")
    return GOOntology(source, tuple(headers), tuple(terms), tuple(other))


@dataclass(frozen=True)
class GAFAnnotation:
    line_number: int
    columns: tuple[str, ...]
    qualifiers: tuple[str, ...]
    relation: str | None
    negated: bool
    annotation_date: str

    @property
    def subject(self) -> str:
        return self.columns[16] or f"{self.columns[0]}:{self.columns[1]}"


@dataclass(frozen=True)
class GAFDocument:
    source: DocumentSource
    version: str
    headers: tuple[str, ...]
    annotations: tuple[GAFAnnotation, ...]


def parse_gaf(text: str, source_id: str, **source_times) -> GAFDocument:
    """Validate local column/date/ID shapes; do not imply complete GO compliance."""
    source = DocumentSource.from_text(text, source_id, **source_times)
    headers, annotations, version = [], [], None
    for number, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        if line.startswith("!"):
            headers.append(line)
            if line.startswith("!gaf-version:"):
                if version is not None or annotations:
                    raise ValueError("duplicate or late GAF version header")
                version = line.partition(":")[2].strip()
                if version not in {"2.1", "2.2"}:
                    raise ValueError("only GAF 2.1 and 2.2 are supported")
            continue
        if version is None:
            raise ValueError("GAF requires a version header before annotations")
        columns = tuple(line.split("\t"))
        if len(columns) != 17:
            raise ValueError(f"GAF line {number}: expected 17 columns")
        if any(not columns[i].strip() for i in (0, 1, 2, 4, 5, 6, 8, 11, 12, 13, 14)):
            raise ValueError(f"GAF line {number}: missing required column")
        if not GO_ID.fullmatch(columns[4]) or columns[8] not in NAMESPACES:
            raise ValueError(f"GAF line {number}: invalid GO identifier or aspect")
        if ":" in columns[0] or not CURIE.fullmatch(f"{columns[0]}:{columns[1]}"):
            raise ValueError("invalid GAF subject identifier")
        if any(not CURIE.fullmatch(v) for v in columns[5].split("|")):
            raise ValueError("invalid reference identifier")
        taxa = columns[12].split("|")
        if len(taxa) not in {1, 2} or any(not re.fullmatch(r"taxon:[1-9][0-9]*", t) for t in taxa):
            raise ValueError("invalid GAF taxon identifiers")
        if columns[16] and not CURIE.fullmatch(columns[16]):
            raise ValueError("invalid gene product form identifier")
        if not re.fullmatch(r"[0-9]{8}", columns[13]):
            raise ValueError("annotation date must be YYYYMMDD")
        try:
            annotation_date = datetime.strptime(columns[13], "%Y%m%d").date().isoformat()
        except ValueError as error:
            raise ValueError("invalid annotation date") from error
        qualifiers = tuple(columns[3].split("|")) if columns[3] else ()
        if any(not q for q in qualifiers) or len(set(qualifiers)) != len(qualifiers):
            raise ValueError("empty or duplicate GAF qualifier")
        relation = tuple(q for q in qualifiers if q != "NOT")
        if len(relation) > 1 or (version == "2.2" and (len(relation) != 1 or qualifiers not in (relation, ("NOT", *relation)))):
            raise ValueError("GAF 2.2 requires a relation optionally prefixed with NOT")
        if version == "2.1" and any(q not in {"NOT", "contributes_to", "colocalizes_with"} for q in qualifiers):
            raise ValueError("unsupported GAF 2.1 qualifier")
        annotations.append(GAFAnnotation(number, columns, qualifiers, relation[0] if relation else None,
                                         "NOT" in qualifiers, annotation_date))
    if version is None:
        raise ValueError("missing GAF version header")
    if version == "2.2":
        for required in ("!generated-by:", "!date-generated:"):
            if not any(h.startswith(required) and h[len(required):].strip() for h in headers):
                raise ValueError(f"GAF 2.2 requires {required} header")
    return GAFDocument(source, version, tuple(headers), tuple(annotations))


@dataclass(frozen=True)
class MappedAnnotation:
    id: str
    subject: str
    subject_category: str | None
    predicate: str | None
    object: str
    object_category: str | None
    negated: bool
    annotation: GAFAnnotation
    issues: tuple[str, ...]


@dataclass(frozen=True)
class OntologyExport:
    ontology: GOOntology
    gaf: GAFDocument
    associations: tuple[MappedAnnotation, ...]

    def to_dict(self) -> dict:
        return {"profile": PROFILE, "mapping_contract": mapping_contract(), **asdict(self),
                "validation_scope": "local shapes only; no SHACL/OWL/complete Biolink validation",
                "logical_checks": "NOT_EVALUATED; call check_ontology_logic explicitly"}


def map_annotations(gaf: GAFDocument, ontology: GOOntology) -> OntologyExport:
    """Preserve every annotation, including unmapped and obsolete records."""
    index, associations = ontology.index(), []
    for annotation in gaf.annotations:
        cols, issues = annotation.columns, []
        term = index.get(cols[4])
        subject_category = SUBJECT_CATEGORIES.get(cols[11])
        relation = RELATIONS.get(annotation.relation)
        if subject_category is None:
            issues.append("unsupported_subject_type")
        if relation is None:
            issues.append("unmapped_relation")
        elif cols[8] != relation[2]:
            issues.append("relation_aspect_mismatch")
        if term is None:
            issues.append("unresolved_go_identifier")
        else:
            if term.namespace != NAMESPACES[cols[8]]:
                issues.append("go_namespace_aspect_mismatch")
            if term.obsolete:
                issues.append("obsolete_go_term")
        if cols[15]:
            issues.append("uninterpreted_annotation_extension")
        identity = canonical_hash({"source": gaf.source.sha256, "line": annotation.line_number, "columns": cols})
        associations.append(MappedAnnotation(f"urn:invariantcell:gaf:{identity}", annotation.subject,
                                            subject_category, relation[0] if relation else None,
                                            term.id if term else cols[4], CATEGORIES[term.namespace] if term else None,
                                            annotation.negated, annotation, tuple(issues)))
    return OntologyExport(ontology, gaf, tuple(associations))


def check_ontology_shapes(ontology: GOOntology) -> dict:
    """Referential integrity and active GO is_a shape checks, separate from logic."""
    index, errors = ontology.index(), []
    for term in ontology.terms:
        for relation, targets in (("is_a", term.is_a), ("replaced_by", term.replaced_by), ("consider", term.consider)):
            for target in targets:
                parent = index.get(target)
                if parent is None:
                    errors.append((term.id, relation, target, "unresolved_reference"))
                elif relation == "is_a" and (parent.obsolete or parent.namespace != term.namespace):
                    errors.append((term.id, relation, target, "invalid_active_parent_namespace"))
    return {"valid": not errors, "errors": errors, "scope": "local shape/reference checks; not SHACL or logical consistency"}


def ontology_logic_inputs(ontology: GOOntology) -> tuple[dict, list[list[str]]]:
    if not check_ontology_shapes(ontology)["valid"]:
        raise ValueError("ontology shape errors must be resolved before logical checks")
    index = ontology.index()
    facts = sorted({(t.id, "rdfs:subClassOf", index[p].id) for t in ontology.terms if not t.obsolete for p in t.is_a})
    profile = {"logic": "finite-positive-horn-triples-v1", "rules": [
        {"id": "go-is-a-transitivity", "body": [["?a", "rdfs:subClassOf", "?b"], ["?b", "rdfs:subClassOf", "?c"]],
         "head": ["?a", "rdfs:subClassOf", "?c"]}], "irreflexive": ["rdfs:subClassOf"], "disjoint_types": []}
    return profile, [list(t) for t in facts]


def check_ontology_logic(ontology: GOOntology, *, max_facts: int = 1000) -> dict:
    """Replay is_a closure and a GO DAG constraint; an OWL cycle need not be inconsistent."""
    profile, facts = ontology_logic_inputs(ontology)
    if len(facts) > max_facts:
        raise ValueError("initial ontology facts exceed finite closure budget")
    certificate = reason(profile, facts, max_facts=max_facts)
    ignored = sorted({k for t in ontology.terms for k, _ in t.raw_tags
                      if k in {"relationship", "intersection_of", "union_of", "equivalent_to", "disjoint_from"}})
    return {"profile": "is_a transitivity and GO DAG constraint only", "certificate": certificate,
            "replay": verify(certificate, profile, facts), "go_dag_valid": not certificate["conflicts"],
            "uninterpreted_axiom_tags": ignored,
            "imports_not_loaded": [v for k, v in ontology.headers if k == "import"],
            "unmodeled_stanza_count": len(ontology.unmodeled_stanzas),
            "owl_consistency": "NOT_EVALUATED", "unsatisfiable_classes": "NOT_EVALUATED",
            "annotation_entailment": "NOT_EVALUATED; NOT is not passed to positive Horn logic"}


@dataclass(frozen=True)
class TemporalMapping:
    snapshot: Snapshot
    excluded: tuple[tuple[str, tuple[str, ...]], ...]


def to_temporal_snapshot(export: OntologyExport, snapshot_id: str, *, context: str,
                         mapping_available_at: str | None = None,
                         mapping_ingested_at: str | None = None,
                         validity_by_id: dict[str, tuple[str | None, str | None]] | None = None,
                         state_kind_by_id: dict[str, str] | None = None) -> TemporalMapping:
    """Bridge attributed assertions, never infer validity or release dates from GAF dates.

    Explicit per-annotation validity is an additional caller assertion. Without
    it, existing valid-at queries abstain. State kind also requires an explicit
    per-annotation declaration: a GO evidence code is not a measurement.
    Mapping availability covers the chosen
    Biolink schema and this mapping contract, not the GAF annotation date.
    Mapped ingestion is the latest declared GAF/OBO/mapping ingestion; any
    unknown dependency ingestion keeps mapped ingestion unknown.
    """
    validity_by_id = {} if validity_by_id is None else validity_by_id
    state_kind_by_id = {} if state_kind_by_id is None else state_kind_by_id
    if not context.strip():
        raise ValueError("context must be nonempty")
    if (set(validity_by_id) | set(state_kind_by_id)) - {a.id for a in export.associations}:
        raise ValueError("metadata supplied for unknown annotation")
    if any(value not in {"observed", "predicted"} for value in state_kind_by_id.values()):
        raise ValueError("state kind must be observed or predicted")
    if not check_ontology_shapes(export.ontology)["valid"]:
        raise ValueError("ontology shape errors block temporal mapping")
    dependencies = (
        DependencyRef(f"gaf:{export.gaf.source.sha256}", export.gaf.source.sha256, export.gaf.source.public_available_at, "source"),
        DependencyRef(f"go:{export.ontology.source.sha256}", export.ontology.source.sha256,
                      export.ontology.source.public_available_at, "ontology"),
        DependencyRef(f"mapping:{PROFILE}", canonical_hash(mapping_contract()), mapping_available_at, "mapping"))
    ingestion_times = (export.gaf.source.ingested_at, export.ontology.source.ingested_at,
                       normalized_time(mapping_ingested_at))
    mapped_ingested_at = (max(ingestion_times, key=time_value)
                          if all(t is not None for t in ingestion_times) else None)
    evidence, excluded = [], []
    for association in export.associations:
        issues = association.issues + (() if association.id in state_kind_by_id else ("evidence_state_not_declared",))
        if issues:
            excluded.append((association.id, issues))
            continue
        annotation = association.annotation
        scoped_context = context + ":gaf-scope:" + canonical_hash({"taxa": annotation.columns[12],
                                                                  "extension": annotation.columns[15]})
        bounds = validity_by_id.get(association.id, (None, None))
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError("validity must contain [from, to) bounds")
        evidence.append(EvidenceRecord(
            evidence_id=association.id, subject=association.subject, predicate=association.predicate,
            value=association.object, context=scoped_context, polarity="refutes" if association.negated else "supports",
            public_available_at=export.gaf.source.public_available_at, ingested_at=mapped_ingested_at,
            source_sha256=canonical_hash(asdict(association)), dependencies=tuple(d.dependency_id for d in dependencies),
            synthetic=export.gaf.source.synthetic or export.ontology.source.synthetic,
            state_kind=state_kind_by_id[association.id],
            validity_kind="interval" if association.id in validity_by_id else "unknown",
            valid_from=bounds[0], valid_to=bounds[1]))
    return TemporalMapping(Snapshot(snapshot_id, tuple(evidence), dependencies), tuple(excluded))


_IRI = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#", "xsd": "http://www.w3.org/2001/XMLSchema#",
        "biolink": "https://w3id.org/biolink/vocab/", "ic": "urn:invariantcell:ontology-mapping:"}


def _iri(value: str) -> str:
    if value.startswith("urn:invariantcell:gaf:"):
        return "<" + value + ">"
    prefix, _, local = value.partition(":")
    if prefix == "GO" and GO_ID.fullmatch(value):
        return "<http://purl.obolibrary.org/obo/GO_" + local + ">"
    # Unknown namespaces are encoded losslessly as local URNs, not invented URLs.
    return "<" + (_IRI[prefix] + quote(local, safe="_") if prefix in _IRI
                  else "urn:invariantcell:curie:" + quote(value, safe="")) + ">"


def export_ntriples(export: OntologyExport) -> str:
    """Typed RDF reification. Neither a positive nor a NOT annotation asserts a bare edge."""
    triples: set[str] = set()

    def add(subject, predicate, value, datatype=None):
        obj = _iri(value) if datatype is None else json.dumps(value, ensure_ascii=False) + "^^" + _iri(datatype)
        triples.add(f"{_iri(subject)} {_iri(predicate)} {obj} .")

    index = export.ontology.index()
    for term in export.ontology.terms:
        add(term.id, "rdf:type", "owl:Class")
        add(term.id, "biolink:category", CATEGORIES[term.namespace])
        add(term.id, "rdfs:label", term.name, "xsd:string")
        add(term.id, "ic:go_namespace", term.namespace, "xsd:string")
        add(term.id, "owl:deprecated", str(term.obsolete).lower(), "xsd:boolean")
        add(term.id, "ic:raw_obo_tags", json.dumps(term.raw_tags, ensure_ascii=False), "xsd:string")
        add(term.id, "ic:ontology_sha256", export.ontology.source.sha256, "xsd:string")
        if export.ontology.version is not None:
            add(term.id, "ic:ontology_version", export.ontology.version, "xsd:string")
        for key, targets in (("ic:alt_id", term.alt_ids), ("ic:replaced_by", term.replaced_by), ("ic:consider", term.consider)):
            for target in targets:
                add(term.id, key, target)
        for parent in term.is_a:
            add(term.id, "rdfs:subClassOf", index[parent].id if parent in index else parent)
    for association in export.associations:
        aid, annotation = association.id, association.annotation
        add(aid, "rdf:type", "biolink:Association")
        add(aid, "rdf:subject", association.subject)
        add(aid, "rdf:object", association.object)
        if association.predicate:
            add(aid, "rdf:predicate", association.predicate)
        if association.subject_category:
            add(aid, "biolink:subject_category", association.subject_category)
        if association.object_category:
            add(aid, "biolink:object_category", association.object_category)
        add(aid, "biolink:negated", str(association.negated).lower(), "xsd:boolean")
        for qualifier in annotation.qualifiers:
            add(aid, "ic:gaf_qualifier", qualifier, "xsd:string")
        add(aid, "ic:annotation_date", annotation.annotation_date, "xsd:date")
        add(aid, "ic:raw_gaf_row", "\t".join(annotation.columns), "xsd:string")
        add(aid, "ic:original_go_id", annotation.columns[4], "xsd:string")
        add(aid, "ic:evidence_code", annotation.columns[6], "xsd:string")
        add(aid, "ic:source_sha256", export.gaf.source.sha256, "xsd:string")
        add(aid, "ic:ontology_sha256", export.ontology.source.sha256, "xsd:string")
        add(aid, "ic:mapping_contract_sha256", canonical_hash(mapping_contract()), "xsd:string")
        for field in ("public_available_at", "ingested_at"):
            value = getattr(export.gaf.source, field)
            if value is not None:
                add(aid, "ic:" + field, value, "xsd:dateTime")
        for issue in association.issues:
            add(aid, "ic:mapping_issue", issue, "xsd:string")
    return "\n".join(sorted(triples)) + "\n"

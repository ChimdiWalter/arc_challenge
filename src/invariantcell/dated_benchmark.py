"""Frozen dated-source replay for a bounded biomedical regulatory-event corpus.

Day-only source dates remain day-only. Derived conservative UTC bounds are
explicit computational bounds, never observed publication or event timestamps.
This module is shared temporal-method engineering, not NIH-prize evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time as clock

from .core import DependencyRef, canonical_hash, normalized_time, time_value
from .temporal import EvidenceRecord, Snapshot, TemporalQuery, evaluate, issue_certificate
from .replay import verify_certificate

METHODS = ("static_kg", "timestamp_filter", "simple_temporal_rules", "verified_temporal")
PURPOSE = "shared-temporal-method-development"
VERSION = "dated-public-regulatory-v1"


def date_bounds(day: str | None) -> tuple[str | None, str | None]:
    """Conservative bounds for an unknown time/zone, allowing UTC offsets -12..+14.

    A civil day's first possible instant is previous-day 10:00Z and last possible
    upper bound is following-day 12:00Z. Neither bound is an observed instant.
    """
    if day is None:
        return None, None
    d = date.fromisoformat(day)
    if d.isoformat() != day:
        raise ValueError("date must have YYYY-MM-DD precision")
    midnight = datetime.combine(d, time(), timezone.utc)
    return ((midnight - timedelta(hours=14)).isoformat().replace("+00:00", "Z"),
            (midnight + timedelta(hours=36)).isoformat().replace("+00:00", "Z"))


@dataclass(frozen=True)
class Benchmark:
    records: tuple[dict, ...]
    sources: tuple[dict, ...]
    queries: tuple[dict, ...]
    gold: tuple[dict, ...]
    freeze_sha256: str

    def source_index(self):
        return {s["source_id"]: s for s in self.sources}

    def record_index(self):
        return {r["record_id"]: r for r in self.records}


def validate_bundle(bundle: Benchmark) -> None:
    sources, records = bundle.source_index(), bundle.record_index()
    if len(sources) != len(bundle.sources) or len(records) != len(bundle.records):
        raise ValueError("duplicate source or assertion identity")
    for source in bundle.sources:
        date_bounds(source["published_date"])
        normalized_time(source["retrieved_at"])
        DependencyRef(source["source_id"], source["sha256"], date_bounds(source["published_date"])[1])
    for record in bundle.records:
        source = sources.get(record["source_id"])
        if source is None or record["source_sha256"] != source["sha256"]:
            raise ValueError("unresolved or mismatched source provenance")
        date_bounds(record["event_date"])
        if record["event_time"] is not None:
            raise ValueError("this day-granularity benchmark must not invent exact event times")
        if record["polarity"] not in {"supports", "refutes"}:
            raise ValueError("invalid assertion polarity")
        if record["validity"] not in {"administrative_from_event", "unknown_start", "unknown_biological"}:
            raise ValueError("unsupported validity contract")
        for predecessor in record["supersedes"]:
            old = records.get(predecessor)
            if old is None or any(old[k] != record[k] for k in ("subject", "predicate", "value", "context")):
                raise ValueError("supersession must identify the same scoped proposition")
            if not old["event_date"] or not record["event_date"] or old["event_date"] >= record["event_date"]:
                raise ValueError("supersession requires strictly ordered explicit event days")
    qids = [q["query_id"] for q in bundle.queries]
    gids = [g["query_id"] for g in bundle.gold]
    if len(qids) != len(set(qids)) or len(gids) != len(set(gids)) or set(qids) != set(gids):
        raise ValueError("one independent gold outcome is required per unique query")
    for query in bundle.queries:
        if query["kind"] not in {"status", "history", "event_before", "path"}:
            raise ValueError("unsupported query kind")
        for field in ("valid_at", "known_as_of", "system_as_of"):
            normalized_time(query[field])
        if query["kind"] == "path" and (len(query.get("path", [])) < 2 or any(i not in records for i in query["path"])):
            raise ValueError("path requires at least two existing assertions")
        if query["kind"] == "event_before" and query.get("record_id") not in records:
            raise ValueError("event query requires an existing assertion")


def _safe_path(root: Path, name: str) -> Path:
    resolved = (root / name).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("corpus path escapes project root")
    return resolved


def load_benchmark(root: str | Path) -> Benchmark:
    """Verify frozen bytes and source anchors before exposing any method inputs."""
    root = Path(root).resolve()
    freeze_path = root / "configs/dated_benchmark_freeze.json"
    freeze_raw = freeze_path.read_bytes()
    freeze = json.loads(freeze_raw)
    if freeze["method_outputs_examined"] is not False:
        raise ValueError("gold freeze is not declared pre-evaluation")
    for name, expected in freeze["files"].items():
        if hashlib.sha256(_safe_path(root, name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen input changed: {name}")
    def read(name):
        return json.loads(_safe_path(root, name).read_text())
    sources = read("data/public/dated_benchmark/sources.json")["sources"]
    records = read("data/public/dated_benchmark/records.json")["records"]
    bundle = Benchmark(tuple(records), tuple(sources), tuple(read("configs/dated_benchmark_queries.json")["queries"]),
                       tuple(read("configs/dated_benchmark_gold.json")["gold"]), hashlib.sha256(freeze_raw).hexdigest())
    validate_bundle(bundle)
    for record in records:
        source = bundle.source_index()[record["source_id"]]
        text = _safe_path(root, source["text_path"]).read_text()
        start = record["anchor_start"]
        if text[start:start + len(record["anchor"])] != record["anchor"]:
            raise ValueError("source anchor does not replay")
        if hashlib.sha256(_safe_path(root, source["html_path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("source checksum mismatch")
    return bundle


def _available(record, source, query, *, system=False):
    upper = date_bounds(source["published_date"])[1]
    return (upper is not None and time_value(upper) <= time_value(query["known_as_of"])
            and (not system or time_value(source["retrieved_at"]) <= time_value(query["system_as_of"])))


def _matching(bundle, query):
    return [r for r in bundle.records if all(r[k] == query[k] for k in ("subject", "predicate", "value", "context"))]


def _summarize(records):
    polarities = {r["polarity"] for r in records}
    return "CONFLICT" if len(polarities) > 1 else "SUPPORTED" if polarities == {"supports"} else "REFUTED" if polarities else "UNKNOWN"


def _result(outcome, used=(), *, certificates=(), notes=()):
    return {"outcome": outcome, "used_record_ids": sorted(set(used)),
            "certificates": list(certificates), "notes": list(notes)}


def _state_interval(record, admitted):
    if record["validity"] != "administrative_from_event" or record["event_date"] is None:
        return "unknown", None, None
    start = date_bounds(record["event_date"])[1]
    endings = [date_bounds(r["event_date"])[0] for r in admitted if record["record_id"] in r["supersedes"]]
    return "interval", start, min(endings) if endings else None


def _verified_snapshot(bundle, query, records, *, history):
    """Existing engine checks projected conservative bounds against external anchors.

    The source data keep exact times unknown. The snapshot explicitly reasons
    about date-bound-admissible attributed assertions, not measured event times.
    """
    sources = bundle.source_index()
    admitted = [r for r in records if _available(r, sources[r["source_id"]], query, system=True)]
    dependencies = {s["source_id"]: DependencyRef("source:" + s["source_id"], s["sha256"],
                                                date_bounds(s["published_date"])[1], "source") for s in sources.values()}
    evidence = []
    for r in records:
        source = sources[r["source_id"]]
        upper = date_bounds(source["published_date"])[1]
        kind, start, end = ("interval", upper, None) if history and upper else _state_interval(r, admitted)
        projected_predicate = "documentary_history:" + r["predicate"] if history else r["predicate"]
        evidence.append(EvidenceRecord(r["record_id"], r["subject"], projected_predicate, r["value"], r["context"],
                                       r["polarity"], upper, source["retrieved_at"], canonical_hash(r),
                                       valid_from=start, valid_to=end, event_time=None,
                                       dependencies=("source:" + source["source_id"],), synthetic=r["synthetic"],
                                       validity_kind=kind))
    snapshot = Snapshot("dated:" + query["query_id"], tuple(evidence), tuple(dependencies.values()))
    intended = TemporalQuery(query["subject"], "documentary_history:" + query["predicate"] if history else query["predicate"], query["context"], query["known_as_of"],
                             query["system_as_of"], query["known_as_of"] if history else query["valid_at"])
    certificate = issue_certificate(snapshot, intended)
    replay = verify_certificate(certificate, snapshot, canonical_hash(snapshot.to_dict()), intended)
    if not replay.valid:
        raise ValueError("existing temporal certificate failed independent replay")
    result = evaluate(snapshot, intended)
    used = sorted({i for a in result.answers for i in (*a.support_ids, *a.refute_ids)})
    certificate_record = {"certificate": certificate, "replay_valid": replay.valid,
                          "projection": "conservative civil-date bounds; original exact event times remain unknown"}
    return used, certificate_record


def run_method(bundle: Benchmark, query: dict, method: str) -> dict:
    """Every method sees identical typed records; gold is never inspected here."""
    if method not in METHODS:
        raise ValueError("unknown method")
    sources, index = bundle.source_index(), bundle.record_index()
    timed = method != "static_kg"
    rules = method in {"simple_temporal_rules", "verified_temporal"}
    verified = method == "verified_temporal"
    certificates = []

    def admitted(records, history=False):
        if verified:
            if not records:
                return []
            ids, certificate = _verified_snapshot(bundle, query, records, history=history)
            certificates.append(certificate)
            return [r for r in records if r["record_id"] in ids]
        if timed:
            return [r for r in records if _available(r, sources[r["source_id"]], query, system=rules)]
        return list(records)

    if query["kind"] in {"status", "history"}:
        matched = _matching(bundle, query)
        selected = admitted(matched, history=query["kind"] == "history")
        controls = []
        if query["kind"] == "status":
            if not rules:
                superseded = {old for r in selected for old in r["supersedes"]}
                controls = [r["record_id"] for r in selected if r["supersedes"]]
                selected = [r for r in selected if r["record_id"] not in superseded]
            elif not verified:
                possible = selected
                selected = []
                at = time_value(query["valid_at"])
                for r in possible:
                    kind, start, end = _state_interval(r, possible)
                    if kind == "interval" and time_value(start) <= at and (end is None or at < time_value(end)):
                        selected.append(r)
                controls = [r["record_id"] for r in possible if r["supersedes"]]
            else:
                controls = [r["record_id"] for r in matched if r["supersedes"] and _available(r, sources[r["source_id"]], query, system=True)]
        return _result(_summarize(selected), [r["record_id"] for r in selected] + controls, certificates=certificates)

    if query["kind"] == "event_before":
        r = index[query["record_id"]]
        selected = admitted([r], history=True)
        if not selected:
            return _result("UNKNOWN", certificates=certificates)
        if not rules:
            return _result("SUPPORTED", [r["record_id"]])
        lower, upper = date_bounds(r["event_date"])
        at = time_value(query["valid_at"])
        outcome = ("UNKNOWN" if lower is None else "SUPPORTED" if time_value(upper) <= at else
                   "REFUTED" if at < time_value(lower) else "UNKNOWN")
        return _result(outcome, [r["record_id"]], certificates=certificates)

    path = [index[i] for i in query["path"]]
    # Shared non-temporal compatibility rule: a drug-history path has one subject.
    if any(r["subject"] != query["subject"] for r in path):
        return _result("REJECTED_CONTEXT", [r["record_id"] for r in path], notes=("drug identity mismatch",))
    selected = []
    for r in path:
        if verified:
            step_query = {**query, "subject": r["subject"], "predicate": r["predicate"], "value": r["value"], "context": r["context"]}
            ids, certificate = _verified_snapshot(bundle, step_query, [r], history=True)
            certificates.append(certificate)
            if ids:
                selected.append(r)
        elif not timed or _available(r, sources[r["source_id"]], query, system=rules):
            selected.append(r)
    used = [r["record_id"] for r in selected]
    if len(selected) != len(path):
        return _result("UNKNOWN", used, certificates=certificates)
    if not rules:
        return _result("PATH_SUPPORTED", used)
    bounds = [date_bounds(r["event_date"]) for r in path]
    if any(lo is None or time_value(hi) > time_value(query["valid_at"]) for lo, hi in bounds):
        return _result("UNKNOWN", used, certificates=certificates)
    for (_, left_hi), (right_lo, _) in zip(bounds, bounds[1:]):
        if time_value(left_hi) >= time_value(right_lo):
            certainly_reversed = any(time_value(a[0]) >= time_value(b[1]) for a, b in zip(bounds, bounds[1:]))
            return _result("REJECTED_ORDER" if certainly_reversed else "UNKNOWN", used, certificates=certificates)
    return _result("PATH_SUPPORTED", used, certificates=certificates, notes=("strict precedence only; no causal entailment",))


def kg_metrics(bundle: Benchmark) -> dict:
    """Declared, versioned metrics for a local reified assertion/source schema."""
    sources = {r["source_id"] for r in bundle.records}
    # Exact logical view: two classes, two object properties, nine literal properties.
    triples = set()
    for r in bundle.records:
        rid = "assertion:" + r["record_id"]
        triples.add((rid, "rdf:type", "ic:DatedAssertion"))
        triples.add((rid, "ic:source", "source:" + r["source_id"]))
        for p in ("subject", "predicate", "value", "context", "polarity", "event_date"):
            if r[p] is not None:
                triples.add((rid, "ic:" + p, json.dumps(r[p])))
        for old in r["supersedes"]:
            triples.add((rid, "ic:supersedes", "assertion:" + old))
    for s in bundle.sources:
        if s["source_id"] not in sources:
            continue
        sid = "source:" + s["source_id"]
        triples.add((sid, "rdf:type", "ic:SourceDocument"))
        for p in ("published_date", "retrieved_at", "sha256"):
            if s[p] is not None:
                triples.add((sid, "ic:" + p, json.dumps(s[p])))
    object_properties = {"ic:source", "ic:supersedes"}
    literal_properties = {"ic:" + p for p in ("subject", "predicate", "value", "context", "polarity", "event_date", "published_date", "retrieved_at", "sha256")}
    used_properties = {p for _, p, _ in triples} - {"rdf:type"}
    return {"definition_version": "reified-two-class-view-v1", "asserted_data_triples": len(triples), "inferred_triples": 0,
            "distinct_predicates_including_rdf_type": len({p for _, p, _ in triples}),
            "relationship_richness": {"value": 1.0, "object_properties": 2, "subclass_relationships": 0,
                                      "formula": "2 / (2 + 0); descriptive local schema ratio, not quality"},
            "attribute_richness": {"value": 4.5, "formula": "9 declared literal properties / 2 declared classes"},
            "class_instantiation_ratio": {"value": 1.0 if bundle.records else None, "formula": "populated declared classes / 2"},
            "property_instantiation_ratio": {"value": len(used_properties) / len(object_properties | literal_properties),
                                              "formula": "used declared properties / 11"},
            "source_interlinking_completeness": {"value": sum(r["source_id"] in bundle.source_index() for r in bundle.records) / len(bundle.records) if bundle.records else None,
                                                 "formula": "assertions with resolved source identity / all assertions"},
            "logical_inconsistency": "NOT_EVALUATED on this regulatory schema",
            "unsatisfiable_classes": "NOT_EVALUATED", "shacl_conformance": "NOT_EVALUATED on this regulatory schema",
            "limitations": "GO mapping/OWL tests from another fixture are not credited as checks of this graph."}


def integrity_demo(bundle: Benchmark) -> dict:
    """Public content/hash/anchor/date diagnostics only; never execute the four methods."""
    validate_bundle(bundle)
    return {"scope": PURPOSE + "; public corpus integrity only, no method evaluation", "freeze_sha256": bundle.freeze_sha256,
            "records": len(bundle.records), "queries": len(bundle.queries), "sources_used": len({r["source_id"] for r in bundle.records}),
            "exact_event_times_known": sum(r["event_time"] is not None for r in bundle.records),
            "event_dates_unknown": [r["record_id"] for r in bundle.records if r["event_date"] is None],
            "supersessions": [{"record_id": r["record_id"], "supersedes": r["supersedes"]} for r in bundle.records if r["supersedes"]],
            "gold_not_executed": True, "nih_eligibility": "NIH_RESOURCE_ELIGIBILITY_UNRESOLVED", "kg_metrics": kg_metrics(bundle)}


def evaluate_benchmark(bundle: Benchmark, *, purpose: str) -> dict:
    if purpose != PURPOSE:
        raise ValueError("NIH-prize empirical evidence is not cleared; only explicit shared-temporal-method-development is supported")
    validate_bundle(bundle)
    started = clock.monotonic()
    gold = {g["query_id"]: g for g in bundle.gold}
    records, sources = bundle.record_index(), bundle.source_index()
    rows, summaries = [], {}
    for method in METHODS:
        answers = []
        for query in bundle.queries:
            answer = run_method(bundle, query, method)
            truth = gold[query["query_id"]]
            used = answer["used_record_ids"]
            future = [i for i in used if date_bounds(sources[records[i]["source_id"]]["published_date"])[1] is None
                      or time_value(date_bounds(sources[records[i]["source_id"]]["published_date"])[1]) > time_value(query["known_as_of"])]
            row = {"method": method, "query_id": query["query_id"], "kind": query["kind"], "expected": truth["expected"],
                   **answer, "correct": answer["outcome"] == truth["expected"],
                   "future_evidence_accepted": bool(future) and answer["outcome"] not in {"UNKNOWN", "REJECTED_CONTEXT", "REJECTED_ORDER"},
                   "future_record_ids": future,
                   "event_order_violation": query["kind"] == "path" and answer["outcome"] == "PATH_SUPPORTED" and truth["expected"] == "REJECTED_ORDER",
                   "source_provenance_correct": all(i in records and records[i]["source_id"] in sources and records[i]["source_sha256"] == sources[records[i]["source_id"]]["sha256"] for i in used)}
            answers.append(row)
        def fraction(n, d): return {"numerator": n, "denominator": d, "value": n/d if d else None}
        path_gold = [a for a in answers if a["kind"] == "path" and a["expected"] == "PATH_SUPPORTED"]
        conflict_gold = [a for a in answers if a["expected"] == "CONFLICT"]
        changed = [a for a in answers if a["query_id"] in {"hcq_july_revoked", "hcq_retrospective_april"}]
        certificates = [c for a in answers for c in a["certificates"]]
        summaries[method] = {
            "correctness": fraction(sum(a["correct"] for a in answers), len(answers)),
            "future_evidence_acceptance": fraction(sum(a["future_evidence_accepted"] for a in answers), len(answers)),
            "event_order_violations": fraction(sum(a["event_order_violation"] for a in answers), sum(a["kind"] == "path" for a in answers)),
            "valid_path_coverage": fraction(sum(a["outcome"] == "PATH_SUPPORTED" for a in path_gold), len(path_gold)),
            "recorded_opposition_accuracy": fraction(sum(a["correct"] for a in conflict_gold), len(conflict_gold)),
            "supersession_resolution": fraction(sum(a["correct"] for a in changed), len(changed)),
            "source_provenance_correctness": fraction(sum(a["source_provenance_correct"] for a in answers if a["used_record_ids"]), sum(bool(a["used_record_ids"]) for a in answers)),
            "source_citation_precision": fraction(sum(i in records and records[i]["source_id"] in sources and records[i]["source_sha256"] == sources[records[i]["source_id"]]["sha256"] for a in answers for i in a["used_record_ids"]), sum(len(a["used_record_ids"]) for a in answers)),
            "unsupported_path_acceptance": fraction(sum(a["outcome"] == "PATH_SUPPORTED" and a["expected"] != "PATH_SUPPORTED" for a in answers), sum(a["outcome"] == "PATH_SUPPORTED" for a in answers)),
            "certificate_replay": fraction(sum(c["replay_valid"] for c in certificates), len(certificates)),
            "nonabstention_coverage": fraction(sum(a["outcome"] != "UNKNOWN" for a in answers), len(answers)),
            "unresolved_scientific_conflict": "NOT_EVALUATED; retained documentary opposition is a different endpoint"}
        rows.extend(answers)
    return {"version": VERSION, "scope": PURPOSE, "nih_prize_eligible": False,
            "nih_eligibility": "NIH_RESOURCE_ELIGIBILITY_UNRESOLVED", "freeze_sha256": bundle.freeze_sha256,
            "methods": summaries, "rows": rows, "kg_metrics": kg_metrics(bundle), "wall_seconds": clock.monotonic() - started,
            "interpretation": "Small source-selected retrospective administrative/evidence replay. No biological efficacy, representative benchmark, blinded expert gold, full OWL, or verified-over-simple-rule improvement is presumed."}

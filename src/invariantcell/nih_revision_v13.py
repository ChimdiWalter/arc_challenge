"""Matched documentary revision checks. Gold is never a solver input.

The predicate concerns editorial acceptance of one revision, not biological
truth or the continuing clinical validity of an evidence item.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .core import canonical_hash, time_value
from .temporal import EvidenceRecord, Snapshot, TemporalQuery, issue_certificate
from .replay import verify_certificate

METHODS = ("static_kg", "timestamp_filter", "simple_temporal_rules", "verified_temporal")


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def available(record, query):
    return (record.public_available_at is not None
            and time_value(record.public_available_at) <= time_value(query.evidence_cutoff)
            and record.ingested_at is not None
            and time_value(record.ingested_at) <= time_value(query.ingestion_cutoff))


def interval_valid(record, query):
    return (record.validity_kind == "interval"
            and (record.valid_from is None or time_value(record.valid_from) <= time_value(query.valid_at))
            and (record.valid_to is None or time_value(query.valid_at) < time_value(record.valid_to)))


def event_valid(record, query):
    return query.event_cutoff is None or (record.event_time is not None
            and time_value(record.event_time) <= time_value(query.event_cutoff))


def outcome(records, predicate="editorially_accepted_revision"):
    if predicate == "accepted_replacement_value":
        values = sorted({r.value for r in records if r.polarity == "supports"})
        return "ABSTAIN" if not values else "VALUE:" + values[0] if len(values) == 1 else "MULTIPLE_VALUES"
    signs = {r.polarity for r in records}
    return ("CONFLICT" if len(signs) == 2 else "ACCEPTED" if signs == {"supports"}
            else "NOT_ACCEPTED" if signs else "ABSTAIN")


def solve(records, query_record, method):
    """Only typed assertions and a query enter any method; no labels/metadata."""
    if method not in METHODS:
        raise ValueError("unknown comparator")
    query = TemporalQuery(**query_record["query"])
    snapshot = Snapshot("v13:" + query_record["id"], tuple(EvidenceRecord(**r) for r in records))
    selected = [r for r in snapshot.evidence if (r.subject, r.predicate, r.context) ==
                (query.subject, query.predicate, query.context)]
    certificate = None
    if method == "verified_temporal":
        certificate = issue_certificate(snapshot, query)
        replay = verify_certificate(certificate, snapshot, canonical_hash(snapshot.to_dict()), query)
        if not replay.valid:
            raise ValueError("independent certificate replay failed")
        used = {i for a in certificate["result"]["answers"] for i in (*a["support_ids"], *a["refute_ids"])}
        selected = [r for r in snapshot.evidence if r.evidence_id in used]
    else:
        if method != "static_kg":
            selected = [r for r in selected if available(r, query)]
        if method == "simple_temporal_rules":
            selected = [r for r in selected if interval_valid(r, query) and event_valid(r, query)]
    return {"outcome": outcome(selected, query.predicate), "used_ids": sorted(r.evidence_id for r in selected),
            "certificate": certificate, "certificate_replay_valid": True if certificate else None}


def validate_corpus(corpus):
    """Shape/source-projection checks, not benchmark answer evaluation."""
    ids = [r["evidence_id"] for r in corpus["records"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate assertion")
    for row in corpus["records"]:
        record = EvidenceRecord(**row)
        if record.dependencies or record.state_kind != "observed" or record.synthetic:
            raise ValueError("v13 profile only supports observed documentary records without models")
        if record.predicate not in {"editorially_accepted_revision", "accepted_replacement_value"}:
            raise ValueError("out-of-scope predicate")
    qids = [q["id"] for q in corpus["queries"]]
    if len(qids) != len(set(qids)):
        raise ValueError("duplicate query")
    for q in corpus["queries"]:
        TemporalQuery(**q["query"])
        if q["split"] not in {"development", "validation", "final_internal_test"}:
            raise ValueError("invalid split")
    return {"records": len(ids), "queries": len(qids), "source_projection_only": True}


def verify_freeze(root, freeze_path):
    root = Path(root).resolve()
    freeze = json.loads(Path(freeze_path).read_text())
    for name, expected in freeze["files"].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or file_sha(path) != expected:
            raise ValueError("frozen artifact mismatch: " + name)
    if freeze["empirical_outputs_examined"] or freeze["eligibility_established"]:
        raise ValueError("invalid development/eligibility declaration")
    return freeze


def evaluate_predictions(corpus, predictions, gold):
    """Evaluator-only: all methods' predictions must already be materialized."""
    if set(gold) != {q["id"] for q in corpus["queries"]}:
        raise ValueError("gold/query mismatch")
    by_id = {r["evidence_id"]: EvidenceRecord(**r) for r in corpus["records"]}
    output = []
    for method in METHODS:
        for split in ("development", "validation", "final_internal_test"):
            qs = [q for q in corpus["queries"] if q["split"] == split]
            n = len(qs)
            counts = dict(correct=0, unsupported_availability_uses=0, invalid_interval_uses=0,
                          event_order_violations=0, conflict_outputs=0, provenance_missing=0,
                          correct_abstentions=0, gold_abstentions=0, required_support_hits=0,
                          required_support_total=0, certificates_replayed=0)
            for q in qs:
                p, g = predictions[method][q["id"]], gold[q["id"]]
                query = TemporalQuery(**q["query"])
                counts["correct"] += p["outcome"] == g["outcome"]
                counts["conflict_outputs"] += p["outcome"] == "CONFLICT"
                counts["gold_abstentions"] += g["outcome"] == "ABSTAIN"
                counts["correct_abstentions"] += p["outcome"] == g["outcome"] == "ABSTAIN"
                counts["certificates_replayed"] += p["certificate_replay_valid"] is True
                counts["required_support_hits"] += len(set(p["used_ids"]) & set(g["support_ids"]))
                counts["required_support_total"] += len(g["support_ids"])
                for rid in p["used_ids"]:
                    record = by_id[rid]
                    counts["unsupported_availability_uses"] += not available(record, query)
                    counts["invalid_interval_uses"] += not interval_valid(record, query)
                    counts["event_order_violations"] += not event_valid(record, query)
                    counts["provenance_missing"] += rid not in corpus["anchors"]
            output.append({"method": method, "split": split, "queries": n, **counts,
                           "accuracy": counts["correct"] / n if n else None,
                           "future_publication_leakage": None,
                           "future_publication_leakage_status": "NOT_IDENTIFIABLE_FIRST_PUBLIC_TIME_UNKNOWN",
                           "path_coverage": None, "real_unresolved_conflict_accuracy": None,
                           "event_order_scope": "curation event cutoff, not biological sequence",
                           "conflict_scope": "incompatible editorial acceptance assertions across time"})
    return output


def validate_anchors(root, corpus):
    """Independent JSON-pointer and copied-field check; never reads gold."""
    cache = {}
    for record in corpus["records"]:
        a = corpus["anchors"][record["evidence_id"]]
        path = Path(root) / a["source_path"]
        if not path.resolve().is_relative_to(Path(root).resolve()) or path.is_symlink():
            raise ValueError("source boundary")
        if file_sha(path) != record["source_sha256"] or record["source_sha256"] != a["source_sha256"]:
            raise ValueError("source digest mismatch")
        if str(path) not in cache:
            cache[str(path)] = json.loads(path.read_text())
        raw = cache[str(path)]
        for key in a["json_pointer"].lstrip("/").split("/"):
            raw = raw[int(key)] if isinstance(raw, list) else raw[key]
        if (raw["currentValue"] != a["stored_prior_value"] or raw["suggestedValue"] != a["proposed_value"]
                or raw["fieldName"] != a["field_name"] or raw["createdAt"] != a["created_at"]
                or raw["status"] != a["revision_status"] or record["subject"] != "RID" + str(raw["id"])):
            raise ValueError("source projection mismatch")
        if record["predicate"] == "accepted_replacement_value":
            expected = json.dumps(raw["suggestedValue"], separators=(",", ":"), sort_keys=True, ensure_ascii=False)
            if raw["status"] != "ACCEPTED" or record["value"] != expected:
                raise ValueError("unsupported accepted replacement")
    return len(corpus["records"])

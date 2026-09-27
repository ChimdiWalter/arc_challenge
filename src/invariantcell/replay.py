"""Independent certificate replay against caller-supplied trust anchors.

This module deliberately does not call temporal.evaluate or issue_certificate.
Both the intended query and snapshot hash must come from outside a certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from .core import canonical_hash, time_value
from .temporal import ALGORITHM, LIMITATIONS, Snapshot, TemporalQuery


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"valid": self.valid, "errors": list(self.errors),
                "scope": "deterministic_replay_against_supplied_trust_anchors"}


def _reconstruct(snapshot: Snapshot, query: TemporalQuery) -> dict:
    """Separate implementation of the small temporal selection specification."""
    public_limit = time_value(query.evidence_cutoff)
    ingestion_limit = time_value(query.ingestion_cutoff)
    valid_instant = time_value(query.valid_at)
    by_id = {dep.dependency_id: dep for dep in snapshot.dependencies}
    model_failures = [dep.dependency_id for dep in snapshot.dependencies
                      if dep.kind == "model" and (dep.available_at is None or time_value(dep.available_at) > public_limit)]
    evidence_for_value = {}
    excluded = []
    for record in snapshot.evidence:
        failed = []
        if record.subject != query.subject or record.predicate != query.predicate or record.context != query.context:
            failed.append("query_or_context_mismatch")
        if record.public_available_at is None or time_value(record.public_available_at) > public_limit:
            failed.append("public_evidence_after_cutoff_or_unknown")
        if record.ingested_at is None or time_value(record.ingested_at) > ingestion_limit:
            failed.append("ingestion_after_cutoff_or_unknown")
        if record.validity_kind == "unknown":
            failed.append("unknown_validity")
        if record.valid_from is not None and valid_instant < time_value(record.valid_from):
            failed.append("before_valid_interval")
        if record.valid_to is not None and valid_instant >= time_value(record.valid_to):
            failed.append("after_valid_interval")
        if query.event_cutoff is not None:
            if record.event_time is None or time_value(record.event_time) > time_value(query.event_cutoff):
                failed.append("event_after_cutoff_or_unknown")
        if record.state_kind == "predicted" and query.allow_predictions is False:
            failed.append("prediction_not_observation")
        for dep_id in record.dependencies:
            dependency = by_id[dep_id]
            if dependency.available_at is None or time_value(dependency.available_at) > public_limit:
                failed.append("dependency_after_cutoff_or_unknown")
                break
        if model_failures:
            failed.append("snapshot_model_cutoff_violation")
        if failed:
            excluded.append({"evidence_id": record.evidence_id, "reasons": failed})
            continue
        evidence_for_value.setdefault(record.value, []).append(record)
    answers = []
    for value in sorted(evidence_for_value):
        supports = [record.evidence_id for record in evidence_for_value[value] if record.polarity == "supports"]
        refutes = [record.evidence_id for record in evidence_for_value[value] if record.polarity == "refutes"]
        support_kinds = sorted({record.state_kind for record in evidence_for_value[value] if record.polarity == "supports"})
        refute_kinds = sorted({record.state_kind for record in evidence_for_value[value] if record.polarity == "refutes"})
        status = "conflicted"
        if not refutes:
            status = "observed_support" if "observed" in support_kinds else "prediction_with_valid_provenance"
        elif not supports:
            status = "observed_refutation" if "observed" in refute_kinds else "prediction_refutation_with_valid_provenance"
        answers.append({"value": value, "status": status, "support_ids": supports, "refute_ids": refutes,
                        "support_state_kinds": support_kinds, "refute_state_kinds": refute_kinds})
    outcome = "abstain_unsupported"
    if model_failures:
        outcome = "rejected_dependency"
    elif any(answer["status"] == "conflicted" for answer in answers):
        outcome = "conflict"
    elif any(answer["status"] == "observed_support" for answer in answers):
        outcome = "observed_support"
    elif any(answer["status"] == "prediction_with_valid_provenance" for answer in answers):
        outcome = "prediction_with_valid_provenance"
    elif any("dependency_after_cutoff_or_unknown" in item["reasons"] for item in excluded):
        outcome = "rejected_dependency"
    return {"outcome": outcome, "answers": answers, "excluded": excluded, "model_cutoff_violations": model_failures,
            "checks_passed": not model_failures, "algorithm": ALGORITHM, "limitations": list(LIMITATIONS)}


def verify_certificate(certificate: dict, trusted_snapshot: Snapshot, expected_snapshot_sha256: str,
                       expected_query: TemporalQuery) -> VerificationResult:
    """Require external snapshot and intended-query anchors; internal hashes alone are insufficient."""
    errors = []
    required = {"schema_version", "snapshot", "snapshot_sha256", "dependencies_sha256", "query", "query_sha256",
                "result", "certificate_sha256"}
    if not isinstance(certificate, dict) or set(certificate) != required:
        return VerificationResult(False, ("invalid_certificate_schema",))
    try:
        trusted = trusted_snapshot.to_dict()
        if certificate["schema_version"] != 1 or type(certificate["schema_version"]) is not int:
            errors.append("unsupported_schema_version")
        if canonical_hash(trusted) != expected_snapshot_sha256:
            errors.append("trusted_snapshot_does_not_match_external_hash")
        if canonical_hash(certificate["snapshot"]) != expected_snapshot_sha256 or certificate["snapshot_sha256"] != expected_snapshot_sha256:
            errors.append("snapshot_trust_anchor_mismatch")
        if canonical_hash(certificate["snapshot"]) != canonical_hash(trusted):
            errors.append("supplied_snapshot_mismatch")
        if certificate["dependencies_sha256"] != canonical_hash(trusted["dependencies"]):
            errors.append("dependency_hash_mismatch")
        if canonical_hash(certificate["query"]) != canonical_hash(expected_query.to_dict()):
            errors.append("intended_query_mismatch")
        if certificate["query_sha256"] != canonical_hash(certificate["query"]):
            errors.append("query_hash_mismatch")
        payload = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
        if certificate["certificate_sha256"] != canonical_hash(payload):
            errors.append("certificate_hash_mismatch")
        expected = _reconstruct(trusted_snapshot, expected_query)
        if canonical_hash(certificate["result"]) != canonical_hash(expected):
            errors.append("independent_replay_mismatch")
        if expected["model_cutoff_violations"]:
            errors.append("model_dependencies_violate_cutoff")
    except (ValueError, TypeError, KeyError, AttributeError):
        errors.append("malformed_certificate")
    return VerificationResult(not errors, tuple(errors))

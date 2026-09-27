"""Pure, bounded reliability-triage policies over a frozen prediction panel.

This module does no I/O. The caller must materialize/hash decisions before a
separate evaluator process opens outcomes. Schema restrictions detect undeclared
features, not semantic leakage hidden inside otherwise permitted identifiers.
Formal/evidence checks are not probabilities or evidence of biological benefit.
"""

from bisect import bisect_right
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
import math
import random


POLICIES = ("random", "native", "calibrated", "fixed_rules", "verification")
CANDIDATE_FIELDS = {
    "candidate_id", "context_id", "target_id", "intervention_protocol",
    "prediction_score", "model_hash", "prediction_hash", "native_uncertainty",
    "native_uncertainty_kind", "evidence_available_at", "evidence_context_id",
    "evidence_status", "formal_status", "evidence_snapshot_hash",
    "supporting_evidence_refs", "dependency_available_at",
}
PROTOCOL_FIELDS = {
    "objective", "objective_version", "budget", "seed", "knowledge_boundary",
    "strict_historical", "reliability_max_error", "calibration_bin_edges",
    "minimum_calibration_support", "fixed_rules", "risk_limit",
}


def digest(value):
    """Canonical JSON hash; nonfinite numbers and non-JSON values fail closed."""
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _copy(value):
    digest(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _keys(value, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown or missing:
        raise ValueError(f"Undeclared/indirect features {sorted(unknown)}; missing {sorted(missing)}")


def _number(value, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected finite numeric value")
    if nonnegative and value < 0:
        raise ValueError("Expected nonnegative value")
    return value


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty string")
    return value


def _time(value):
    parsed = datetime.fromisoformat(_text(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def _protocol(protocol):
    _keys(protocol, PROTOCOL_FIELDS, PROTOCOL_FIELDS)
    if protocol["objective"] != "reliability_triage":
        raise ValueError("Only reliability_triage is implemented; no biological-benefit endpoint")
    _text(protocol["objective_version"])
    for field in ("budget", "minimum_calibration_support"):
        if type(protocol[field]) is not int or protocol[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if type(protocol["seed"]) is not int or type(protocol["strict_historical"]) is not bool:
        raise ValueError("Invalid seed/historical flag")
    _time(protocol["knowledge_boundary"])
    _number(protocol["reliability_max_error"], nonnegative=True)
    edges = protocol["calibration_bin_edges"]
    if not isinstance(edges, list):
        raise ValueError("Calibration edges must be a list")
    for edge in edges:
        _number(edge, nonnegative=True)
    if edges != sorted(set(edges)):
        raise ValueError("Calibration edges must be strictly increasing")
    if protocol["risk_limit"] is not None and not 0 <= _number(protocol["risk_limit"]) <= 1:
        raise ValueError("risk_limit must be null or in [0,1]")
    rules = protocol["fixed_rules"]
    _keys(rules, {"max_native_uncertainty", "require_supported_evidence"},
          {"max_native_uncertainty", "require_supported_evidence"})
    if rules["max_native_uncertainty"] is not None:
        _number(rules["max_native_uncertainty"], nonnegative=True)
    if type(rules["require_supported_evidence"]) is not bool:
        raise ValueError("require_supported_evidence must be boolean")


def freeze_panel(candidates, protocol, model_snapshot):
    """Freeze predictions and admissible metadata; outcomes are never accepted.

    model_snapshot is {model_hash, dependencies:[{name,sha256,available_at}]}.
    Strict historical reconstruction requires dates for all model dependencies.
    Evidence supplied after the boundary is rejected even for policies ignoring it.
    """
    _protocol(protocol)
    _keys(model_snapshot, {"model_hash", "dependencies"}, {"model_hash", "dependencies"})
    _text(model_snapshot["model_hash"])
    if not isinstance(model_snapshot["dependencies"], list):
        raise ValueError("Model dependencies must be a list")
    boundary = _time(protocol["knowledge_boundary"])
    if protocol["strict_historical"] and not model_snapshot["dependencies"]:
        raise ValueError("Strict historical dependency inventory is missing")
    for dependency in model_snapshot["dependencies"]:
        _keys(dependency, {"name", "sha256", "available_at"}, {"name", "sha256", "available_at"})
        _text(dependency["name"])
        _text(dependency["sha256"])
        when = dependency["available_at"]
        if when is not None:
            _time(when)
        if protocol["strict_historical"] and (when is None or _time(when) > boundary):
            raise ValueError("Model dependency unavailable at historical boundary")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate universe must be a nonempty list")
    if protocol["budget"] > len(candidates):
        raise ValueError("Budget exceeds candidate universe")
    ids, units, uncertainty_kinds = set(), set(), set()
    for candidate in candidates:
        required = {"candidate_id", "context_id", "target_id", "intervention_protocol",
                    "prediction_score", "model_hash", "prediction_hash"}
        _keys(candidate, CANDIDATE_FIELDS, required)
        for field in required - {"prediction_score"}:
            _text(candidate[field])
        _number(candidate["prediction_score"])
        if candidate["model_hash"] != model_snapshot["model_hash"]:
            raise ValueError("Model hash mismatch")
        unit = tuple(candidate[x] for x in ("context_id", "target_id", "intervention_protocol"))
        if candidate["candidate_id"] in ids or unit in units:
            raise ValueError("Duplicate candidate or target/context/protocol experimental unit")
        ids.add(candidate["candidate_id"])
        units.add(unit)
        for field in ("native_uncertainty_kind", "evidence_context_id", "evidence_snapshot_hash"):
            if candidate.get(field) is not None:
                _text(candidate[field])
        uncertainty = candidate.get("native_uncertainty")
        if uncertainty is not None:
            _number(uncertainty, nonnegative=True)
            uncertainty_kinds.add(_text(candidate.get("native_uncertainty_kind")))
        for field in ("evidence_available_at", "dependency_available_at"):
            when = candidate.get(field)
            if when is not None and _time(when) > boundary:
                raise ValueError(f"{field} exceeds information boundary")
        status = candidate.get("evidence_status", "missing")
        if status not in {"supported", "conflicting", "missing"}:
            raise ValueError("Invalid evidence status")
        if candidate.get("formal_status", "unknown") not in {"passed", "failed", "unknown"}:
            raise ValueError("Invalid formal status")
        if status != "missing":
            for field in ("evidence_available_at", "evidence_context_id", "evidence_snapshot_hash"):
                _text(candidate.get(field))
        refs = candidate.get("supporting_evidence_refs", [])
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            raise ValueError("Evidence references must be strings")
    if len(uncertainty_kinds) > 1:
        raise ValueError("Incomparable native uncertainty kinds in one panel")
    body = {"schema_version": 1, "candidates": sorted(_copy(candidates), key=lambda c: c["candidate_id"]),
            "protocol_hash": digest(protocol), "model_snapshot": _copy(model_snapshot)}
    return {**body, "panel_hash": digest(body)}


def _panel(panel, protocol):
    rebuilt = freeze_panel(panel["candidates"], protocol, panel["model_snapshot"])
    if panel != rebuilt:
        raise ValueError("Frozen panel/protocol snapshot mismatch or tampering")


def fit_calibration(development_rows, evaluation_panel, protocol):
    """Fixed-bin empirical failure rates, only from disjoint development units.

    Each row: candidate_id, context_id, split='development', model_hash,
    native_uncertainty, native_uncertainty_kind, agreement_error, available_at.
    No threshold, bin boundary, or rule is fitted from evaluation outcomes.
    Empty/undersupported bins have unknown risk, never an invented probability.
    """
    _panel(evaluation_panel, protocol)
    eval_ids = {c["candidate_id"] for c in evaluation_panel["candidates"]}
    eval_contexts = {c["context_id"] for c in evaluation_panel["candidates"]}
    eval_kinds = {c.get("native_uncertainty_kind") for c in evaluation_panel["candidates"]
                  if c.get("native_uncertainty") is not None}
    fields = {"candidate_id", "context_id", "split", "model_hash", "native_uncertainty",
              "native_uncertainty_kind", "agreement_error", "available_at"}
    bins = [{"count": 0, "failures": 0} for _ in range(len(protocol["calibration_bin_edges"]) + 1)]
    seen, contexts, kinds = set(), set(), set()
    for row in development_rows:
        _keys(row, fields, fields)
        if row["split"] != "development":
            raise ValueError("Calibration accepts development split only")
        cid, ctx = _text(row["candidate_id"]), _text(row["context_id"])
        if cid in eval_ids or ctx in eval_contexts or cid in seen:
            raise ValueError("Calibration IDs/contexts overlap evaluation or duplicate development IDs")
        if row["model_hash"] != evaluation_panel["model_snapshot"]["model_hash"]:
            raise ValueError("Calibration model hash mismatch")
        if _time(row["available_at"]) > _time(protocol["knowledge_boundary"]):
            raise ValueError("Calibration label unavailable at information boundary")
        seen.add(cid)
        contexts.add(ctx)
        error = _number(row["agreement_error"], nonnegative=True)
        if row["native_uncertainty"] is not None:
            uncertainty = _number(row["native_uncertainty"], nonnegative=True)
            kinds.add(_text(row["native_uncertainty_kind"]))
            bucket = bins[bisect_right(protocol["calibration_bin_edges"], uncertainty)]
            bucket["count"] += 1
            bucket["failures"] += int(error > protocol["reliability_max_error"])
    if len(kinds | eval_kinds) > 1:
        raise ValueError("Calibration uncertainty kind mismatch")
    for bucket in bins:
        n = bucket["count"]
        bucket["risk"] = bucket["failures"] / n if n >= protocol["minimum_calibration_support"] else None
    body = {"schema_version": 1, "method": "fixed_bin_development_failure_rate",
            "protocol_hash": digest(protocol), "model_hash": evaluation_panel["model_snapshot"]["model_hash"],
            "evaluation_panel_hash": evaluation_panel["panel_hash"], "bins": bins,
            "development_hash": digest(development_rows), "development_ids": sorted(seen),
            "development_contexts": sorted(contexts), "uncertainty_kind": next(iter(kinds), None),
            "limitations": "Empirical development-bin risk; no guaranteed context transfer or biological benefit."}
    return {**body, "calibration_hash": digest(body)}


def _calibration(calibration, panel, protocol):
    if calibration is None:
        return
    fields = {"schema_version", "method", "protocol_hash", "model_hash", "evaluation_panel_hash",
              "bins", "development_hash", "development_ids", "development_contexts",
              "uncertainty_kind", "limitations", "calibration_hash"}
    _keys(calibration, fields, fields)
    if calibration["schema_version"] != 1 or calibration["method"] != "fixed_bin_development_failure_rate":
        raise ValueError("Unsupported calibration method/version")
    for field in ("development_ids", "development_contexts"):
        if not isinstance(calibration[field], list):
            raise ValueError("Calibration IDs/contexts must be lists")
        for value in calibration[field]:
            _text(value)
    known_kinds = {c.get("native_uncertainty_kind") for c in panel["candidates"] if c.get("native_uncertainty") is not None}
    if calibration["uncertainty_kind"] is not None and known_kinds and known_kinds != {calibration["uncertainty_kind"]}:
        raise ValueError("Calibration uncertainty kind mismatch")
    body = {key: value for key, value in calibration.items() if key != "calibration_hash"}
    if digest(body) != calibration.get("calibration_hash"):
        raise ValueError("Calibration tampering")
    if (calibration["evaluation_panel_hash"] != panel["panel_hash"] or
            calibration["protocol_hash"] != digest(protocol) or
            calibration["model_hash"] != panel["model_snapshot"]["model_hash"]):
        raise ValueError("Calibration snapshot mismatch")
    if (set(calibration["development_ids"]) & {c["candidate_id"] for c in panel["candidates"]} or
            set(calibration["development_contexts"]) & {c["context_id"] for c in panel["candidates"]}):
        raise ValueError("Calibration/evaluation overlap")
    if len(calibration["bins"]) != len(protocol["calibration_bin_edges"]) + 1:
        raise ValueError("Calibration bin count mismatch")
    for bucket in calibration["bins"]:
        _keys(bucket, {"count", "failures", "risk"}, {"count", "failures", "risk"})
        n, failures = bucket["count"], bucket["failures"]
        if type(n) is not int or type(failures) is not int or not 0 <= failures <= n:
            raise ValueError("Invalid calibration support")
        expected = failures / n if n >= protocol["minimum_calibration_support"] else None
        if bucket["risk"] is not None:
            _number(bucket["risk"], nonnegative=True)
        if bucket["risk"] != expected:
            raise ValueError("Calibration risk disagrees with development counts")


def select(panel, protocol, policy, calibration=None):
    """Select without file/network access. All policies share one fixed slot budget.

    Native reliability ranks lower native uncertainty first. Missing uncertainty
    falls back to unscored ID order. Fixed rules and verification use identical
    metadata access; verification adds calibrated risk when supported. Neither
    backfills ineligible candidates. A null risk always remains explicitly unknown.
    """
    _panel(panel, protocol)
    _calibration(calibration, panel, protocol)
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    candidates = panel["candidates"]
    random_order = list(range(len(candidates)))
    random.Random(protocol["seed"]).shuffle(random_order)
    random_ranks = {index: rank for rank, index in enumerate(random_order)}
    ranked, records = [], []
    for index, candidate in enumerate(candidates):
        uncertainty, risk = candidate.get("native_uncertainty"), None
        if calibration is not None and uncertainty is not None:
            risk = calibration["bins"][bisect_right(protocol["calibration_bin_edges"], uncertainty)]["risk"]
        reasons, hard, deferred = [], [], []
        if policy in {"fixed_rules", "verification"}:
            if candidate.get("formal_status", "unknown") == "failed":
                hard.append("FORMAL_CHECK_FAILED")
            if candidate.get("evidence_context_id") not in (None, candidate["context_id"]):
                hard.append("EVIDENCE_CONTEXT_MISMATCH")
            if candidate.get("evidence_status") == "conflicting":
                hard.append("UNRESOLVED_EVIDENCE_CONFLICT")
            if protocol["fixed_rules"]["require_supported_evidence"] and candidate.get("evidence_status") != "supported":
                deferred.append("SUPPORTED_EVIDENCE_REQUIRED")
            limit = protocol["fixed_rules"]["max_native_uncertainty"]
            if limit is not None and (uncertainty is None or uncertainty > limit):
                deferred.append("NATIVE_UNCERTAINTY_RULE_NOT_MET")
        if policy in {"calibrated", "verification"}:
            if risk is None:
                deferred.append("CALIBRATED_RISK_UNKNOWN")
            elif protocol["risk_limit"] is not None and risk > protocol["risk_limit"]:
                deferred.append("CALIBRATED_RISK_ABOVE_LIMIT")
        if policy == "verification" and candidate.get("formal_status", "unknown") != "passed":
            deferred.append("FORMAL_CHECK_UNVERIFIED")
        if policy == "random":
            score, meaning, key = None, "prespecified_seed_random_order", (random_ranks[index],)
        elif policy in {"calibrated", "verification"}:
            score, meaning, key = risk, "development_failure_rate_lower_first", (risk is None, risk or 0, uncertainty is None, uncertainty or 0)
        else:
            score, meaning, key = uncertainty, "native_uncertainty_lower_first_else_unscored_id_order", (uncertainty is None, uncertainty or 0)
            if uncertainty is None:
                reasons.append("NO_NATIVE_UNCERTAINTY_UNSCORED_ID_FALLBACK")
        record = {"candidate_id": candidate["candidate_id"], "context_id": candidate["context_id"],
                  "target_id": candidate["target_id"], "intervention_protocol": candidate["intervention_protocol"],
                  "risk_estimate": risk if policy in {"calibrated", "verification"} else None,
                  "rank_score": score, "score_meaning": meaning, "formal_applicability_status": candidate.get("formal_status", "unknown"),
                  "disposition": "INELIGIBLE_UNDER_PROTOCOL" if hard else "DEFER",
                  "reason_codes": hard + deferred + reasons, "supporting_evidence_refs": candidate.get("supporting_evidence_refs", []),
                  "evidence_snapshot_hash": candidate.get("evidence_snapshot_hash"), "rank": None}
        records.append(record)
        if not hard and not deferred:
            ranked.append((key + (candidate["candidate_id"],), index))
    selected = []
    for rank, (_, index) in enumerate(sorted(ranked), 1):
        record = records[index]
        record["rank"] = rank
        if rank <= protocol["budget"]:
            record["disposition"] = "SELECT"
            record["reason_codes"].append("WITHIN_FIXED_BUDGET")
            selected.append(record["candidate_id"])
        else:
            record["reason_codes"].append("BUDGET_EXHAUSTED")
    body = {"schema_version": 1, "policy": policy, "policy_version": "1",
            "panel_hash": panel["panel_hash"], "protocol_hash": digest(protocol),
            "model_hash": panel["model_snapshot"]["model_hash"], "budget": protocol["budget"],
            "calibration_hash": calibration["calibration_hash"] if calibration else None,
            "records": records, "selected_ids": selected,
            "resource_cost": {"candidate_visits": len(candidates), "external_tool_calls": 0,
                              "model_inference_calls": 0, "wall_seconds": None},
            "limitations": ["Formal checks do not establish biological truth.",
                            "Risk is empirical and may fail under context shift.",
                            "No biological-benefit outcome is defined.",
                            "Unknown native uncertainty uses an unscored ID fallback."]}
    return {**body, "decision_hash": digest(body)}


def verify_decisions(decisions, trusted_panel, protocol, calibration=None):
    """Replay against trusted snapshots; a self-consistent rewritten hash is insufficient."""
    replay_calibration = calibration if decisions.get("calibration_hash") is not None else None
    expected = select(trusted_panel, protocol, decisions["policy"], replay_calibration)
    if decisions != expected:
        raise ValueError("Decision replay/tamper verification failed")
    return True


def evaluate(decisions_by_policy, trusted_panel, protocol, outcomes, calibration=None):
    """Join outcomes only after replaying all decisions. Missing outcomes are errors.

    outcomes: list of {candidate_id, agreement_error}; error is a predeclared,
    independently computed withheld-measurement statistic, not an Arc score.
    Confidence intervals cluster paired policy differences by context, never cells.
    Fewer than five contexts returns no interval and an explicit limitation.
    """
    if not decisions_by_policy:
        raise ValueError("No frozen decisions supplied")
    for policy, decision in decisions_by_policy.items():
        if policy != decision["policy"]:
            raise ValueError("Policy name mismatch")
        verify_decisions(decision, trusted_panel, protocol, calibration)
    errors = {}
    for outcome in outcomes:
        _keys(outcome, {"candidate_id", "agreement_error"}, {"candidate_id", "agreement_error"})
        cid = _text(outcome["candidate_id"])
        if cid in errors:
            raise ValueError("Duplicate outcome")
        errors[cid] = _number(outcome["agreement_error"], nonnegative=True)
    candidates = {c["candidate_id"]: c for c in trusted_panel["candidates"]}
    if set(errors) != set(candidates):
        raise ValueError("Missing/mismatched outcomes: utility is not identifiable")
    reliable = {cid: error <= protocol["reliability_max_error"] for cid, error in errors.items()}
    contexts = sorted({c["context_id"] for c in candidates.values()})
    budget, n = protocol["budget"], len(candidates)
    reports, per_context_success = {}, {}
    for policy, decision in decisions_by_policy.items():
        chosen = decision["selected_ids"]
        successes = sum(reliable[cid] for cid in chosen)
        report = {"reliable_yield_at_budget": successes / budget, "selected_precision": successes / len(chosen) if chosen else None,
                  "coverage": len(chosen) / n, "unused_capacity": budget - len(chosen), "selected_count": len(chosen),
                  "mean_agreement_error": sum(errors[cid] for cid in chosen) / len(chosen) if chosen else None,
                  "biological_benefit": None, "per_context": {}, "prefix_quality_coverage_curve": [],
                  "resource_cost": decision["resource_cost"]}
        per_context_success[policy] = {}
        for context in contexts:
            universe = {cid for cid, candidate in candidates.items() if candidate["context_id"] == context}
            subset = [cid for cid in chosen if cid in universe]
            good = sum(reliable[cid] for cid in subset)
            per_context_success[policy][context] = good
            report["per_context"][context] = {"selected_count": len(subset), "reliable_count": good,
                "coverage": len(subset) / len(universe), "selected_precision": good / len(subset) if subset else None,
                "contribution_to_total_yield": good / budget}
        for slots in range(1, budget + 1):
            prefix = chosen[:slots]
            good = sum(reliable[cid] for cid in prefix)
            report["prefix_quality_coverage_curve"].append({"slots": slots, "yield": good / slots,
                "precision": good / len(prefix) if prefix else None, "coverage": len(prefix) / n})
        reports[policy] = report
    paired = []
    for left, right in combinations(sorted(reports), 2):
        differences = [per_context_success[right][ctx] - per_context_success[left][ctx] for ctx in contexts]
        interval = None
        if len(contexts) >= 5:
            rng = random.Random(protocol["seed"])
            draws = sorted(sum(rng.choices(differences, k=len(contexts))) / budget for _ in range(1000))
            interval = [draws[24], draws[974]]
        paired.append({"comparison": f"{right} minus {left}", "yield_difference": sum(differences) / budget,
                       "interval_95": interval, "cluster_unit": "context", "cluster_count": len(contexts),
                       "method": "paired_context_cluster_percentile_bootstrap_1000" if interval else "not_estimated_fewer_than_five_contexts"})
    return {"panel_hash": trusted_panel["panel_hash"], "protocol_hash": digest(protocol),
            "decision_hashes": {key: value["decision_hash"] for key, value in decisions_by_policy.items()},
            "outcomes_hash": digest(outcomes), "experimental_unit_count": n, "context_count": len(contexts),
            "policies": reports, "paired_comparisons": paired,
            "interpretation": "Retrospective prediction reliability only; not biological benefit, clinical efficacy, or experiments saved.",
            "uncertainty_limitations": "No cell pseudoreplication. Few contexts limit generalization; any interval assumes independent exchangeable contexts."}

"""Bounded synthetic bridge; selection subprocesses never receive outcomes."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .core import canonical_hash
from .decision import POLICIES, freeze_panel, fit_calibration, evaluate, verify_decisions
from .io import file_hash, utc_now, write_json


def protocol():
    # A fixture criterion, fixed before generating data; no real biological claim.
    return {"objective": "reliability_triage", "objective_version": "SYNTHETIC-software-smoke-v1",
            "budget": 3, "seed": 17, "knowledge_boundary": "2026-01-01T00:00:00Z",
            "strict_historical": False, "reliability_max_error": 0.5,
            "calibration_bin_edges": [], "minimum_calibration_support": 2,
            "fixed_rules": {"max_native_uncertainty": None, "require_supported_evidence": False},
            "risk_limit": 0.5}


def agreement(prediction, truth):
    """Independent candidate-level mean-expression MSE, never a VCC score."""
    if prediction["genes"] != truth["genes"] or len(prediction["counts"][0]) != len(truth["counts"][0]):
        raise ValueError("outcome gene order mismatch")
    a, b = prediction["counts"], truth["counts"]
    return sum((sum(row[g] for row in a) / len(a) - sum(row[g] for row in b) / len(b)) ** 2
               for g in range(len(a[0]))) / len(a[0])


def run_bridge(project, prefix, training, predictions, observations, references, cert_refs):
    project = Path(project)
    stamp = "2026-01-01T00:00:00Z"
    settings = json.loads((project / prefix / "decision_protocol.json").read_text())
    contexts = sorted({r.context for r in predictions})
    if len(contexts) < 2:
        raise ValueError("synthetic decision pilot needs separate development/evaluation contexts")
    dev_context, test_context = contexts[0], contexts[-1]
    model_hash = canonical_hash({"training": [r.to_dict() for r in training], "method": "pooled_delta:shrinkage=1"})
    model = {"model_hash": model_hash, "dependencies": [{"name": "SYNTHETIC-training", "sha256": canonical_hash([r.to_dict() for r in training]), "available_at": stamp}]}
    test_predictions = [r for r in predictions if r.context == test_context]
    candidates = [{"candidate_id": r.record_id, "context_id": r.context, "target_id": r.perturbation,
                   "intervention_protocol": "SYNTHETIC-perturbation-fixture", "prediction_score": 0.0,
                   "model_hash": model_hash, "prediction_hash": canonical_hash(r.to_dict()),
                   "native_uncertainty": None, "native_uncertainty_kind": None,
                   "formal_status": "passed" if cert_refs[r.record_id]["verified"] else "failed",
                   "evidence_status": "missing", "supporting_evidence_refs": []} for r in test_predictions]
    panel = freeze_panel(candidates, settings, model)
    by_key = {(r.context, r.perturbation): r for r in observations}
    dev_rows = [{"candidate_id": r.record_id, "context_id": r.context, "split": "development", "model_hash": model_hash,
                 "native_uncertainty": None, "native_uncertainty_kind": None,
                 "agreement_error": agreement(r.to_dict(), by_key[(r.context, r.perturbation)].to_dict()), "available_at": stamp}
                for r in predictions if r.context == dev_context]
    calibration = fit_calibration(dev_rows, panel, settings)
    inputs = {"panel.json": panel, "calibration.json": calibration,
              "split.json": {"synthetic": True, "training_contexts": sorted({r.context for r in training}), "development_context": dev_context, "evaluation_context": test_context},
              "prediction_records.json": {"synthetic": True, "records": [r.to_dict() for r in test_predictions]},
              "fallback_records.json": {"synthetic": True, "records": [r.to_dict() for r in references if r.context == test_context]},
              "locked/outcomes.json": {"synthetic": True, "records": [by_key[(r.context, r.perturbation)].to_dict() for r in test_predictions]}}
    for name, value in inputs.items():
        write_json(project, f"{prefix}/{name}", value, overwrite=False)
    env = dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    timings, decision_paths = {}, []
    for policy_name in POLICIES:
        output = f"{prefix}/decisions/{policy_name}.json"
        args = [sys.executable, "-B", "-m", "invariantcell", "--project", str(project), "decision-select",
                "--frozen-panel", str(project / prefix / "panel.json"), "--protocol", str(project / prefix / "decision_protocol.json"),
                "--calibration", str(project / prefix / "calibration.json"), "--policy", policy_name, "--output", output]
        start = time.monotonic()
        completed = subprocess.run(args, cwd=project, env=env, capture_output=True, text=True, timeout=8)
        timings[policy_name] = time.monotonic() - start
        if completed.returncode:
            raise RuntimeError(f"selection subprocess {policy_name} failed: {completed.stderr[:1000]}")
        decision_paths.append(str(project / output))
    commitment = {"synthetic": True, "at": utc_now(), "panel_hash": panel["panel_hash"],
                  "files": {name: file_hash(name) for name in decision_paths},
                  "selector_boundary": "separate processes with no outcome input; not an OS sandbox"}
    write_json(project, f"{prefix}/decision_commitment.json", commitment, overwrite=False)
    args = [sys.executable, "-B", "-m", "invariantcell.synthetic_decision", "evaluate", str(project), prefix]
    completed = subprocess.run(args, cwd=project, env=env, capture_output=True, text=True, timeout=8)
    if completed.returncode:
        raise RuntimeError("synthetic evaluator failed: " + completed.stderr[:1000])
    write_json(project, f"{prefix}/decision_resources.json", {"synthetic": True, "selector_wall_seconds": timings,
               "external_tool_calls": 0, "limitations": "Predictor has no native uncertainty; calibrated policies correctly defer rather than invent risk."}, overwrite=False)
    return {"status": "SYNTHETIC_EVALUATED", "panel_hash": panel["panel_hash"], "protocol_hash": canonical_hash(settings),
            "policies": list(POLICIES), "evaluation": f"{prefix}/decision_evaluation.json", "calibrated_risk": "UNKNOWN_NO_NATIVE_UNCERTAINTY",
            "interpretation": "software behavior only; no biological or selection-advantage claim"}


def evaluate_files(project, prefix):
    """Only this evaluator subprocess joins withheld synthetic observations."""
    project = Path(project)
    base = project / prefix
    read = lambda name: json.loads((base / name).read_text())
    commitment = read("decision_commitment.json")
    panel, settings, calibration = read("panel.json"), read("decision_protocol.json"), read("calibration.json")
    decisions = {}
    for path, expected_hash in commitment["files"].items():
        if file_hash(path) != expected_hash:
            raise ValueError("committed decision changed before outcome access")
        decision = json.loads(Path(path).read_text())
        verify_decisions(decision, panel, settings, calibration)
        decisions[decision["policy"]] = decision
    write_json(project, f"{prefix}/locked/outcome_access.json", {"synthetic": True, "opened_at": utc_now(),
               "commitment_sha256": file_hash(base / "decision_commitment.json"), "actor": "evaluator_subprocess"}, overwrite=False)
    predictions, truths = read("prediction_records.json")["records"], read("locked/outcomes.json")["records"]
    by_key = {(r["context"], r["perturbation"]): r for r in truths}
    outcomes = [{"candidate_id": r["record_id"], "agreement_error": agreement(r, by_key[(r["context"], r["perturbation"])])} for r in predictions]
    report = evaluate(decisions, panel, settings, outcomes, calibration)
    reference_errors = [agreement(r, by_key[(r["context"], r["perturbation"])]) for r in read("fallback_records.json")["records"]]
    report.update({"synthetic": True, "scope": "single evaluation-context software smoke; not biological performance",
                   "count_baselines": {"pooled_delta_mean_expression_mse": sum(o["agreement_error"] for o in outcomes) / len(outcomes),
                                       "no_effect_mean_expression_mse": sum(reference_errors) / len(reference_errors)}})
    write_json(project, f"{prefix}/decision_evaluation.json", report, overwrite=False)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "evaluate":
        raise SystemExit("usage: python -m invariantcell.synthetic_decision evaluate PROJECT RUN_PREFIX")
    evaluate_files(sys.argv[2], sys.argv[3])

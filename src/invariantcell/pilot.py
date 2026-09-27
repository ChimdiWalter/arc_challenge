"""Gated local pilot orchestration. No external source code or network access."""
from __future__ import annotations
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import time
import uuid

from .io import digest, file_hash, load, output_path, require_experiment_clearance, utc_now, write_json


def run_pilot(project, cfg, suite, config=None):
    require_experiment_clearance(cfg, "synthetic" if suite == "synthetic" else "arc")
    if suite not in {"synthetic", "real-transfer", "arc-evaluation"}:
        raise ValueError("unknown pilot suite")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    # A distinct lock prevents duplicate InvariantCell experiment processes.
    lock = output_path(project, "state/experiment.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise PermissionError("experiment lock exists; inspect recorded process before resuming")
    os.write(fd, json.dumps({"pid": os.getpid(), "created_at": utc_now()}).encode())
    os.close(fd)
    start = time.monotonic()
    started_at = utc_now()
    run_id = suite + "-" + uuid.uuid4().hex[:12]
    budget_relative = cfg["compute"].get("experiment_budget_path", "state/experiment_budget.json")
    budget_path = output_path(project, budget_relative)
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else {"used_wall_seconds": 0, "reserved_wall_seconds": 0, "runs": []}
    cap = cfg["compute"]["max_cumulative_experiment_wall_minutes"] * 60
    # Reserve this tiny job before any generation; a crash leaves its reserve charged.
    reservation = min(180 if suite == "synthetic" else 600, cfg["compute"]["max_single_process_wall_minutes"] * 60)
    try:
        if not cfg["compute"].get("cumulative_experiment_time_is_telemetry_only", False) and budget["used_wall_seconds"] + budget["reserved_wall_seconds"] + reservation > cap:
            raise PermissionError("cumulative experiment budget insufficient")
        budget["reserved_wall_seconds"] += reservation
        write_json(project, budget_relative, budget)
        import signal
        def timeout(signum, frame):
            raise TimeoutError("pilot wall-time reservation exhausted")
        old_handler = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(int(reservation))
        # Single-thread stdlib pipeline; address-space limit is process-local.
        ram = int(cfg["compute"]["max_ram_gib"] * 1024**3)
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (min(ram, hard) if hard > 0 else ram, hard))
        try:
            if suite == "synthetic":
                result = _synthetic(project, cfg, run_id, config)
            elif suite == "arc-evaluation":
                from .arc_evaluation import run
                result = run(project, cfg, run_id, config)
            else:
                if config is not None:
                    raise ValueError("real pilot uses immutable frozen manifests")
                from .real_pilot import run
                result = run(project, cfg, run_id)
            result["resource_usage"] = {"wall_seconds": time.monotonic() - start, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "cpu_threads": 1, "gpu_hours": 0, "paid_usd": 0}
            result["started_at"] = started_at
            result["completed_at"] = utc_now()
            write_json(project, f"runs/{run_id}/manifest.json", result, overwrite=False)
            status = "COMPLETE"
        except Exception as error:
            result = {"run_id": run_id, "synthetic": suite == "synthetic", "status": "FAILED", "error": str(error)}
            status = "FAILED"
            write_json(project, f"runs/{run_id}/failure.json", result, overwrite=False)
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            elapsed = time.monotonic() - start
            budget["reserved_wall_seconds"] -= reservation
            budget["used_wall_seconds"] += elapsed
            budget["runs"].append({"run_id": run_id, "wall_seconds": elapsed, "status": locals().get("status", "FAILED")})
            write_json(project, budget_relative, budget)
            with output_path(project, "state/run_registry.jsonl").open("a") as stream:
                stream.write(json.dumps(budget["runs"][-1]) + "\n")
        return result
    finally:
        lock.unlink()


def _synthetic(project, cfg, run_id, config):
    from .core import miniature_fixture, pooled_delta, no_effect, validate_miniature_panel, canonical_hash
    from .temporal import TemporalQuery, issue_certificate, snapshot_from_transitions
    from .replay import verify_certificate
    from .synthetic_decision import protocol, run_bridge
    dimensions = dict(cfg["pilot"]["synthetic"])
    dimensions.pop("label_every_artifact_as_synthetic", None)
    if config:
        extra = load(config)
        if set(extra) - {"contexts", "perturbations", "genes", "cells_per_group"}:
            raise ValueError("synthetic config has unsupported fields")
        dimensions.update(extra)
    if dimensions["contexts"] < 3 or dimensions["contexts"] > 4 or dimensions["perturbations"] > 8 or dimensions["genes"] > 64 or dimensions["cells_per_group"] > 32:
        raise ValueError("tiny synthetic scope exceeded")
    prefix = f"runs/{run_id}"
    write_json(project, f"{prefix}/decision_protocol.json", protocol(), overwrite=False)
    stamp = "2026-01-01T00:00:00+00:00"
    records = tuple(replace(r, evidence_available_at=stamp, ingested_at=stamp) for r in miniature_fixture(seed=cfg["execution"]["seed"], **dimensions))
    predictions, references = [], []
    targets = sorted({r.perturbation for r in records if r.perturbation is not None})
    training_context = sorted({r.context for r in records})[0]
    training = tuple(r for r in records if r.context == training_context)
    for control in (r for r in records if r.perturbation is None and r.context != training_context):
        for target in targets:
            predictions.append(pooled_delta(training, control, target))
            references.append(no_effect(control, target))
    expected = [(r.context, r.perturbation) for r in predictions]
    validation = validate_miniature_panel(predictions, records[0].genes, expected, dimensions["cells_per_group"])
    # Synthetic software history only; not asserted real publication or event times.
    stamp = "2026-01-01T00:00:00+00:00"
    first = predictions[0]
    snapshot = snapshot_from_transitions((first,), "SYNTHETIC:" + run_id, generated_at=stamp, public_available_at=stamp, ingested_at=stamp, validity_kind="interval", valid_from=stamp)
    evidence = snapshot.evidence[0]
    query = TemporalQuery(evidence.subject, evidence.predicate, evidence.context,
                          stamp, stamp, stamp, allow_predictions=True)
    certificate = issue_certificate(snapshot, query)
    replay = verify_certificate(certificate, snapshot, canonical_hash(snapshot.to_dict()), query)
    if not replay.valid:
        raise ValueError("independent replay failed")
    cert_refs = {}
    for prediction in predictions:
        candidate_snapshot = snapshot_from_transitions((prediction,), "SYNTHETIC:" + prediction.record_id,
            generated_at=stamp, public_available_at=stamp, ingested_at=stamp, validity_kind="interval", valid_from=stamp)
        candidate_evidence = candidate_snapshot.evidence[0]
        candidate_query = TemporalQuery(candidate_evidence.subject, candidate_evidence.predicate, candidate_evidence.context, stamp, stamp, stamp, allow_predictions=True)
        candidate_certificate = issue_certificate(candidate_snapshot, candidate_query)
        receipt = verify_certificate(candidate_certificate, candidate_snapshot, canonical_hash(candidate_snapshot.to_dict()), candidate_query)
        if not receipt.valid:
            raise ValueError("candidate certificate replay failed")
        cert_name = canonical_hash(prediction.record_id)[:16] + ".json"
        path = write_json(project, f"{prefix}/certificates/{cert_name}", candidate_certificate, overwrite=False)
        cert_refs[prediction.record_id] = {"verified": True, "path": str(path), "sha256": file_hash(path)}
    decision = run_bridge(project, prefix, training, predictions, records, references, cert_refs)
    artifacts = {"transitions.json": {"synthetic": True, "records": [r.to_dict() for r in predictions]}, "fallback.json": {"synthetic": True, "records": [r.to_dict() for r in references]}, "arc_miniature_validation.json": validation, "snapshot.json": snapshot.to_dict(), "query.json": query.to_dict(), "certificate.json": certificate, "replay_receipt.json": replay.to_dict()}
    hashes = {}
    for name, value in artifacts.items():
        path = write_json(project, f"{prefix}/{name}", value, overwrite=False)
        hashes[name] = file_hash(path)
    for path in (Path(project) / prefix).rglob("*"):
        if path.is_file():
            hashes[str(path.relative_to(Path(project) / prefix))] = file_hash(path)
    import sys, platform
    sources = {str(path.relative_to(project)): file_hash(path) for path in (Path(project) / "src/invariantcell").glob("*.py")}
    result = {"run_id": run_id, "synthetic": True, "status": "CPU_PILOT_WORKING", "scope": "software integration only",
              "command": "PYTHONPATH=src python -B -m invariantcell pilot --suite synthetic",
              "config": dimensions, "configuration_sha256": digest(cfg), "code_sha256": digest(sources),
              "code_files": sources, "seed": cfg["execution"]["seed"], "artifact_sha256": hashes,
              "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
              "checks": {"miniature_panel": validation, "temporal_replay": replay.to_dict(), "candidate_certificates_replayed": len(cert_refs),
                         "decision_policies_evaluated": 5, "selection_before_outcome_access": True},
              "official_arc_metrics": "NOT_EVALUATED", "biological_performance": "NOT_EVALUATED", "nih_formal_ontology": "NOT_IMPLEMENTED",
              "decision_pilot": decision, "eligibility": cfg["compute"].get("nih_resource_eligibility", "NIH_RESOURCE_ELIGIBILITY_UNRESOLVED"),
              "resource_provenance": {"local_authorization": cfg["compute"]["resource_clearance"], "evidence": cfg["compute"]["resource_clearance_evidence_path"],
                                      "nih_resource_eligibility": cfg["compute"].get("nih_resource_eligibility")}}
    write_json(project, f"{prefix}/configuration.json", cfg, overwrite=False)
    result["artifact_sha256"]["configuration.json"] = file_hash(Path(project) / prefix / "configuration.json")
    return result

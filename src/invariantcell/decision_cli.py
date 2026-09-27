"""Separate CLI processes for committing selections and opening outcomes."""
from .io import load, require_experiment_clearance, write_json, file_hash, output_path, utc_now
import json


def register(sub):
    freeze = sub.add_parser("decision-freeze")
    freeze.add_argument("--prediction-run", required=True)
    freeze.add_argument("--protocol", required=True)
    freeze.add_argument("--output", required=True)
    select = sub.add_parser("decision-select")
    select.add_argument("--frozen-panel", required=True)
    select.add_argument("--protocol", required=True)
    select.add_argument("--policy", choices=["random", "native", "calibrated", "fixed_rules", "verification"], required=True)
    select.add_argument("--calibration")
    select.add_argument("--output", required=True)
    evaluate = sub.add_parser("decision-evaluate")
    evaluate.add_argument("--decisions", nargs="+", required=True)
    evaluate.add_argument("--frozen-panel", required=True)
    evaluate.add_argument("--protocol", required=True)
    evaluate.add_argument("--outcomes", required=True)
    evaluate.add_argument("--calibration")
    evaluate.add_argument("--output", required=True)
    query = sub.add_parser("query")
    query.add_argument("--snapshot", required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--output", required=True)
    verify = sub.add_parser("verify-certificate")
    verify.add_argument("--certificate", required=True)
    verify.add_argument("--snapshot", required=True)
    verify.add_argument("--snapshot-hash", required=True)
    verify.add_argument("--query", required=True)


def execute(args, project, cfg):
    if args.command == "query":
        from .temporal import Snapshot, TemporalQuery, issue_certificate
        result = issue_certificate(Snapshot.from_dict(load(args.snapshot)), TemporalQuery(**load(args.query)))
        write_json(project, args.output, result, overwrite=False)
        return result
    if args.command == "verify-certificate":
        from .temporal import Snapshot, TemporalQuery
        from .replay import verify_certificate
        result = verify_certificate(load(args.certificate), Snapshot.from_dict(load(args.snapshot)), args.snapshot_hash, TemporalQuery(**load(args.query))).to_dict()
        if not result["valid"]:
            raise ValueError("certificate replay failed: " + str(result["errors"]))
        return result
    require_experiment_clearance(cfg)
    from .decision import freeze_panel, select, evaluate, verify_decisions
    if args.command == "decision-freeze":
        run = load(args.prediction_run)
        result = freeze_panel(run["candidates"], load(args.protocol), run["model_snapshot"])
    elif args.command == "decision-select":
        result = select(load(args.frozen_panel), load(args.protocol), args.policy, load(args.calibration) if args.calibration else None)
    elif args.command == "decision-evaluate":
        panel, protocol = load(args.frozen_panel), load(args.protocol)
        calibration = load(args.calibration) if args.calibration else None
        decisions = {}
        for path in args.decisions:
            decision = load(path)
            verify_decisions(decision, panel, protocol, calibration)
            if decision["policy"] in decisions:
                raise ValueError("duplicate policy decisions")
            decisions[decision["policy"]] = decision
        # Materialized decisions are validated BEFORE this process opens outcomes.
        output = output_path(project, args.output)
        if output.exists():
            raise FileExistsError(output)
        access = {"at": utc_now(), "outcomes_path": str(args.outcomes), "decision_file_hashes": {str(p): file_hash(p) for p in args.decisions}, "scope": "evaluator_only"}
        with output_path(project, "state/locked_outcome_access.jsonl").open("a") as stream:
            stream.write(json.dumps(access) + "\n")
        result = evaluate(decisions, panel, protocol, load(args.outcomes)["outcomes"], calibration)
    else:
        raise ValueError("unsupported command")
    write_json(project, args.output, result, overwrite=False)
    return result

"""Explicit local commands; no submission, registration, or remote writes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from .io import configuration, load, require_experiment_clearance, utc_now, write_json

PROJECT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(prog="invariantcell")
    p.add_argument("--project", type=Path, default=PROJECT)
    sub = p.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("discover", help="bounded resumable read-only metadata census")
    scan.add_argument("--resume", action="store_true", help="resume is always used")
    scan.add_argument("--seconds", type=float, default=60)
    scan.add_argument("--max-entries", type=int, default=100000)
    sub.add_parser("discovery-report")
    review = sub.add_parser("discovery-review")
    review.add_argument("path", type=Path)
    review.add_argument("--start-line", type=int, default=1)
    review.add_argument("--max-lines", type=int, default=160)
    sub.add_parser("audit")
    check = sub.add_parser("status", help="read saved state without rerunning experiments")
    check.add_argument("--verify-artifacts", action="store_true")
    sub.add_parser("arc-preflight", help="inspect configured official bundle metadata only")
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--suite", choices=["synthetic", "real-transfer", "arc-evaluation"], required=True)
    pilot.add_argument("--config", type=Path)
    from .decision_cli import register
    register(sub)
    return p


def audit(project, cfg):
    sources = []
    for role, path in cfg.get("source_repos", {}).items():
        if role == "other":
            continue
        sources.append({"role": role, "path": path, "status": "SUPPLIED_EXISTS" if path and Path(path).exists() else "DISCOVERY_PENDING"})
    data = [{"path": p, "status": "PLACEHOLDER_NOT_USABLE" if str(p).startswith("/your/") else "SUPPLIED_EXISTS" if Path(p).exists() else "UNAVAILABLE"} for p in cfg.get("data_roots", [])]
    blockers = []
    try:
        require_experiment_clearance(cfg)
    except PermissionError as error:
        blockers.append(str(error))
    result = {"observed_at": utc_now(), "workspace": cfg["workspace"], "project": str(project), "sources": sources, "data_roots": data, "approved_search_roots": cfg.get("approved_search_roots", []), "experiment_blockers": blockers, "status": "AUDIT_ONLY", "eligibility": "NOT_CLEARED_FOR_NIH"}
    write_json(project, "audit/configuration_audit.json", result)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    project = args.project.resolve()
    try:
        cfg = configuration(project)
        if args.command == "status":
            from .readiness import status
            result = status(project, cfg, args.verify_artifacts)
        elif args.command == "arc-preflight":
            from .readiness import arc_preflight
            result = arc_preflight(project, cfg)
        elif args.command == "discover":
            from .discovery import scan, catalogs
            root = cfg.get("discovery", {}).get("recursive_root")
            approved = [str(Path(x).resolve()) for x in cfg.get("approved_search_roots", [])]
            if not root or str(Path(root).resolve()) not in approved:
                raise PermissionError("discovery root requires exact explicit authorization")
            result = scan(project, root, exclusions=[cfg["workspace"]], seconds=min(args.seconds, cfg["compute"]["max_single_process_wall_minutes"] * 60), max_entries=args.max_entries, cumulative_seconds=cfg["compute"]["max_cumulative_experiment_wall_minutes"] * 60)
            result["catalogs"] = catalogs(project)
        elif args.command == "discovery-report":
            from .discovery import coverage, catalogs
            result = coverage(project)
            result["catalogs"] = catalogs(project)
        elif args.command == "discovery-review":
            from .discovery import review_document
            result = review_document(project, args.path, args.start_line, args.max_lines)
        elif args.command == "audit":
            result = audit(project, cfg)
        elif args.command == "pilot":
            require_experiment_clearance(cfg, "synthetic" if args.suite == "synthetic" else "arc")
            from .pilot import run_pilot
            result = run_pilot(project, cfg, args.suite, args.config)
        else:
            from .decision_cli import execute
            result = execute(args, project, cfg)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, PermissionError, FileNotFoundError, FileExistsError, NotImplementedError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error), "command": args.command}), file=sys.stderr)
        return 2

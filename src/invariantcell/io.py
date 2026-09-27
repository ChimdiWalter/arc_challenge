"""Small explicit path, serialization, and configuration helpers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inside(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def output_path(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if not inside(path, root):
        raise ValueError("output escapes workspace")
    return path


def write_json(root, relative, value, overwrite=True):
    path = output_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(path)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    output_path(root, temporary)
    with temporary.open("x") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if not overwrite:
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink()
    else:
        os.replace(temporary, path)
    return path


def load(path):
    import yaml
    with Path(path).open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def merge(base, overlay):
    """Recursive explicit overlay; no inferred permissions."""
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def configuration(project, inputs=None):
    project = Path(project).resolve()
    defaults = load(project / "configs/execution_defaults.yaml")
    local = load(inputs or project / "inputs.local.yaml")
    cfg = merge(defaults, local)
    if not cfg.get("workspace") or not inside(project, cfg["workspace"]):
        raise ValueError("project is outside the explicitly configured workspace")
    return cfg


def require_experiment_clearance(cfg, purpose="arc"):
    """Local authorization and NIH prize eligibility are independent gates."""
    compute = cfg["compute"]
    local = compute.get("resource_clearance") == "user_authorized_local"
    legacy = compute.get("resource_clearance") == "cleared"
    if not (local or legacy) or not compute.get("resource_clearance_note"):
        raise PermissionError("RESOURCE_CLEARANCE_REQUIRED: record explicit local authorization")
    if purpose == "nih":
        if compute.get("nih_resource_eligibility") != "CLEARED":
            raise PermissionError("NIH_RESOURCE_ELIGIBILITY_UNRESOLVED: local authorization is not NIH eligibility")
    elif local and purpose not in compute.get("local_authorized_purposes", []):
        raise PermissionError("activity is outside the recorded local authorization")
    kind = compute.get("execution_resource_kind")
    if kind == "shared_login_node" and purpose not in {"engineering", "discovery", "synthetic"}:
        raise PermissionError("shared login nodes cannot run research model fitting")
    if kind not in {"workstation", "compute_node", "user_authorized_local_host", "shared_login_node"}:
        raise PermissionError("EXECUTION_RESOURCE_UNVERIFIED: record the execution resource")
    if compute.get("device") != "cpu":
        raise PermissionError("this implementation authorizes CPU execution only")

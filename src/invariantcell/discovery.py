"""Resumable metadata census. Never imports or executes discovered material.

SQLite is the compact equivalent of file/directory JSONL inventories. Content
review is a separate, explicit operation. Exclusions remain queryable records.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import time

from .io import inside, output_path, utc_now, write_json

DOCUMENT_SUFFIXES = {".md", ".rst", ".txt", ".org", ".tex", ".ipynb", ".pdf", ".docx"}
ASSET_SUFFIXES = {".h5ad", ".h5", ".hdf5", ".zarr", ".parquet", ".mtx", ".npz", ".npy", ".csv", ".tsv", ".pt", ".pth", ".pkl", ".pickle", ".zip", ".gz", ".tar", ".safetensors"}
PROTECTED = re.compile(r"(^|[/_.-])(locked|heldout|holdout|ground_truth|secrets?|credentials?|private|tokens?)([/_.-]|$)", re.I)
OUTCOME = re.compile(r"held[-_ ]?out|locked[-_ ]?test|test[-_ ]?(labels|outcomes|scores)|ground[-_ ]?truth", re.I)
PLAN = re.compile(r"readme|overview|plan|roadmap|update|progress|status|handoff|resume|decision|blocker|notes|summary|claim", re.I)


class ClosingConnection(sqlite3.Connection):
    """Commit/rollback and close on context exit, including NFS file handles."""
    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self.close()


def connect(project):
    dbpath = output_path(project, "audit/discovery/inventory.sqlite")
    dbpath.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(dbpath, factory=ClosingConnection)
    db.execute("PRAGMA journal_mode=DELETE")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS entries(
      path TEXT PRIMARY KEY,resolved TEXT,parent TEXT,kind TEXT,size INTEGER,
      mtime_ns INTEGER,observed_at TEXT,status TEXT,reason TEXT,content_status TEXT);
    CREATE TABLE IF NOT EXISTS queue(id INTEGER PRIMARY KEY,path TEXT UNIQUE,status TEXT);
    CREATE TABLE IF NOT EXISTS visited(identity TEXT PRIMARY KEY,path TEXT);
    CREATE TABLE IF NOT EXISTS reads(
      id INTEGER PRIMARY KEY,path TEXT,start_line INTEGER,end_line INTEGER,
      total_lines INTEGER,sha256 TEXT,status TEXT,observed_at TEXT,reason TEXT);
    """)
    return db


def metadata(db, key, default=None):
    row = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_metadata(db, key, value):
    db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, json.dumps(value)))


def _boundary(path, root, exclusions):
    resolved = path.resolve(strict=True)
    if not inside(resolved, root):
        return "DEFERRED", "external symlink boundary", resolved
    if any(inside(resolved, item) for item in exclusions):
        return "PRUNED_WITH_REASON", "integration workspace excluded", resolved
    if resolved != root and os.path.ismount(resolved):
        return "DEFERRED", "nested mount requires scope/resource review", resolved
    if resolved.is_dir():
        if (resolved / "pyvenv.cfg").is_file():
            return "PRUNED_WITH_REASON", "verified virtual environment marker pyvenv.cfg", resolved
        if resolved.name == "objects" and resolved.parent.name == ".git" and (resolved.parent / "HEAD").is_file():
            return "PRUNED_WITH_REASON", "verified Git object store", resolved
        if resolved.name == "node_modules" and (resolved / ".package-lock.json").is_file():
            return "PRUNED_WITH_REASON", "verified installed dependency tree", resolved
    return "ENUMERATED", "", resolved


def scan(project, root, exclusions=(), seconds=60, max_entries=100000, cumulative_seconds=1800):
    project, root = Path(project).resolve(), Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("discovery root is not a directory")
    exclusions = tuple(Path(p).resolve() for p in exclusions) + (project,)
    if any(inside(root, p) for p in exclusions):
        raise ValueError("discovery root overlaps excluded output root")
    if seconds <= 0 or max_entries <= 0:
        raise ValueError("scan caps must be positive")
    db = connect(project)
    start = time.monotonic()
    previous = metadata(db, "scan_wall_seconds", 0.0)
    duration = min(seconds, max(0, cumulative_seconds - previous))
    oldroot = metadata(db, "root")
    if oldroot is not None and oldroot != str(root):
        raise ValueError("inventory belongs to a different root")
    set_metadata(db, "root", str(root))
    set_metadata(db, "exclusions", [str(x) for x in exclusions])
    if metadata(db, "started_at") is None:
        set_metadata(db, "started_at", utc_now())
    db.execute("INSERT OR IGNORE INTO queue(path,status) VALUES (?, 'PENDING')", (str(root),))
    new_count = 0
    def exhausted():
        return new_count >= max_entries or time.monotonic() - start >= duration
    while not exhausted():
        row = db.execute("SELECT path FROM queue WHERE status='PENDING' ORDER BY id LIMIT 1").fetchone()
        if not row:
            break
        directory = Path(row[0])
        try:
            status, reason, resolved = _boundary(directory, root, exclusions)
            if status != "ENUMERATED":
                db.execute("UPDATE queue SET status=? WHERE path=?", (status, str(directory)))
                db.execute("UPDATE entries SET status=?,reason=? WHERE path=?", (status, reason, str(directory)))
                continue
            st = resolved.stat()
            identity = f"{st.st_dev}:{st.st_ino}"
            seen = db.execute("SELECT path FROM visited WHERE identity=?", (identity,)).fetchone()
            if seen and seen[0] != str(directory):
                db.execute("UPDATE queue SET status='ALIAS' WHERE path=?", (str(directory),))
                db.execute("UPDATE entries SET status='DEFERRED',reason=? WHERE path=?", ("duplicate directory/loop alias of " + seen[0], str(directory)))
                continue
            db.execute("INSERT OR IGNORE INTO visited VALUES (?,?)", (identity, str(directory)))
            completed = True
            with os.scandir(directory) as children:
                for entry in children:
                    if exhausted():
                        completed = False
                        break
                    logical = Path(entry.path)
                    if db.execute("SELECT 1 FROM entries WHERE path=?", (str(logical),)).fetchone():
                        continue
                    new_count += 1
                    try:
                        st = entry.stat(follow_symlinks=False)
                        kind = "symlink" if stat.S_ISLNK(st.st_mode) else "directory" if stat.S_ISDIR(st.st_mode) else "file" if stat.S_ISREG(st.st_mode) else "special"
                        status, reason, resolved = _boundary(logical, root, exclusions)
                        protected = PROTECTED.search(str(logical.relative_to(root))) or logical.name == ".env" or logical.suffix.lower() in {".pem", ".key"}
                        content = "PROTECTED" if protected else "METADATA_ONLY"
                        if protected and status == "ENUMERATED":
                            reason = "content protected; metadata traversal only"
                        db.execute("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?)", (str(logical), str(resolved), str(logical.parent), kind, st.st_size, st.st_mtime_ns, utc_now(), status, reason, content))
                        if status == "ENUMERATED" and resolved.is_dir():
                            db.execute("INSERT OR IGNORE INTO queue(path,status) VALUES (?, 'PENDING')", (str(logical),))
                    except (OSError, RuntimeError) as error:
                        db.execute("INSERT OR IGNORE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?)", (str(logical), None, str(logical.parent), "unknown", None, None, utc_now(), "INACCESSIBLE", type(error).__name__ + ": " + str(error), "METADATA_ONLY"))
            if completed:
                db.execute("UPDATE queue SET status='DONE' WHERE path=?", (str(directory),))
        except (OSError, RuntimeError) as error:
            db.execute("UPDATE queue SET status='INACCESSIBLE' WHERE path=?", (str(directory),))
            db.execute("INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?)", (str(directory), None, str(directory.parent), "directory", None, None, utc_now(), "INACCESSIBLE", type(error).__name__ + ": " + str(error), "METADATA_ONLY"))
        db.commit()
    elapsed = time.monotonic() - start
    set_metadata(db, "scan_wall_seconds", previous + elapsed)
    set_metadata(db, "last_scan_at", utc_now())
    db.commit()
    result = coverage(project, db)
    result["new_entries_this_call"] = new_count
    result["wall_seconds_this_call"] = elapsed
    write_json(project, "state/discovery_resume.json", result)
    db.close()
    return result


def coverage(project, db=None):
    own = db is None
    db = db or connect(project)
    statuses = dict(db.execute("SELECT status,count(*) FROM entries GROUP BY status"))
    queues = dict(db.execute("SELECT status,count(*) FROM queue GROUP BY status"))
    content = dict(db.execute("SELECT content_status,count(*) FROM entries GROUP BY content_status"))
    result = {"root": metadata(db, "root"), "started_at": metadata(db, "started_at"), "last_scan_at": metadata(db, "last_scan_at"), "scan_wall_seconds": metadata(db, "scan_wall_seconds", 0), "entry_status_counts": statuses, "directory_queue_counts": queues, "content_status_counts": content, "read_operations": db.execute("SELECT count(*) FROM reads").fetchone()[0], "enumeration_status": "PARTIAL_DISCOVERY" if queues.get("PENDING", 0) else "QUEUE_EXHAUSTED_UNDER_EXCLUSIONS", "semantic_review_status": "PARTIAL", "globally_atomic_snapshot": False, "pending_queue": "audit/discovery/inventory.sqlite: queue WHERE status='PENDING'", "next_command": "PYTHONPATH=src python -B -m invariantcell discover --resume --seconds 60"}
    write_json(project, "audit/discovery/scan_manifest.json", result)
    path = output_path(project, "audit/discovery/coverage_report.md")
    path.write_text("# Discovery coverage\n\n" + json.dumps(result, indent=2) + "\n\nMetadata enumeration does not establish document or semantic review.\nAll assets remain metadata-only until separately read. Protected content is not indexed.\nFull exclusion/error details and pending paths are in inventory.sqlite.\nThe scan is incremental, not a globally atomic snapshot.\n")
    skipped = output_path(project, "audit/discovery/skipped_inaccessible_and_deferred.csv")
    with skipped.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "status", "reason", "content_status"])
        writer.writerows(db.execute("SELECT path,status,reason,content_status FROM entries WHERE status!='ENUMERATED' OR content_status='PROTECTED'"))
    if own:
        db.close()
    return result


def catalogs(project):
    db = connect(project)
    projects = {}
    assets = output_path(project, "audit/discovery/asset_catalog.csv")
    documents = output_path(project, "audit/discovery/document_candidates.csv")
    with assets.open("w", newline="") as a, documents.open("w", newline="") as d:
        aw, dw = csv.writer(a), csv.writer(d)
        aw.writerow(["path", "size_bytes", "content_status", "dimensions", "license", "split_membership"])
        dw.writerow(["path", "size_bytes", "mtime_ns", "content_status", "triage_hint"])
        for path, parent, kind, size, mtime, content in db.execute("SELECT path,parent,kind,size,mtime_ns,content_status FROM entries"):
            p = Path(path)
            if kind not in {"file", "directory"}:
                continue
            if p.suffix.lower() in ASSET_SUFFIXES:
                aw.writerow([path, size, content, "UNKNOWN", "UNRESOLVED", "UNKNOWN"])
            if p.suffix.lower() in DOCUMENT_SUFFIXES:
                dw.writerow([path, size, mtime, content, "PLAN_OR_UPDATE_NAME" if PLAN.search(p.name) else "UNREVIEWED_DOCUMENT"])
            if p.name.lower().startswith("readme") or p.name in {"pyproject.toml", "setup.py", "package.json", ".git"}:
                projects.setdefault(parent, set()).add(p.name)
    with output_path(project, "audit/discovery/project_catalog.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["candidate_directory", "boundary_signals", "review_status", "relevance_status"])
        for path, signals in sorted(projects.items()):
            writer.writerow([path, ";".join(sorted(signals)), "METADATA_ONLY", "UNRESOLVED"])
    db.close()
    return {"project_candidates": len(projects), "status": "METADATA_ONLY; boundary signals require document review"}


def review_document(project, path, start_line=1, max_lines=160, max_bytes=262144):
    """Explicit bounded plain-text read, with outcome screening before return.

    Screening is a software safeguard, not proof of absence of hidden outcomes.
    Protected docs are never returned. Notebook outputs and binary formats are
    deliberately deferred until a separate safe reader is implemented.
    """
    if start_line < 1 or max_lines < 1:
        raise ValueError("line bounds must be positive")
    db = connect(project)
    try:
        path = Path(path).absolute()
        root = Path(metadata(db, "root"))
        if not inside(path, root):
            raise ValueError("document outside authorized root")
        row = db.execute("SELECT content_status FROM entries WHERE path=?", (str(path),)).fetchone()
        if not row:
            raise ValueError("metadata inventory must observe a document before content review")
        if row[0] == "PROTECTED":
            raise PermissionError("protected document content is not available to discovery")
        if any(inside(path, p) for p in metadata(db, "exclusions", [])):
            raise PermissionError("integration outputs excluded from source review")
        if path.suffix.lower() not in {".md", ".txt", ".rst", ".org", ".tex", ".yaml", ".yml", ".json", ".toml", ".py"}:
            raise ValueError("unsupported content format; remains metadata-only")
        before = path.stat()
        if before.st_size > max_bytes:
            raise ValueError("document exceeds bounded reader cap; content review deferred")
        with path.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError("document grew beyond read cap")
        sha = hashlib.sha256(raw).hexdigest()
        lines = raw.decode("utf-8", errors="strict").splitlines()
        after = path.stat()
        changed = (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size)
        selected = lines[start_line - 1:start_line - 1 + max_lines]
        end = min(len(lines), start_line - 1 + max_lines)
        reason = ""
        if OUTCOME.search("\n".join(selected)):
            status, reason = "PROTECTED", "answer-bearing terms in requested range; content deferred before agent exposure"
            selected = []
        elif changed:
            status, reason, selected = "CHANGED_DURING_SCAN", "observed file changed while read", []
        else:
            status = "READ_COMPLETE" if start_line == 1 and end == len(lines) else "READ_PARTIAL"
        db.execute("INSERT INTO reads(path,start_line,end_line,total_lines,sha256,status,observed_at,reason) VALUES (?,?,?,?,?,?,?,?)", (str(path), start_line, end, len(lines), sha, status, utc_now(), reason))
        db.execute("UPDATE entries SET content_status=? WHERE path=?", (status, str(path)))
        db.commit()
        db.close()
        return {"path": str(path), "sha256": sha, "status": status, "reason": reason, "start_line": start_line, "end_line": end, "total_lines": len(lines), "continuation": end + 1 if end < len(lines) else None, "lines": [f"{start_line+i}: {line}" for i, line in enumerate(selected)]}
    finally:
        db.close()

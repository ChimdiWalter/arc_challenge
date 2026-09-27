#!/usr/bin/env python3
"""Safely ingest the supplied official controls and audit every count, offline."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import resource
import shutil
import signal
import stat
import time
import zipfile

import h5py
import numpy as np
from anndata.io import read_elem
from scipy import sparse
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT.parent / "controls.zip"
DEST = PROJECT / "data/arc/official_validation_2026"
MEMBERS = {"context_A.h5ad", "context_B.h5ad", "context_C.h5ad", "gene_names.csv", "pert_counts.csv", "manifest.json"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def ordered_digest(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def checked_entries(infos: list[zipfile.ZipInfo], limit: int) -> list[zipfile.ZipInfo]:
    names = [i.filename for i in infos]
    if len(names) != len(set(names)):
        raise ValueError("duplicate archive names")
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if path.is_absolute() or len(path.parts) != 1 or "\\" in info.filename or ":" in info.filename:
            raise ValueError("unsafe archive path")
        if info.is_dir() or (stat.S_IFMT(mode) and not stat.S_ISREG(mode)):
            raise ValueError("archive member is not a regular file")
        if info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise ValueError("unsupported or encrypted archive entry")
        if info.file_size < 0 or info.file_size > limit or info.file_size > max(1, info.compress_size) * 200:
            raise ValueError("archive member exceeds size/ratio limit")
    if set(names) != MEMBERS:
        raise ValueError("unexpected membership; retain source and inventory before changing allowlist")
    if sum(i.file_size for i in infos) > limit:
        raise ValueError("archive expansion exceeds limit")
    return infos


def extract(source: Path, dest: Path, limit: int = 1024**3) -> dict:
    source_hash = digest(source)
    with zipfile.ZipFile(source) as archive:
        entries = checked_entries(archive.infolist(), limit)
        needed = sum(x.file_size for x in entries)
        if shutil.disk_usage(dest.parent if dest.parent.exists() else PROJECT).free < needed + 128 * 1024**2:
            raise ValueError("insufficient disk space")
        receipt_path = dest / "extraction_receipt.json"
        if dest.exists():
            if dest.is_symlink() or not receipt_path.is_file():
                raise ValueError("destination exists without a completed extraction receipt")
            receipt = json.loads(receipt_path.read_text())
            if receipt["archive_sha256"] != source_hash:
                raise ValueError("source changed since extraction")
            for row in receipt["files"]:
                path = dest / row["name"]
                if path.is_symlink() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                    raise ValueError("extracted file changed")
            return receipt
        dest.mkdir(parents=True, mode=0o700)
        rows = []
        for entry in entries:
            final = dest / entry.filename
            temporary = dest / (entry.filename + ".part")
            h = hashlib.sha256()
            total = 0
            with archive.open(entry) as incoming, temporary.open("xb") as outgoing:
                for block in iter(lambda: incoming.read(4 * 1024**2), b""):
                    total += len(block)
                    if total > entry.file_size:
                        raise ValueError("expanded member exceeds declared size")
                    h.update(block)
                    outgoing.write(block)
            if total != entry.file_size:
                raise ValueError("truncated member")
            temporary.chmod(0o444)
            os.replace(temporary, final)
            rows.append({"name": entry.filename, "bytes": total, "sha256": h.hexdigest(),
                         "zip_crc32": f"{entry.CRC:08x}", "compressed_bytes": entry.compress_size})
        receipt = {"status": "EXTRACTED_CRC_AND_SHA256_VERIFIED", "source": str(source),
                   "archive_bytes": source.stat().st_size, "archive_sha256": source_hash,
                   "extracted_bytes": needed, "files": rows, "source_modified": False,
                   "counts_audited": False}
        write_json(receipt_path, receipt)
        return receipt


def read_list(path: Path, column: str, expected: int) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != [column]:
            raise ValueError(f"unexpected CSV columns: {path.name}")
        values = [row[column] for row in reader]
    if len(values) != expected or len(set(values)) != expected or any(not x or x != x.strip() for x in values):
        raise ValueError(f"invalid or duplicate axis: {path.name}")
    return values


def distribution(values: np.ndarray) -> dict:
    q = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return dict(zip(["min", "p05", "p25", "median", "p75", "p95", "max"], map(float, q)))


def characterize(path: Path, context: str, genes: list[str], manifest: dict, chunk_rows: int = 128) -> tuple[dict, dict]:
    expected = manifest["per_context"][context]
    with h5py.File(path, "r") as handle:
        obs = read_elem(handle["obs"])
        var = read_elem(handle["var"])
        if var.index.astype(str).tolist() != genes:
            raise ValueError("H5AD axis differs from official ordered genes")
        n, g = len(obs), len(genes)
        if n != expected["control_cells"] or g != manifest["n_genes"]:
            raise ValueError("H5AD shape differs from manifest")
        if not obs.index.is_unique:
            raise ValueError("duplicate cell identifiers")
        for col, value in [(manifest["context_col"], context), (manifest["pert_col"], manifest["control_label"])]:
            if col not in obs or obs[col].isna().any() or set(obs[col].astype(str)) != {value}:
                raise ValueError(f"invalid control/context label in {col}")
        if "ntc_id" not in obs or obs["ntc_id"].isna().any():
            raise ValueError("missing guide labels")
        guides = sorted(obs["ntc_id"].astype(str).unique())
        if len(guides) != expected["n_ntc_ids"]:
            raise ValueError("guide count differs from manifest")
        guide_lookup = {x: i for i, x in enumerate(guides)}
        codes = np.array([guide_lookup[x] for x in obs["ntc_id"].astype(str)])
        guide_n = np.bincount(codes, minlength=len(guides))
        if not np.all(guide_n == manifest["cells_per_pert"]):
            raise ValueError("unexpected cells per control guide")
        x = handle["X"]
        storage = x.attrs.get("encoding-type", "dense")
        if isinstance(storage, bytes):
            storage = storage.decode()
        if isinstance(x, h5py.Group):
            if storage != "csr_matrix" or tuple(x.attrs["shape"]) != (n, g):
                raise ValueError("unsupported sparse encoding/shape")
            ptr = x["indptr"][:]
            if len(ptr) != n + 1 or ptr[0] != 0 or np.any(np.diff(ptr) < 0) or ptr[-1] != len(x["data"]) or len(x["indices"]) != len(x["data"]):
                raise ValueError("malformed CSR structure")
            dtype = str(x["data"].dtype)
        else:
            if x.shape != (n, g):
                raise ValueError("wrong dense shape")
            dtype = str(x.dtype)
        total = np.zeros(g); squares = np.zeros(g); detected = np.zeros(g, dtype=np.int64)
        cp_total = np.zeros(g); cp_squares = np.zeros(g)
        guide_total = np.zeros((len(guides), g)); guide_cp = np.zeros_like(guide_total)
        libraries = np.zeros(n); nonzero = np.zeros(n, dtype=np.int64)
        fixed = np.unique(np.linspace(0, g - 1, min(128, g), dtype=int))
        descriptors = np.empty((n, len(fixed)), dtype=np.float32)
        stored_entries = 0; stored_zeros = 0; max_count = 0
        for start in range(0, n, chunk_rows):
            end = min(n, start + chunk_rows)
            if isinstance(x, h5py.Group):
                lo, hi = int(ptr[start]), int(ptr[end]); values = x["data"][lo:hi]
                indexes = x["indices"][lo:hi]
                if np.any(indexes < 0) or np.any(indexes >= g):
                    raise ValueError("CSR column index outside official axis")
                matrix = sparse.csr_matrix((values, indexes, ptr[start:end + 1] - lo), shape=(end-start, g))
                if not matrix.has_sorted_indices:
                    matrix.sort_indices()
                if not matrix.has_canonical_format:
                    raise ValueError("duplicate CSR entries")
                dense = matrix.toarray().astype(np.float64)
            else:
                values = x[start:end, :]; dense = values.astype(np.float64)
            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
                raise ValueError("nonfinite/negative/fractional counts in complete audit")
            stored_entries += values.size; stored_zeros += int(np.count_nonzero(values == 0))
            max_count = max(max_count, float(values.max(initial=0)))
            depth = dense.sum(axis=1)
            if np.any(depth > 1_000_000):
                raise ValueError("control cell exceeds raw-count cap")
            libraries[start:end] = depth
            nonzero[start:end] = np.count_nonzero(dense, axis=1)
            total += dense.sum(axis=0); squares += np.square(dense).sum(axis=0); detected += np.count_nonzero(dense, axis=0)
            normalized = dense * (10_000 / np.maximum(depth, 1))[:, None]
            cp_total += normalized.sum(axis=0); cp_squares += np.square(normalized).sum(axis=0)
            descriptors[start:end] = np.log1p(normalized[:, fixed])
            for code in np.unique(codes[start:end]):
                mask = codes[start:end] == code
                guide_total[code] += dense[mask].sum(axis=0)
                guide_cp[code] += normalized[mask].sum(axis=0)
        mean = total / n
        variance = np.maximum(0, (squares - total * total / n) / (n - 1))
        cp_mean = cp_total / n
        cp_var = np.maximum(0, (cp_squares - cp_total * cp_total / n) / (n - 1))
        guide_mean = guide_total / guide_n[:, None]
        guide_cp_mean = guide_cp / guide_n[:, None]
        guide_se = guide_mean.std(axis=0, ddof=1) / np.sqrt(len(guides))
        between = guide_cp_mean.var(axis=0, ddof=1)
        noise_corrected = np.maximum(0, between - cp_var / guide_n.mean())
        fraction = noise_corrected / np.maximum(cp_var, 1e-12)
        eligible = detected >= max(2, n // 100)
        half = np.arange(n) % 2 == 0
        cov = np.cov(descriptors.astype(np.float64), rowvar=False)
        cov_a = np.cov(descriptors[half].astype(np.float64), rowvar=False)
        cov_b = np.cov(descriptors[~half].astype(np.float64), rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        ordering = np.argsort(eigenvalues)[::-1][:min(16, len(fixed))]
        feature = {"mean_counts": mean, "variance_counts": variance,
                   "dispersion_excess": np.maximum(variance - mean, 0) / np.maximum(mean * mean, 1e-12),
                   "detection_frequency": detected / n, "guide_mean_counts": guide_mean,
                   "guide_mean_cp10k": guide_cp_mean, "guide_mean_standard_error": guide_se,
                   "guide_nuisance_variance_fraction_cp10k": fraction,
                   "guide_ids": np.asarray(guides), "guide_n_cells": guide_n,
                   "library_sizes": libraries, "nonzeros_per_cell": nonzero,
                   "fixed_gene_positions": fixed, "descriptor_mean_log_cp10k": descriptors.mean(axis=0),
                   "descriptor_covariance": cov, "descriptor_eigenvalues": eigenvalues[ordering],
                   "descriptor_eigenvectors": eigenvectors[:, ordering]}
        guide_depth = [float(libraries[codes == i].mean()) for i in range(len(guides))]
        summary = {"context": context, "n_obs": n, "n_vars": g, "obs_columns": obs.columns.tolist(),
                   "var_columns": var.columns.tolist(), "X_storage": storage, "X_dtype": dtype,
                   "all_stored_counts_audited": True, "all_implicit_sparse_zeros_valid": True,
                   "finite_nonnegative_integral": True, "max_count_entry": max_count,
                   "stored_entries": stored_entries, "explicit_stored_zeros": stored_zeros,
                   "nonzero_entries": int(nonzero.sum()), "sparsity_fraction": 1 - nonzero.sum() / (n*g),
                   "gene_axis_sha256": ordered_digest(genes), "library_size": distribution(libraries),
                   "nonzeros_per_cell": distribution(nonzero), "ntc_id_counts": dict(zip(guides, map(int, guide_n))),
                   "guide_mean_library_sizes": distribution(np.asarray(guide_depth)),
                   "guide_nuisance_fraction_cp10k_genes_detected_at_least_1pct": distribution(fraction[eligible]),
                   "guide_descriptor_standard_error_median": float(np.median(guide_se[eligible])),
                   "covariance_split_relative_frobenius": float(np.linalg.norm(cov_a-cov_b)/max(np.linalg.norm(cov), 1e-12)),
                   "covariance_scope": "128 fixed evenly spaced official-axis genes; alternating-cell split; descriptive only",
                   "descriptor_top16_variance_fraction": float(eigenvalues[ordering].sum()/max(np.trace(cov), 1e-12)),
                   "uncertainty_scope": "Guide means are nuisance pseudo-replicates, not biological replicates; normal approximation descriptive only"}
        return summary, feature


def workspace_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run and not args.extract_only:
        parser.error("choose --extract-only or --run")
    started = time.monotonic()
    def deadline(*_):
        raise TimeoutError("official bundle audit exceeded 1200 seconds")
    signal.signal(signal.SIGALRM, deadline); signal.alarm(1200)
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    report_path = PROJECT / "reports/arc/official_bundle_audit.json"
    if report_path.exists():
        raise ValueError("completed audit exists; do not repeat")
    before = workspace_bytes(PROJECT)
    if before + (0 if DEST.exists() else SOURCE.stat().st_size) + 128 * 1024**2 > 30 * 1024**3:
        raise ValueError("official audit cannot fit workspace cap")
    try:
        with threadpool_limits(limits=1):
            extraction = extract(SOURCE, DEST)
            manifest = json.loads((DEST / "manifest.json").read_text())
            if manifest.get("season") != "2026" or manifest.get("partition") != "val" or manifest.get("contexts") != ["A", "B", "C"] or manifest.get("n_genes") != 18533 or manifest.get("n_constructs") != 300 or manifest.get("cells_per_pert") != 400:
                raise ValueError("provided archive is not the expected validation phase contract")
            genes = read_list(DEST / "gene_names.csv", "gene_name", manifest["n_genes"])
            targets = read_list(DEST / "pert_counts.csv", manifest["pert_col"], manifest["n_constructs"])
            if not set(targets).issubset(set(genes)):
                raise ValueError("official targets missing from official gene axis")
            preliminary = {"status": "EXTRACTION_AND_METADATA_VERIFIED_COUNTS_PENDING", "manifest": manifest,
                           "gene_axis_sha256": ordered_digest(genes), "target_axis_sha256": ordered_digest(targets),
                           "axis_hash_encoding": "SHA256 compact UTF-8 JSON ordered string array; no sorting",
                           "extraction": extraction}
            write_json(DEST / "bundle_audit_manifest.json", preliminary)
            if args.extract_only:
                print(json.dumps(preliminary, indent=2)); return
            summaries = []
            arrays = {"genes": np.asarray(genes), "targets": np.asarray(targets), "contexts": np.asarray(manifest["contexts"])}
            for context in manifest["contexts"]:
                summary, features = characterize(DEST / f"context_{context}.h5ad", context, genes, manifest)
                summaries.append(summary)
                arrays.update({context + "_" + k: value for k, value in features.items()})
                write_json(DEST / f"context_{context}_audit.json", summary)
                print(f"context {context}: complete all-count audit {summary['nonzero_entries']} nonzeros", flush=True)
            artifact = PROJECT / "artifacts/arc/context_features_validation.npz"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            if artifact.exists():
                raise ValueError("context feature artifact already exists")
            np.savez_compressed(artifact, **arrays)
            report = {**preliminary, "status": "COMPLETE", "counts_audited": True, "contexts": summaries,
                      "feature_artifact": str(artifact.relative_to(PROJECT)), "feature_sha256": digest(artifact),
                      "feature_bytes": artifact.stat().st_size, "script_sha256": digest(Path(__file__)),
                      "wall_seconds": time.monotonic()-started, "max_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "workspace_bytes_before": before, "workspace_bytes_after_before_receipt": workspace_bytes(PROJECT),
                      "cpu_threads": 1, "ram_limit_gib": 8, "wall_limit_seconds": 1200,
                      "download_bytes": 0, "paid_usd": 0, "gpu_hours": 0, "hidden_outcomes_read": False,
                      "deanonymization_attempted": False, "pathway_summaries": "NOT_RUN_WITHIN_THIS_BOUNDED_AUDIT",
                      "public_context_analogues": "NOT_RUN; no identity inference",
                      "decoder_comparison": "CONTROL_MOMENTS_ONLY; no response accuracy or decoder winner claim"}
            write_json(report_path, report)
            write_json(DEST / "bundle_audit_manifest.json", {**preliminary, "status": "COMPLETE", "counts_audited": True,
                       "report": "reports/arc/official_bundle_audit.json", "report_sha256": digest(report_path)})
            markdown = ["# Official validation bundle audit", "", "**COMPLETE: original archive preserved; all counts audited.** No hidden responses, downloads, authentication or identity inference.", "",
                        f"Archive SHA256 `{extraction['archive_sha256']}`; {extraction['archive_bytes']:,} source bytes; {extraction['extracted_bytes']:,} extracted bytes.",
                        f"Phase `{manifest['season']}/{manifest['partition']}/{manifest['panel_id']}`. Ordered official axis: {len(genes):,} genes; {len(targets)} target symbols; contexts A/B/C retained.",
                        f"Gene axis `{ordered_digest(genes)}`; target list `{ordered_digest(targets)}`. Hash encoding is compact UTF-8 JSON ordered strings; CSV file hashes are separate in the JSON report.", "",
                        "| Context | Cells | Guides | Library min / median / max | Sparsity | Stored count dtype |", "|---|---:|---:|---|---:|---|"]
            for s in summaries:
                lib = s["library_size"]
                markdown.append(f"| {s['context']} | {s['n_obs']} | {len(s['ntc_id_counts'])} × 400 | {lib['min']:.0f} / {lib['median']:.0f} / {lib['max']:.0f} | {s['sparsity_fraction']:.4f} | {s['X_dtype']} |")
            markdown += ["", "Every stored value was checked in 128-row chunks for finiteness, nonnegativity and integrality; sparse implicit zeros are valid. No transformation was written to the original H5ADs. Count integrity is consistent with raw count input; it does not independently reconstruct experimental preprocessing.", "",
                         "The feature artifact stores complete gene means, variances, overdispersion, detection frequencies, per-guide raw/CP10K means, guide standard errors, library sizes and a covariance descriptor on128 fixed axis positions. Guide variation is a nuisance diagnostic; guide strata are not biological replicates or perturbation predictors. Covariance split stability is descriptive, not a validation of a response decoder.", "",
                         "Pathway/GO summaries and nearest-public-context transfer descriptors remain NOT_RUN in this bounded audit; no context identity is inferred. Decoder accuracy requires public-response development evidence. Source manifest ground_truth_cells is metadata only; no such outcome matrix is present in this archive.", "",
                         f"Wall {report['wall_seconds']:.3f}s; peak RSS {report['max_rss_KiB']/1024:.1f}MiB; CPU1; zero download/GPU/paid usage. Exact file hashes, metadata columns, guide counts and distribution quantiles: [JSON report](official_bundle_audit.json). Features: [NPZ](../../artifacts/arc/context_features_validation.npz), SHA256 `{report['feature_sha256']}`.", "",
                         "Next action: consume the actual preserved manifest/CSV paths for target-coverage and local `vcc prep` packaging; do not rerun completed ingestion. No submission was performed."]
            (PROJECT / "reports/arc/official_bundle_audit.md").write_text("\n".join(markdown)+"\n")
            (PROJECT / "reports/arc/official_context_characterization.md").write_text("\n".join(markdown)+"\n")
            print(json.dumps({k:report[k] for k in ("status", "wall_seconds", "max_rss_KiB", "feature_bytes", "feature_sha256")}, indent=2))
    except BaseException as error:
        write_json(PROJECT / "reports/arc/official_bundle_audit_failure.json", {"status": "FAILED", "error": f"{type(error).__name__}: {error}", "wall_seconds": time.monotonic()-started, "max_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()

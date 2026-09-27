"""Versioned measurement adapters. Source declarations are not clinical validation.

No source project imports, model training, clinical outcome joins or network I/O.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

from .core import DependencyRef, normalized_time, time_value
from .io import digest, file_hash
from .temporal import EvidenceRecord, Snapshot


@dataclass(frozen=True)
class PhenotypeRegion:
    region_id: str
    parent_region: str | None
    region_type: str


@dataclass(frozen=True)
class PhenotypeFeature:
    name: str
    value: float
    unit: str
    quantity: str
    support: str = 'unknown'

    def __post_init__(self):
        if not self.name or not self.unit or not math.isfinite(self.value):
            raise ValueError('feature requires name, unit and finite value')
        if self.quantity not in {'area', 'count', 'pixel', 'embedding', 'density', 'morphology'}:
            raise ValueError('unknown measurement quantity')
        if self.support not in {'supported', 'unsupported', 'unknown'}:
            raise ValueError('support must be explicit per quantity')


@dataclass(frozen=True)
class PhenotypeMeasurement:
    measurement_id: str
    subject_id: str
    context_id: str
    organism: str
    assay_type: str
    source_asset_ref: str
    source_sha256: str
    annotation_ref: str | None
    annotation_sha256: str | None
    annotation_version: str | None
    phenotype_features: tuple[PhenotypeFeature, ...]
    regions: tuple[PhenotypeRegion, ...]
    measurement_method: str
    measurement_version: str
    record_kind: str
    uncertainty: float | None
    quality_flags: tuple[str, ...]
    observed_at: str | None
    public_available_at: str | None
    ingested_at: str | None
    limitations: tuple[str, ...]

    def __post_init__(self):
        for key in ('measurement_id', 'subject_id', 'context_id', 'organism', 'assay_type',
                    'source_asset_ref', 'measurement_method', 'measurement_version'):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError('missing '+key)
        if self.record_kind not in {'OBSERVED', 'PREDICTED', 'INFERRED', 'SYNTHETIC'}:
            raise ValueError('invalid record kind')
        DependencyRef(self.source_asset_ref, self.source_sha256, self.public_available_at, 'data')
        if self.annotation_ref is not None:
            if not self.annotation_sha256 or not self.annotation_version:
                raise ValueError('annotation must retain hash and version')
            DependencyRef(self.annotation_ref, self.annotation_sha256, None, 'data')
        elif self.annotation_sha256 is not None or self.annotation_version is not None:
            raise ValueError('annotation metadata without annotation source')
        if self.uncertainty is not None and (not math.isfinite(self.uncertainty) or self.uncertainty < 0):
            raise ValueError('uncertainty must be nonnegative or unknown')
        for key in ('observed_at', 'public_available_at', 'ingested_at'):
            object.__setattr__(self, key, normalized_time(getattr(self, key)))
        for key in ('phenotype_features', 'regions', 'quality_flags', 'limitations'):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if not all(isinstance(f, PhenotypeFeature) for f in self.phenotype_features):
            raise ValueError('typed phenotype features required')
        if len({f.name for f in self.phenotype_features}) != len(self.phenotype_features):
            raise ValueError('duplicate feature names')
        ids = {r.region_id for r in self.regions}
        if len(ids) != len(self.regions) or any(not r.region_id or not r.region_type for r in self.regions):
            raise ValueError('invalid or duplicate region')
        parents = {r.region_id: r.parent_region for r in self.regions}
        for r in self.regions:
            seen = {r.region_id}; parent = r.parent_region
            while parent is not None:
                if parent not in ids or parent in seen:
                    raise ValueError('region hierarchy is dangling or cyclic')
                seen.add(parent); parent = parents[parent]

    def to_dict(self):
        return asdict(self)


def measurement_decision(record, quantity):
    relevant = [f for f in record.phenotype_features if f.quantity == quantity]
    if not relevant or any(f.support != 'supported' for f in relevant):
        return 'ABSTAIN_UNSUPPORTED_QUANTITY'
    if record.record_kind != 'OBSERVED':
        return 'DEFER_DERIVED_OR_SYNTHETIC_MEASUREMENT'
    return 'ELIGIBLE_MEASUREMENT_NOT_BIOLOGICAL_BENEFIT'


def as_of(records, public_cutoff, ingestion_cutoff):
    """Return known versions, never silently substitute a later annotation."""
    records = tuple(records)
    if len({r.measurement_id for r in records}) != len(records):
        raise ValueError('measurement identifiers are append-only and unique')
    return tuple(r for r in records if r.public_available_at is not None and r.ingested_at is not None
                 and time_value(r.public_available_at) <= time_value(public_cutoff)
                 and time_value(r.ingested_at) <= time_value(ingestion_cutoff))


def measurement_snapshot(record):
    """Bridge inference conservatively to predicted evidence; never to observation."""
    h = digest(record.to_dict())
    dep = DependencyRef(record.measurement_id, h, record.public_available_at, 'data')
    evidence = EvidenceRecord('measurement:'+record.measurement_id, record.subject_id,
        'has_phenotype_measurement', record.measurement_id, record.context_id, 'supports',
        record.public_available_at, record.ingested_at, h, event_time=record.observed_at,
        dependencies=(record.measurement_id,), synthetic=record.record_kind == 'SYNTHETIC',
        state_kind='observed' if record.record_kind == 'OBSERVED' else 'predicted')
    return Snapshot('phenotype:'+record.measurement_id, (evidence,), (dep,))


class PathologyImageAdapter:
    """Measure a bounded 2-D label mask; count reliability is independently declared.

    manifest is a trusted, separately hashed declaration. The source kind cannot
    be overridden at measurement time. Pixel/area values describe the mask, not
    segmentation accuracy or experimentally established biology.
    """
    version = 'mask-label-measurement-v1'

    def measure(self, image, labels, manifest):
        import numpy as np
        image, labels = np.asarray(image), np.asarray(labels)
        if labels.ndim != 2 or labels.size == 0 or image.shape[:2] != labels.shape:
            raise ValueError('image and mask must be nonempty and co-registered')
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError('mask requires nonnegative integer labels')
        if digest(image.tolist()) != manifest['image_content_sha256'] or digest(labels.tolist()) != manifest['mask_content_sha256']:
            raise ValueError('source image or annotation content mismatch')
        kind = manifest['source_kind']
        if kind not in {'OBSERVED', 'PREDICTED', 'INFERRED', 'SYNTHETIC'}:
            raise ValueError('unknown source kind')
        # Automated annotations on a real image are derived measurements.
        if kind == 'OBSERVED' and manifest['annotation_kind'] != 'manual_observation':
            kind = 'INFERRED'
        ids = np.unique(labels); ids = ids[ids > 0]
        features = (
            PhenotypeFeature('foreground_fraction', float(np.mean(labels > 0)), 'fraction', 'area', manifest['area_support']),
            PhenotypeFeature('label_instance_count', float(len(ids)), 'instances', 'count',
                             manifest['count_support'] if manifest['instance_labels'] else 'unsupported'))
        regions = (PhenotypeRegion('image', None, 'image'),) + tuple(
            PhenotypeRegion('label:'+str(int(i)), 'image', 'mask_region') for i in ids)
        return PhenotypeMeasurement(manifest['measurement_id'], manifest['subject_id'], manifest['context_id'],
            manifest['organism'], 'image_mask', manifest['image_ref'], manifest['image_content_sha256'],
            manifest['mask_ref'], manifest['mask_content_sha256'], manifest['annotation_version'], features,
            regions, self.version, manifest['measurement_version'], kind, manifest['uncertainty'],
            tuple(manifest['quality_flags']), manifest['observed_at'], manifest['public_available_at'],
            manifest['ingested_at'], ('Pixel fraction does not establish area accuracy.',
            'Distinct labels are not necessarily biological instances.', 'No clinical or causal validation.'))

    def certificate(self, record, manifest):
        return {'algorithm': self.version, 'measurement_sha256': digest(record.to_dict()),
                'trusted_manifest_sha256': digest(manifest), 'record_kind': record.record_kind}

    def replay(self, certificate, record, image, labels, trusted_manifest, trusted_manifest_hash):
        if digest(trusted_manifest) != trusted_manifest_hash:
            return False
        recomputed = self.measure(image, labels, trusted_manifest)
        return certificate == self.certificate(recomputed, trusted_manifest) and record == recomputed


class PathologyEmbeddingAdapter:
    """Bounded P09 real-asset intake, without expression/outcome joins.

    Embedding summaries are inferred representation features. They are neither
    measured cancer-cell counts nor a validated molecular/clinical phenotype.
    """
    def measure(self, embeddings_path, *, row, subject_id, ingested_at):
        import numpy as np
        matrix = np.load(embeddings_path, mmap_mode='r', allow_pickle=False)
        if matrix.ndim != 2 or not 0 <= row < matrix.shape[0]:
            raise ValueError('invalid embedding matrix or row')
        values = np.asarray(matrix[row], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError('nonfinite embedding')
        return PhenotypeMeasurement('P09:embedding:'+str(row)+':v1', subject_id, 'TCGA-BRCA:histopathology',
            'Homo sapiens', 'image_derived_embedding', str(Path(embeddings_path).resolve()),
            file_hash(embeddings_path), None, None, None,
            (PhenotypeFeature('embedding_l2', float(np.linalg.norm(values)), 'embedding_units', 'embedding'),),
            (), 'P09_embedding_intake', 'v1', 'INFERRED', None,
            ('raw_image_link_not_reconstructed', 'not_a_count_or_area_measurement', 'license_review_pending'),
            None, None, ingested_at, ('No outcome, expression, survival or treatment label loaded.',
            'No cell-level ground truth; no causal or clinical validation.',
            'Source preprocessing and training overlap not independently reproduced.'))

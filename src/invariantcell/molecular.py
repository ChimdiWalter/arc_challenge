"""Original molecular intake contracts; no source imports or biological inference.

Hashes and declarations establish computational identity, not measured accuracy.
Coordinates must be hashed independently of residue names to detect relabeling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .core import DependencyRef, normalized_time
from .io import digest
from .phenotype import PhenotypeMeasurement
from .temporal import EvidenceRecord, Snapshot


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
RECORD_KINDS = frozenset({"OBSERVED", "PREDICTED", "INFERRED", "SYNTHETIC"})


def coordinate_hash(coordinates):
    """Hash ordered XYZ values only, excluding residue/variant label metadata.

    Exact coordinate identity is not rotation-invariant structural identity.
    Atom correspondence, coordinate units and ordering require upstream review.
    """
    rows = tuple(tuple(row) for row in coordinates)
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError("coordinates require nonempty XYZ rows")
    if any(type(x) not in (int, float) or not math.isfinite(x) for row in rows for x in row):
        raise ValueError("coordinates must be finite numbers")
    return digest([[float(x) for x in row] for row in rows])


def _names(record, *fields):
    for field in fields:
        value = getattr(record, field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("missing " + field)


def _kind(value):
    if value not in RECORD_KINDS:
        raise ValueError("unknown record kind")


@dataclass(frozen=True)
class ResidueMapping:
    """Explicit mapping; author numbering/insertion codes are not sequence indices."""
    chain_id: str
    author_residue_number: int
    insertion_code: str
    sequence_position: int
    amino_acid: str

    def __post_init__(self):
        _names(self, "chain_id")
        if type(self.author_residue_number) is not int or type(self.sequence_position) is not int:
            raise ValueError("residue numbers and positions must be integers")
        if self.sequence_position < 1 or self.amino_acid not in AMINO_ACIDS:
            raise ValueError("invalid sequence position or canonical residue")
        if not isinstance(self.insertion_code, str) or len(self.insertion_code) > 1:
            raise ValueError("insertion code must be zero or one character")


@dataclass(frozen=True)
class MolecularConstruct:
    construct_id: str
    protein_id: str
    context_id: str
    organism: str
    sequence: str
    sequence_source: DependencyRef
    mapping_version: str
    residue_mapping: tuple[ResidueMapping, ...] = ()
    synthetic: bool = False

    def __post_init__(self):
        _names(self, "construct_id", "protein_id", "context_id", "organism", "mapping_version")
        if not isinstance(self.sequence, str) or not self.sequence or set(self.sequence) - AMINO_ACIDS:
            raise ValueError("sequence must use canonical uppercase amino acids")
        if not isinstance(self.sequence_source, DependencyRef) or type(self.synthetic) is not bool:
            raise ValueError("typed sequence provenance and synthetic flag required")
        mapping = tuple(self.residue_mapping)
        if not all(isinstance(row, ResidueMapping) for row in mapping):
            raise ValueError("typed residue mapping required")
        keys = [(r.chain_id, r.author_residue_number, r.insertion_code) for r in mapping]
        positions = [(r.chain_id, r.sequence_position) for r in mapping]
        if len(set(keys)) != len(keys) or len(set(positions)) != len(positions):
            raise ValueError("duplicate structural residue or sequence position within chain")
        for r in mapping:
            if r.sequence_position > len(self.sequence) or self.sequence[r.sequence_position - 1] != r.amino_acid:
                raise ValueError("residue mapping disagrees with construct sequence")
        object.__setattr__(self, "residue_mapping", mapping)

    @property
    def sequence_sha256(self):
        return digest(self.sequence)


@dataclass(frozen=True)
class MolecularIntervention:
    intervention_id: str
    reference: MolecularConstruct
    variant: MolecularConstruct
    position: int
    reference_amino_acid: str
    alternate_amino_acid: str
    modality: str = "single_substitution"

    def __post_init__(self):
        _names(self, "intervention_id")
        if not isinstance(self.reference, MolecularConstruct) or not isinstance(self.variant, MolecularConstruct):
            raise ValueError("typed reference and variant constructs required")
        if self.modality != "single_substitution":
            raise ValueError("only explicit single substitutions are supported")
        if self.reference.construct_id == self.variant.construct_id:
            raise ValueError("WT and variant must have distinct immutable identities")
        for field in ("protein_id", "context_id", "organism", "synthetic"):
            if getattr(self.reference, field) != getattr(self.variant, field):
                raise ValueError("WT/variant " + field + " mismatch")
        if type(self.position) is not int or not 1 <= self.position <= len(self.reference.sequence):
            raise ValueError("substitution position outside reference sequence")
        if (self.reference_amino_acid not in AMINO_ACIDS or self.alternate_amino_acid not in AMINO_ACIDS
                or self.reference_amino_acid == self.alternate_amino_acid):
            raise ValueError("explicit distinct canonical substitution residues required")
        if self.reference.sequence[self.position - 1] != self.reference_amino_acid:
            raise ValueError("substitution reference residue is incorrect")
        expected = (self.reference.sequence[:self.position - 1] + self.alternate_amino_acid
                    + self.reference.sequence[self.position:])
        if self.variant.sequence != expected:
            raise ValueError("variant sequence does not match the declared substitution")


@dataclass(frozen=True)
class ConformerEvidence:
    conformer_id: str
    construct_id: str
    sequence_sha256: str
    coordinate_sha256: str
    source: DependencyRef
    method: str
    method_version: str
    record_kind: str
    uses_variant_sequence: bool = False

    def __post_init__(self):
        _names(self, "conformer_id", "construct_id", "method", "method_version")
        _kind(self.record_kind)
        DependencyRef("sequence", self.sequence_sha256, None)
        DependencyRef("coordinates", self.coordinate_sha256, None)
        if not isinstance(self.source, DependencyRef) or type(self.uses_variant_sequence) is not bool:
            raise ValueError("typed source and explicit sequence-use declaration required")


def mutation_sensitivity(intervention, reference, variant):
    """Input audit, never a scientific validation of mutation-sensitive behavior."""
    for conformer, construct in ((reference, intervention.reference), (variant, intervention.variant)):
        if conformer.construct_id != construct.construct_id or conformer.sequence_sha256 != construct.sequence_sha256:
            raise ValueError("conformer is bound to the wrong construct or sequence")
    if reference.coordinate_sha256 == variant.coordinate_sha256:
        return "SAME_COORDINATES_NOT_MUTATION_AWARE"
    if not variant.uses_variant_sequence:
        return "MUTATION_SENSITIVITY_UNSUPPORTED"
    return "MUTATION_SPECIFIC_INPUTS_DECLARED_NOT_VALIDATED"


@dataclass(frozen=True)
class ExperimentalConstraint:
    """Acquisition ID and endpoint identify raw measurements before preprocessing."""
    acquisition_id: str
    endpoint: str
    construct_id: str
    context_id: str
    source: DependencyRef
    record_kind: str

    def __post_init__(self):
        _names(self, "acquisition_id", "endpoint", "construct_id", "context_id")
        _kind(self.record_kind)
        if not isinstance(self.source, DependencyRef):
            raise ValueError("typed endpoint provenance required")


def validate_independent_endpoints(fitting, independent_validation):
    """Conservative provenance check; disjointness does not prove independence."""
    fitting, independent_validation = tuple(fitting), tuple(independent_validation)
    if any(not isinstance(row, ExperimentalConstraint) for row in (*fitting, *independent_validation)):
        raise ValueError("typed endpoint records required")
    for fit in fitting:
        for validation in independent_validation:
            if ((fit.acquisition_id, fit.endpoint) == (validation.acquisition_id, validation.endpoint)
                    or fit.source.sha256 == validation.source.sha256
                    or fit.source.dependency_id == validation.source.dependency_id):
                raise ValueError("fitting endpoint cannot be independent validation, including aliases")
    if not independent_validation:
        return "NO_INDEPENDENT_VALIDATION"
    if any(row.record_kind != "OBSERVED" for row in independent_validation):
        return "NONOBSERVED_VALIDATION_ENGINEERING_ONLY"
    return "DISJOINT_DECLARED_OBSERVATIONS_NOT_CAUSAL_VALIDATION"


@dataclass(frozen=True)
class MolecularStateEnsemble:
    ensemble_id: str
    construct: MolecularConstruct
    conformers: tuple[ConformerEvidence, ...]
    weights: tuple[float, ...]
    record_kind: str
    method: str
    method_version: str
    uncertainty: float | None = None
    fitting_constraints: tuple[ExperimentalConstraint, ...] = ()
    independent_validation: tuple[ExperimentalConstraint, ...] = ()
    event_time: str | None = None
    public_available_at: str | None = None
    ingested_at: str | None = None

    def __post_init__(self):
        _names(self, "ensemble_id", "method", "method_version")
        if not isinstance(self.construct, MolecularConstruct):
            raise ValueError("typed construct required")
        _kind(self.record_kind)
        if self.record_kind == "OBSERVED":
            raise ValueError("computed ensembles cannot be promoted to observations")
        for field in ("conformers", "weights", "fitting_constraints", "independent_validation"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if not self.conformers or any(not isinstance(c, ConformerEvidence) for c in self.conformers):
            raise ValueError("nonempty typed conformers required")
        if len({c.conformer_id for c in self.conformers}) != len(self.conformers):
            raise ValueError("duplicate conformer identity")
        for c in self.conformers:
            if c.construct_id != self.construct.construct_id or c.sequence_sha256 != self.construct.sequence_sha256:
                raise ValueError("ensemble conformer does not match the construct sequence")
        if self.construct.synthetic or any(c.record_kind == "SYNTHETIC" for c in self.conformers):
            if self.record_kind != "SYNTHETIC":
                raise ValueError("synthetic provenance must remain synthetic")
        if len(self.weights) != len(self.conformers) or any(
                type(w) not in (float, int) or not math.isfinite(w) or w < 0 for w in self.weights):
            raise ValueError("nonnegative finite conformer weights required")
        if not math.isclose(sum(self.weights), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("ensemble weights must sum to one")
        if self.uncertainty is not None and (type(self.uncertainty) not in (float, int)
                or not math.isfinite(self.uncertainty) or self.uncertainty < 0):
            raise ValueError("uncertainty must be nonnegative finite or unknown")
        validate_independent_endpoints(self.fitting_constraints, self.independent_validation)
        for endpoint in (*self.fitting_constraints, *self.independent_validation):
            if endpoint.construct_id != self.construct.construct_id or endpoint.context_id != self.construct.context_id:
                raise ValueError("assay endpoint does not match the ensemble construct/context")
            if endpoint.record_kind == "SYNTHETIC" and self.record_kind != "SYNTHETIC":
                raise ValueError("synthetic constraints cannot support a real ensemble claim")
        for field in ("event_time", "public_available_at", "ingested_at"):
            object.__setattr__(self, field, normalized_time(getattr(self, field)))

    def to_dict(self):
        return asdict(self)


def molecular_snapshot(ensemble):
    """Link the immutable ensemble artifact without asserting a molecular effect."""
    if not isinstance(ensemble, MolecularStateEnsemble):
        raise ValueError("typed molecular ensemble required")
    dependencies = {}
    refs = [ensemble.construct.sequence_source, *(c.source for c in ensemble.conformers),
            *(e.source for e in (*ensemble.fitting_constraints, *ensemble.independent_validation))]
    own_hash = digest(ensemble.to_dict())
    refs.append(DependencyRef(ensemble.ensemble_id, own_hash, ensemble.public_available_at, "data"))
    for ref in refs:
        if ref.dependency_id in dependencies and dependencies[ref.dependency_id] != ref:
            raise ValueError("conflicting molecular dependency")
        dependencies[ref.dependency_id] = ref
    evidence = EvidenceRecord("ensemble:" + ensemble.ensemble_id, ensemble.construct.construct_id,
        "has_molecular_ensemble_record", ensemble.ensemble_id, ensemble.construct.context_id,
        "supports", ensemble.public_available_at, ensemble.ingested_at, own_hash,
        event_time=ensemble.event_time, dependencies=tuple(sorted(dependencies)),
        synthetic=ensemble.record_kind == "SYNTHETIC", state_kind="predicted")
    return Snapshot("molecular:" + ensemble.ensemble_id, (evidence,), tuple(dependencies.values()))


def phenotype_bridge_status(ensemble, intervention, measurement, *, measurement_intervention_id,
                            measurement_construct_id):
    """Audit matching declarations. No return value establishes a causal bridge."""
    if not isinstance(measurement, PhenotypeMeasurement):
        raise ValueError("typed phenotype measurement required")
    if (ensemble.construct != intervention.variant
            or measurement_intervention_id != intervention.intervention_id
            or measurement_construct_id != intervention.variant.construct_id
            or measurement.context_id != ensemble.construct.context_id
            or measurement.organism != ensemble.construct.organism):
        return "ABSTAIN_UNMATCHED_INTERVENTION_CONSTRUCT_OR_CONTEXT"
    if ensemble.record_kind == "SYNTHETIC" or measurement.record_kind == "SYNTHETIC":
        return "SYNTHETIC_BRIDGE_ENGINEERING_ONLY"
    if measurement.record_kind != "OBSERVED":
        return "ABSTAIN_NO_OBSERVED_PHENOTYPE"
    return "MATCHED_DECLARATIONS_CAUSALITY_NOT_ESTABLISHED"

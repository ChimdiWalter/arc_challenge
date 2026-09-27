"""Small, original count baselines and immutable shared records.

These helpers do not read files or run experiments. The miniature validator is
an internal engineering contract, not the Arc evaluator or submission contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import random
from typing import Iterable


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def normalized_time(value: str | None) -> str | None:
    """Unknown stays unknown; supplied times must have an explicit timezone."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string or None")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp requires an explicit timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def time_value(value: str) -> datetime:
    return datetime.fromisoformat(normalized_time(value).replace("Z", "+00:00"))


def _name(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


@dataclass(frozen=True)
class DependencyRef:
    dependency_id: str
    sha256: str
    available_at: str | None
    kind: str = "data"

    def __post_init__(self) -> None:
        _name(self.dependency_id, "dependency_id")
        if (not isinstance(self.sha256, str) or len(self.sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("dependency sha256 must be 64 lowercase hexadecimal characters")
        if self.kind not in {"data", "source", "model", "rule", "mapping", "ontology"}:
            raise ValueError("unsupported dependency kind")
        object.__setattr__(self, "available_at", normalized_time(self.available_at))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BiologicalTransition:
    record_id: str
    context: str
    perturbation: str | None
    genes: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]
    state_kind: str = "observed"
    synthetic: bool = False
    event_time: str | None = None
    evidence_available_at: str | None = None
    ingested_at: str | None = None
    dependencies: tuple[DependencyRef, ...] = ()
    method: str | None = None

    def __post_init__(self) -> None:
        _name(self.record_id, "record_id")
        _name(self.context, "context")
        if self.perturbation is not None:
            _name(self.perturbation, "perturbation")
        if self.state_kind not in {"observed", "predicted"}:
            raise ValueError("state_kind must distinguish observed from predicted")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic must be boolean")
        genes = tuple(self.genes)
        if not genes or len(set(genes)) != len(genes):
            raise ValueError("genes must be nonempty and unique")
        for gene in genes:
            _name(gene, "gene")
        counts = tuple(tuple(row) for row in self.counts)
        if not counts or any(len(row) != len(genes) for row in counts):
            raise ValueError("counts must be nonempty, rectangular, and match gene order")
        if any(type(value) is not int or value < 0 for row in counts for value in row):
            raise ValueError("raw counts must be nonnegative integers (not bools or floats)")
        dependencies = tuple(self.dependencies)
        if any(not isinstance(item, DependencyRef) for item in dependencies):
            raise ValueError("dependencies must be DependencyRef records")
        if len({item.dependency_id for item in dependencies}) != len(dependencies):
            raise ValueError("duplicate dependency identifiers")
        object.__setattr__(self, "genes", genes)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "dependencies", dependencies)
        for field in ("event_time", "evidence_available_at", "ingested_at"):
            object.__setattr__(self, field, normalized_time(getattr(self, field)))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> BiologicalTransition:
        payload = dict(value)
        payload["dependencies"] = tuple(DependencyRef(**item) for item in payload.get("dependencies", ()))
        return cls(**payload)


def _control(control: BiologicalTransition) -> None:
    if control.state_kind != "observed" or control.perturbation is not None:
        raise ValueError("baseline input must be an observed control group")


def _input_references(records: Iterable[BiologicalTransition]) -> tuple[DependencyRef, ...]:
    refs: dict[str, DependencyRef] = {}
    for record in records:
        reference = DependencyRef(record.record_id, canonical_hash(record.to_dict()),
                                  record.evidence_available_at)
        for item in (*record.dependencies, reference):
            if item.dependency_id in refs and refs[item.dependency_id] != item:
                raise ValueError("conflicting provenance dependency")
            refs[item.dependency_id] = item
    return tuple(refs[key] for key in sorted(refs))


def no_effect(control: BiologicalTransition, target: str) -> BiologicalTransition:
    """Use held-out controls unchanged; output remains a prediction with no event time."""
    _control(control)
    _name(target, "target")
    return BiologicalTransition(
        record_id=f"no_effect:{control.record_id}:{target}", context=control.context,
        perturbation=target, genes=control.genes, counts=control.counts,
        state_kind="predicted", synthetic=control.synthetic, event_time=None,
        dependencies=_input_references((control,)), method="no_effect",
    )


def _means(record: BiologicalTransition) -> tuple[float, ...]:
    return tuple(sum(row[i] for row in record.counts) / len(record.counts)
                 for i in range(len(record.genes)))


def pooled_delta(training: Iterable[BiologicalTransition], control: BiologicalTransition,
                 target: str, shrinkage: float = 1.0) -> BiologicalTransition:
    """Average observed training-context deltas; shrink toward no effect.

    Training contexts must be disjoint from the held-out control context. Unknown
    targets use no effect. Rounding/clipping produces integer counts but is not a
    calibrated single-cell distribution or an official Arc-quality model.
    """
    _control(control)
    _name(target, "target")
    if isinstance(shrinkage, bool) or not isinstance(shrinkage, (int, float)) or not math.isfinite(shrinkage) or shrinkage < 0:
        raise ValueError("shrinkage must be finite and nonnegative")
    training = tuple(training)
    groups: dict[tuple[str, str | None], BiologicalTransition] = {}
    for record in training:
        if record.state_kind != "observed":
            raise ValueError("training accepts observed groups only")
        if record.context == control.context:
            raise ValueError("held-out context cannot appear in training")
        if record.genes != control.genes:
            raise ValueError("training genes and order must match held-out controls")
        if record.synthetic != control.synthetic:
            raise ValueError("synthetic and real groups cannot be mixed")
        key = (record.context, record.perturbation)
        if key in groups:
            raise ValueError("duplicate context/perturbation group")
        groups[key] = record
    target_groups = [record for record in training if record.perturbation == target]
    if not target_groups:
        return replace(no_effect(control, target), method="pooled_delta:unknown_target:no_effect")
    deltas: list[tuple[float, ...]] = []
    used: list[BiologicalTransition] = [control]
    for record in target_groups:
        baseline = groups.get((record.context, None))
        if baseline is None:
            raise ValueError("each target training group needs matching context controls")
        deltas.append(tuple(a - b for a, b in zip(_means(record), _means(baseline))))
        used.extend((record, baseline))
    weight = len(deltas) / (len(deltas) + shrinkage)
    delta = tuple(weight * sum(row[i] for row in deltas) / len(deltas)
                  for i in range(len(control.genes)))
    counts = tuple(tuple(max(0, math.floor(value + change + 0.5))
                         for value, change in zip(row, delta)) for row in control.counts)
    return replace(no_effect(control, target), record_id=f"pooled_delta:{control.record_id}:{target}",
                   counts=counts, dependencies=_input_references(used),
                   method=f"pooled_delta:shrinkage={shrinkage:g}")


def miniature_fixture(seed: int = 17, contexts: int = 3, perturbations: int = 6,
                      genes: int = 32, cells_per_group: int = 24) -> tuple[BiologicalTransition, ...]:
    """Generate clearly synthetic internal fixtures; calling this is not a real-data pilot."""
    if any(type(v) is not int or v < 1 for v in (contexts, perturbations, genes, cells_per_group)):
        raise ValueError("fixture dimensions must be positive integers")
    rng = random.Random(seed)
    names = tuple(f"SYNTHETIC_GENE_{g}" for g in range(genes))
    records = []
    for context in range(contexts):
        for target in [None, *range(perturbations)]:
            counts = tuple(tuple(max(0, 3 + context + gene % 4 + rng.randint(-2, 2)
                                      + (2 if target is not None and gene % perturbations == target else 0))
                                 for gene in range(genes)) for _ in range(cells_per_group))
            records.append(BiologicalTransition(
                record_id=f"SYNTHETIC:{context}:{target}", context=f"SYNTHETIC_CONTEXT_{context}",
                perturbation=None if target is None else f"SYNTHETIC_TARGET_{target}",
                genes=names, counts=counts, synthetic=True,
            ))
    return tuple(records)


def validate_miniature_panel(panel: Iterable[BiologicalTransition], expected_genes: Iterable[str],
                             expected_context_targets: Iterable[tuple[str, str]],
                             cells_per_group: int) -> dict:
    """Validate internal coverage, order, integer counts, and exclusion of controls."""
    panel = tuple(panel)
    genes = tuple(expected_genes)
    if type(cells_per_group) is not int or cells_per_group < 1:
        raise ValueError("cells_per_group must be a positive integer")
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("expected genes must be nonempty and unique")
    expected_list = tuple(expected_context_targets)
    expected = set(expected_list)
    if not expected or len(expected) != len(expected_list):
        raise ValueError("expected panel keys must be nonempty and unique")
    actual = set()
    for record in panel:
        # Reconstruct so callers cannot smuggle a malformed unvalidated record.
        BiologicalTransition.from_dict(record.to_dict())
        if record.state_kind != "predicted" or record.perturbation is None:
            raise ValueError("miniature prediction panels exclude observed groups and controls")
        if record.genes != genes or len(record.counts) != cells_per_group:
            raise ValueError("gene order or cell count does not match miniature contract")
        key = (record.context, record.perturbation)
        if key in actual:
            raise ValueError("duplicate prediction group")
        actual.add(key)
    if actual != expected:
        raise ValueError("prediction coverage differs from expected miniature panel")
    if len({record.synthetic for record in panel}) != 1:
        raise ValueError("miniature panels cannot mix synthetic and real predictions")
    return {"valid": True, "scope": "INTERNAL_MINIATURE_ONLY", "official_arc_validation": False,
            "groups": len(panel), "genes": len(genes), "cells_per_group": cells_per_group,
            "synthetic": all(record.synthetic for record in panel)}

"""Deterministic evidence filtering, with independent public and ingestion axes.

Certificates attest to replayable checks over caller-supplied data. They do not
prove biological truth, completeness, or formal OWL entailment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Literal
from .core import BiologicalTransition, DependencyRef, canonical_hash, normalized_time, time_value, _name

ALGORITHM = "invariantcell-temporal-filter-v3"
LIMITATIONS = (
    "Checks consistency of supplied records; does not establish experimental truth.",
    "Caller must independently establish source integrity and snapshot completeness.",
    "This is deterministic filtering, not formal OWL reasoning or a learned biological procedure.",
    "Unknown event times remain unknown; an event-time cutoff excludes them.",
    "Unknown validity excludes a valid-at answer; open bounds require an explicit interval assertion.",
    "Observed support describes supplied observed assertions, not independently established biological truth.",
)


@dataclass(frozen=True)
class EvidenceRecord:
    """An assertion whose biological validity is unknown unless explicitly stated.

    ``validity_kind=interval`` declares a [valid_from, valid_to) interval; only
    with that declaration do null endpoints denote deliberately open bounds.
    Public source availability, artifact generation, and local ingestion never
    supply missing event dates or biological validity bounds.
    """
    evidence_id: str
    subject: str
    predicate: str
    value: str
    context: str
    polarity: str
    public_available_at: str | None
    ingested_at: str | None
    source_sha256: str
    valid_from: str | None = None
    valid_to: str | None = None
    event_time: str | None = None
    dependencies: tuple[str, ...] = ()
    synthetic: bool = False
    state_kind: str = "observed"
    validity_kind: str = "unknown"
    generated_at: str | None = None
    transition_record_id: str | None = None
    perturbation: str | None = None

    def __post_init__(self) -> None:
        for field in ("evidence_id", "subject", "predicate", "value", "context"):
            _name(getattr(self, field), field)
        if self.polarity not in {"supports", "refutes"}:
            raise ValueError("polarity must be supports or refutes")
        DependencyRef(self.evidence_id, self.source_sha256, self.public_available_at, "source")
        for field in ("public_available_at", "ingested_at", "valid_from", "valid_to", "event_time", "generated_at"):
            object.__setattr__(self, field, normalized_time(getattr(self, field)))
        if self.validity_kind not in {"unknown", "interval"}:
            raise ValueError("validity_kind must be unknown or interval")
        if self.validity_kind == "unknown" and (self.valid_from is not None or self.valid_to is not None):
            raise ValueError("bounds require an explicit interval validity assertion")
        if self.valid_from and self.valid_to and time_value(self.valid_from) >= time_value(self.valid_to):
            raise ValueError("valid interval must have positive length [from, to)")
        deps = tuple(self.dependencies)
        if len(set(deps)) != len(deps):
            raise ValueError("duplicate evidence dependency")
        for dep in deps:
            _name(dep, "dependency")
        object.__setattr__(self, "dependencies", deps)
        if type(self.synthetic) is not bool or self.state_kind not in {"observed", "predicted"}:
            raise ValueError("invalid evidence state")
        for field in ("transition_record_id", "perturbation"):
            if getattr(self, field) is not None:
                _name(getattr(self, field), field)
        if self.state_kind == "predicted" and self.generated_at is not None:
            for field in ("public_available_at", "ingested_at"):
                available = getattr(self, field)
                if available is not None and time_value(available) < time_value(self.generated_at):
                    raise ValueError("prediction availability cannot precede artifact generation")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    evidence: tuple[EvidenceRecord, ...]
    dependencies: tuple[DependencyRef, ...] = ()

    def __post_init__(self) -> None:
        _name(self.snapshot_id, "snapshot_id")
        evidence = tuple(self.evidence)
        dependencies = tuple(self.dependencies)
        if any(not isinstance(item, EvidenceRecord) for item in evidence):
            raise ValueError("snapshot evidence requires EvidenceRecord records")
        if any(not isinstance(item, DependencyRef) for item in dependencies):
            raise ValueError("snapshot dependencies require DependencyRef records")
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise ValueError("duplicate evidence identifier")
        dep_ids = {item.dependency_id for item in dependencies}
        if len(dep_ids) != len(dependencies):
            raise ValueError("duplicate snapshot dependency")
        if any(dep not in dep_ids for item in evidence for dep in item.dependencies):
            raise ValueError("unresolved snapshot dependency")
        object.__setattr__(self, "evidence", tuple(sorted(evidence, key=lambda x: x.evidence_id)))
        object.__setattr__(self, "dependencies", tuple(sorted(dependencies, key=lambda x: x.dependency_id)))

    def to_dict(self) -> dict:
        return {"snapshot_id": self.snapshot_id, "evidence": [item.to_dict() for item in self.evidence],
                "dependencies": [item.to_dict() for item in self.dependencies]}

    @classmethod
    def from_dict(cls, value: dict) -> Snapshot:
        if set(value) != {"snapshot_id", "evidence", "dependencies"}:
            raise ValueError("unexpected snapshot fields")
        return cls(value["snapshot_id"], tuple(EvidenceRecord(**item) for item in value["evidence"]),
                   tuple(DependencyRef(**item) for item in value["dependencies"]))


def transition_to_evidence(transition: BiologicalTransition, *, generated_at: str | None = None,
                           public_available_at: str | None = None, ingested_at: str | None = None,
                           validity_kind: str = "unknown", valid_from: str | None = None,
                           valid_to: str | None = None) -> EvidenceRecord:
    """Bridge a count record to a provenance assertion, not a biological finding.

    The predicate ``has_count_record`` links the perturbation target (or the
    control record identifier when no target exists) to the count record. Its
    source hash covers the complete original transition. Predicted counts remain
    predicted. No pathway effect or empirical benefit is inferred.

    Explicit generated/public/ingested times describe this artifact's creation,
    public availability, and store ingestion. Generation does not imply public
    availability. Missing availability/ingestion values retain the transition's
    existing values, including None; event_time is copied without substitution.
    Validity remains unknown unless the caller explicitly asserts an interval.
    """
    if not isinstance(transition, BiologicalTransition):
        raise ValueError("bridge requires a BiologicalTransition")
    return EvidenceRecord(
        evidence_id=f"transition:{transition.record_id}",
        subject=transition.perturbation if transition.perturbation is not None else transition.record_id,
        predicate="has_count_record", value=transition.record_id, context=transition.context,
        polarity="supports",
        public_available_at=transition.evidence_available_at if public_available_at is None else public_available_at,
        ingested_at=transition.ingested_at if ingested_at is None else ingested_at,
        source_sha256=canonical_hash(transition.to_dict()),
        valid_from=valid_from, valid_to=valid_to, event_time=transition.event_time,
        dependencies=tuple(sorted({transition.record_id, *(item.dependency_id for item in transition.dependencies)})),
        synthetic=transition.synthetic, state_kind=transition.state_kind,
        validity_kind=validity_kind, generated_at=generated_at,
        transition_record_id=transition.record_id, perturbation=transition.perturbation,
    )


def snapshot_from_transitions(transitions: Iterable[BiologicalTransition], snapshot_id: str, *,
                              generated_at: str | None = None, public_available_at: str | None = None,
                              ingested_at: str | None = None, validity_kind: str = "unknown",
                              valid_from: str | None = None, valid_to: str | None = None) -> Snapshot:
    """Build one shared snapshot, preserving all declared source dependencies.

    Explicit lifecycle timestamps and validity declarations apply to every input
    artifact in this batch. Unknown dependency availability is not filled from
    these timestamps. Conflicting hashes or dependency timestamps are rejected.
    """
    evidence = []
    dependencies: dict[str, DependencyRef] = {}
    for transition in transitions:
        record = transition_to_evidence(
            transition, generated_at=generated_at, public_available_at=public_available_at,
            ingested_at=ingested_at, validity_kind=validity_kind,
            valid_from=valid_from, valid_to=valid_to,
        )
        evidence.append(record)
        own_source = DependencyRef(transition.record_id, record.source_sha256,
                                   record.public_available_at, "data")
        for dependency in (*transition.dependencies, own_source):
            previous = dependencies.get(dependency.dependency_id)
            if previous is not None and previous != dependency:
                raise ValueError("conflicting transition snapshot dependency")
            dependencies[dependency.dependency_id] = dependency
    return Snapshot(snapshot_id, tuple(evidence), tuple(dependencies.values()))


@dataclass(frozen=True)
class TemporalQuery:
    subject: str
    predicate: str
    context: str
    evidence_cutoff: str
    ingestion_cutoff: str
    valid_at: str
    event_cutoff: str | None = None
    allow_predictions: bool = False

    def __post_init__(self) -> None:
        for field in ("subject", "predicate", "context"):
            _name(getattr(self, field), field)
        for field in ("evidence_cutoff", "ingestion_cutoff", "valid_at", "event_cutoff"):
            value = normalized_time(getattr(self, field))
            if field != "event_cutoff" and value is None:
                raise ValueError(f"{field} is required")
            object.__setattr__(self, field, value)
        if type(self.allow_predictions) is not bool:
            raise ValueError("allow_predictions must be boolean")

    def to_dict(self) -> dict:
        return asdict(self)


class TemporalOutcome(str, Enum):
    ABSTAIN_UNSUPPORTED = "abstain_unsupported"
    CONFLICT = "conflict"
    REJECTED_DEPENDENCY = "rejected_dependency"
    OBSERVED_SUPPORT = "observed_support"
    PREDICTION_WITH_VALID_PROVENANCE = "prediction_with_valid_provenance"


AnswerStatus = Literal["observed_support", "prediction_with_valid_provenance", "conflicted",
                       "observed_refutation", "prediction_refutation_with_valid_provenance"]


@dataclass(frozen=True)
class TemporalAnswer:
    value: str
    status: AnswerStatus
    support_ids: tuple[str, ...]
    refute_ids: tuple[str, ...]
    support_state_kinds: tuple[str, ...]
    refute_state_kinds: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"value": self.value, "status": self.status,
                "support_ids": list(self.support_ids), "refute_ids": list(self.refute_ids),
                "support_state_kinds": list(self.support_state_kinds),
                "refute_state_kinds": list(self.refute_state_kinds)}


@dataclass(frozen=True)
class TemporalResult:
    answers: tuple[TemporalAnswer, ...]
    excluded: tuple[tuple[str, tuple[str, ...]], ...]
    model_cutoff_violations: tuple[str, ...]
    outcome: TemporalOutcome

    def to_dict(self) -> dict:
        return {"outcome": self.outcome.value,
                "answers": [answer.to_dict() for answer in self.answers],
                "excluded": [{"evidence_id": item, "reasons": list(reasons)} for item, reasons in self.excluded],
                "model_cutoff_violations": list(self.model_cutoff_violations),
                "checks_passed": not self.model_cutoff_violations,
                "algorithm": ALGORITHM, "limitations": list(LIMITATIONS)}


def evaluate(snapshot: Snapshot, query: TemporalQuery) -> TemporalResult:
    dependencies = {item.dependency_id: item for item in snapshot.dependencies}
    cutoff = time_value(query.evidence_cutoff)
    bad_models = tuple(item.dependency_id for item in snapshot.dependencies
                       if item.kind == "model" and (item.available_at is None or time_value(item.available_at) > cutoff))
    included: dict[str, dict[str, list[EvidenceRecord]]] = {}
    excluded = []
    for item in snapshot.evidence:
        reasons = []
        if (item.subject, item.predicate, item.context) != (query.subject, query.predicate, query.context):
            reasons.append("query_or_context_mismatch")
        if item.public_available_at is None or time_value(item.public_available_at) > cutoff:
            reasons.append("public_evidence_after_cutoff_or_unknown")
        if item.ingested_at is None or time_value(item.ingested_at) > time_value(query.ingestion_cutoff):
            reasons.append("ingestion_after_cutoff_or_unknown")
        if item.validity_kind == "unknown":
            reasons.append("unknown_validity")
        if item.valid_from and time_value(item.valid_from) > time_value(query.valid_at):
            reasons.append("before_valid_interval")
        if item.valid_to and time_value(query.valid_at) >= time_value(item.valid_to):
            reasons.append("after_valid_interval")
        if query.event_cutoff and (item.event_time is None or time_value(item.event_time) > time_value(query.event_cutoff)):
            reasons.append("event_after_cutoff_or_unknown")
        if item.state_kind == "predicted" and not query.allow_predictions:
            reasons.append("prediction_not_observation")
        if any(dependencies[dep].available_at is None or time_value(dependencies[dep].available_at) > cutoff
               for dep in item.dependencies):
            reasons.append("dependency_after_cutoff_or_unknown")
        if bad_models:
            reasons.append("snapshot_model_cutoff_violation")
        if reasons:
            excluded.append((item.evidence_id, tuple(reasons)))
        else:
            included.setdefault(item.value, {"supports": [], "refutes": []})[item.polarity].append(item)
    answers = []
    for value, sides in sorted(included.items()):
        support = tuple(item.evidence_id for item in sides["supports"])
        refute = tuple(item.evidence_id for item in sides["refutes"])
        support_kinds = tuple(sorted({item.state_kind for item in sides["supports"]}))
        refute_kinds = tuple(sorted({item.state_kind for item in sides["refutes"]}))
        if support and refute:
            status = "conflicted"
        elif support:
            status = "observed_support" if "observed" in support_kinds else "prediction_with_valid_provenance"
        else:
            status = "observed_refutation" if "observed" in refute_kinds else "prediction_refutation_with_valid_provenance"
        answers.append(TemporalAnswer(value, status, support, refute, support_kinds, refute_kinds))
    # A top-level label summarizes surviving positive answers. Exclusions remain
    # explicit; a valid certificate can correctly certify an abstention/rejection.
    statuses = {answer.status for answer in answers}
    if bad_models:
        outcome = TemporalOutcome.REJECTED_DEPENDENCY
    elif "conflicted" in statuses:
        outcome = TemporalOutcome.CONFLICT
    elif "observed_support" in statuses:
        outcome = TemporalOutcome.OBSERVED_SUPPORT
    elif "prediction_with_valid_provenance" in statuses:
        outcome = TemporalOutcome.PREDICTION_WITH_VALID_PROVENANCE
    elif any("dependency_after_cutoff_or_unknown" in reasons for _, reasons in excluded):
        outcome = TemporalOutcome.REJECTED_DEPENDENCY
    else:
        outcome = TemporalOutcome.ABSTAIN_UNSUPPORTED
    return TemporalResult(tuple(answers), tuple(excluded), bad_models, outcome)


def issue_certificate(snapshot: Snapshot, query: TemporalQuery) -> dict:
    payload = {"schema_version": 1, "snapshot": snapshot.to_dict(),
               "snapshot_sha256": canonical_hash(snapshot.to_dict()),
               "dependencies_sha256": canonical_hash([item.to_dict() for item in snapshot.dependencies]),
               "query": query.to_dict(), "query_sha256": canonical_hash(query.to_dict()),
               "result": evaluate(snapshot, query).to_dict()}
    payload["certificate_sha256"] = canonical_hash(payload)
    return payload

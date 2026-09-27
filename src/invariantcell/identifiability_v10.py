"""Original finite-library interval-observable analysis, isolated from v9.

Bounds are numerical LP optima conditional on the supplied library and deterministic
error intervals, never statistical confidence intervals or global molecular claims.
"""
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, linprog


class InfeasibleEvidence(ValueError):
    """The declared library and error intervals cannot jointly explain evidence."""


@dataclass(frozen=True)
class MixtureSet:
    forward: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    admitted: np.ndarray

    def __post_init__(self):
        a = np.asarray(self.forward, dtype=float).copy()
        lo, hi = [np.asarray(x, dtype=float).copy() for x in (self.lower, self.upper)]
        mask = np.asarray(self.admitted)
        if a.ndim != 2 or not 1 <= a.shape[1] <= 64 or a.shape[0] > 256:
            raise ValueError("Require 1..64 hypotheses and at most 256 observables")
        if lo.shape != (a.shape[0],) or hi.shape != lo.shape or mask.shape != (a.shape[1],):
            raise ValueError("Observable/library axes disagree")
        if mask.dtype != np.bool_ or not mask.any():
            raise ValueError("Explicit nonempty boolean construct/protocol/QC admission required")
        if not all(np.isfinite(x).all() for x in (a, lo, hi)) or np.any(lo > hi):
            raise ValueError("Finite ordered observable intervals required")
        for key, value in (("forward", a), ("lower", lo), ("upper", hi), ("admitted", mask.copy())):
            value.flags.writeable = False
            object.__setattr__(self, key, value)

    @property
    def size(self):
        return self.forward.shape[1]

    def matrices(self):
        return np.vstack((self.forward, -self.forward)), np.r_[self.upper, -self.lower]


def _vector(value, size):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError("Finite library-aligned vector required")
    return value


def _solve(c, a, b, eq, rhs, admitted):
    bounds = [(0.0, 1.0 if x else 0.0) for x in admitted]
    # HiGHS accepts threads; scipy warns because it forwards this option unchanged.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*", category=OptimizeWarning)
        r = linprog(c, A_ub=a if len(a) else None, b_ub=b if len(b) else None,
                    A_eq=eq, b_eq=rhs, bounds=bounds, method="highs-ds",
                    options={"threads": 1, "time_limit": 10.0,
                             "primal_feasibility_tolerance": 1e-9,
                             "dual_feasibility_tolerance": 1e-9})
    if r.status == 2:
        raise InfeasibleEvidence("Empty feasible set; do not assert a stable decision")
    if not r.success:
        raise RuntimeError(f"LP did not establish an optimum: {r.status}: {r.message}")
    upper = np.asarray([v[1] for v in bounds])
    primal_violation = max(0.0, float(np.max(a @ r.x - b, initial=0)),
                          float(np.max(np.abs(eq @ r.x - rhs))),
                          float(np.max(-r.x)), float(np.max(r.x - upper)))
    dual = float(rhs @ r.eqlin.marginals + upper @ r.upper.marginals)
    stationarity = c - eq.T @ r.eqlin.marginals - r.lower.marginals - r.upper.marginals
    if len(a):
        dual += float(b @ r.ineqlin.marginals)
        stationarity -= a.T @ r.ineqlin.marginals
    gap = abs(float(r.fun) - dual)
    dual_residual = float(np.max(np.abs(stationarity)))
    if max(primal_violation, gap, dual_residual) > 1e-7 * (1 + float(np.max(np.abs(c)))):
        raise RuntimeError("LP numerical diagnostic tolerance exceeded")
    return float(r.fun), {"weights": r.x.tolist(), "solver": "scipy.linprog/highs-ds",
                         "status": int(r.status), "message": r.message, "iterations": int(r.nit),
                         "primal_max_violation": primal_violation,
                         "absolute_primal_dual_gap": gap, "stationarity_max_residual": dual_residual,
                         "scope": "floating-point optimum of declared finite-library LP"}


def outcome_bounds(problem, outcome_lower, outcome_upper=None, threshold=0.5):
    lo = _vector(outcome_lower, problem.size)
    hi = lo if outcome_upper is None else _vector(outcome_upper, problem.size)
    if np.any(lo > hi) or not np.isfinite(threshold):
        raise ValueError("Invalid outcome intervals or decision threshold")
    a, b = problem.matrices()
    args = (a, b, np.ones((1, problem.size)), np.ones(1), problem.admitted)
    lower, lower_diag = _solve(lo, *args)
    neg_upper, upper_diag = _solve(-hi, *args)
    upper = -neg_upper
    # Explicit numerical margin avoids classifying threshold contact as robust.
    decision = "POSITIVE" if lower > threshold + 1e-7 else "NEGATIVE" if upper < threshold - 1e-7 else "UNRESOLVED"
    return {"lower": lower, "upper": upper, "width": max(0.0, upper-lower),
            "threshold": float(threshold), "decision": decision,
            "lower_diagnostics": lower_diag, "upper_diagnostics": upper_diag,
            "confidence_claim": False, "missing_library_states_covered": False}


@dataclass(frozen=True)
class ModeledAssay:
    assay_id: str
    forward: tuple
    absolute_error: float
    cost: float = 1.0
    role: str = "query"
    evidence_kind: str = "SYNTHETIC"

    def validate(self, size):
        if not self.assay_id or self.role != "query" or self.evidence_kind not in {"SYNTHETIC", "MODELED"}:
            raise ValueError("Only explicitly modeled query assays; no final endpoint")
        _vector(self.forward, size)
        if not np.isfinite([self.absolute_error, self.cost]).all() or self.absolute_error < 0 or self.cost <= 0:
            raise ValueError("Finite nonnegative error and positive cost required")


def reveal(problem, assay, observed):
    assay.validate(problem.size)
    if not np.isfinite(observed):
        raise ValueError("Finite observation required")
    return MixtureSet(np.vstack((problem.forward, assay.forward)),
                      np.r_[problem.lower, observed-assay.absolute_error],
                      np.r_[problem.upper, observed+assay.absolute_error], problem.admitted)


def worst_case_remaining_width(problem, assay, outcome_lower, outcome_upper=None):
    """Exact LP for sup_z width(W intersect {|a.w-z|<=epsilon}).

    Two admissible mixtures share some observation iff their predicted assay
    values differ by <=2epsilon. No grid, likelihood or posterior is used.
    """
    assay.validate(problem.size)
    lo = _vector(outcome_lower, problem.size)
    hi = lo if outcome_upper is None else _vector(outcome_upper, problem.size)
    if np.any(lo > hi):
        raise ValueError("Invalid outcome intervals")
    a, b = problem.matrices(); z = np.zeros_like(a); n = problem.size
    coupling = np.r_[assay.forward, -np.asarray(assay.forward)]
    aa = np.vstack((np.hstack((a,z)), np.hstack((z,a)), coupling, -coupling))
    bb = np.r_[b, b, 2*assay.absolute_error, 2*assay.absolute_error]
    eq = np.vstack((np.r_[np.ones(n),np.zeros(n)], np.r_[np.zeros(n),np.ones(n)]))
    value, diagnostic = _solve(np.r_[lo,-hi], aa, bb, eq, np.ones(2), np.tile(problem.admitted,2))
    return {"width": max(0.0,-value), "diagnostics": diagnostic,
            "criterion": "worst-case residual outcome interval width; not expected utility"}


def choose_assay(problem, assays, outcome_lower, outcome_upper=None, *, policy, budget,
                 already=(), seed=0):
    """All policies receive identical initial evidence, assay models and costs.

    A held final endpoint / hidden truth is intentionally absent from this API.
    """
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Invalid budget")
    if len({x.assay_id for x in assays}) != len(assays):
        raise ValueError("Duplicate assay identifiers")
    for assay in assays:
        assay.validate(problem.size)
    available = [x for x in assays if x.assay_id not in already and x.cost <= budget]
    if not available:
        return None
    if policy == "random":
        return available[int(np.random.default_rng(seed).integers(len(available)))].assay_id
    if policy == "fixed":
        return available[0].assay_id
    if policy == "cheapest":
        return min(available, key=lambda x:(x.cost,x.assay_id)).assay_id
    if policy == "uncertainty":
        scores = {x.assay_id:outcome_bounds(problem,x.forward)["width"]/x.cost for x in available}
    elif policy == "worst_case_decision":
        initial = outcome_bounds(problem,outcome_lower,outcome_upper)["width"]
        scores = {x.assay_id:(initial-worst_case_remaining_width(problem,x,outcome_lower,outcome_upper)["width"])/x.cost for x in available}
    else:
        raise ValueError("Unknown policy")
    return min(available, key=lambda x:(-scores[x.assay_id],x.assay_id)).assay_id

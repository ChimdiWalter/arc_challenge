"""Evaluation-only continuous utility metrics; never imported by policy selection."""
from statistics import mean, median
from .decision import evaluate


def continuous_evaluation(decisions, panel, protocol, outcomes, calibration=None):
    # Replay all frozen policies before the outcome join, using existing checks.
    result = evaluate(decisions, panel, protocol, outcomes, calibration)
    errors = {o['candidate_id']: o['agreement_error'] for o in outcomes}
    contexts = {c['candidate_id']: c['context_id'] for c in panel['candidates']}

    def stats(selected, universe):
        values = [errors[c] for c in selected]
        optimum = sorted(errors[c] for c in universe)[:len(values)]
        return {'selected_mean_error': mean(values) if values else None,
                'selected_median_error': median(values) if values else None,
                'oracle_mean_error_at_matched_count': mean(optimum) if optimum else None,
                'mean_error_regret_at_matched_count': mean(values)-mean(optimum) if values else None}

    for policy, decision in decisions.items():
        selected = decision['selected_ids']; report = result['policies'][policy]
        report.update(stats(selected, errors))
        report['oracle_mean_error_at_full_budget'] = mean(sorted(errors.values())[:protocol['budget']])
        for context, detail in report['per_context'].items():
            universe = [c for c in errors if contexts[c] == context]
            detail.update(stats([c for c in selected if contexts[c] == context], universe))
        for point in report['prefix_quality_coverage_curve']:
            point.update(stats(selected[:point['slots']], errors))
    result['continuous_metric_version'] = 'v1-evaluation-only'
    result['oracle_limitations'] = 'Oracle reads withheld errors only in evaluator. Matched-count regret must be read with coverage and unused capacity; abstention is not a gain.'
    return result

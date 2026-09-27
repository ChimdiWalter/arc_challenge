"""MIT-licensed function copied verbatim from sb_class_project; see NOTICE_SB.md.
Copyright (c) 2026 Chimdi Walter Ndubuisi.
"""
import numpy as np

def donor_bootstrap_ci(
    donor_df: "pd.DataFrame",
    metric: str,
    healthy_label: str = "Control",
    covid_label: str = "COVID-19",
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for the difference in donor-level medians.

    Resamples donors (not cells) with replacement.
    """
    rng = np.random.default_rng(seed)
    h = donor_df.loc[donor_df["condition"] == healthy_label, metric].dropna().values
    c = donor_df.loc[donor_df["condition"] == covid_label, metric].dropna().values

    obs_diff = float(np.median(c) - np.median(h))
    boot_diffs = np.empty(n_boot)

    for i in range(n_boot):
        h_boot = rng.choice(h, size=len(h), replace=True)
        c_boot = rng.choice(c, size=len(c), replace=True)
        boot_diffs[i] = np.median(c_boot) - np.median(h_boot)

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_diffs, 100 * alpha))
    hi = float(np.percentile(boot_diffs, 100 * (1 - alpha)))

    return {
        "metric": metric,
        "observed_diff": obs_diff,
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_level": ci,
        "n_bootstrap": n_boot,
        "boot_mean": float(boot_diffs.mean()),
        "boot_std": float(boot_diffs.std()),
    }

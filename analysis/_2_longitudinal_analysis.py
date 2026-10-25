"""
Longitudinal trend analysis of female enrollment in STEM fields (2012-2024).

Tests applied per field: OLS regression, Durbin-Watson, Mann-Kendall,
Theil-Sen estimator, and Spearman's rank correlation.
Reproduces section 4.1 of the paper.

Run from any directory: python analysis/_2_longitudinal_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import theilslopes, spearmanr
import pymannkendall as mk

from _0_preprocessing import load_admissions, compute_female_share

df_share = compute_female_share(load_admissions())
years    = df_share["Year"].values.astype(float)

FIELDS = {
    "All STEM":                "STEM",
    "Engineering & Technology": "ENGTECH",
    "Science & Mathematics":    "SCIMAT",
    "Non-STEM":                 "NONSTEM",
}


def analyze_field(label: str, col: str) -> None:
    values = df_share[col].values.astype(float)

    # OLS: female_share ~ year
    X     = sm.add_constant(years - 2018)
    model = sm.OLS(values, X).fit()
    slope     = model.params[1]
    intercept = model.params[0]
    r2        = model.rsquared
    p_ols     = model.pvalues[1]
    dw        = durbin_watson(model.resid)

    # Non-parametric tests
    mk_result = mk.original_test(values)
    ts        = theilslopes(values, years, alpha=0.95)
    rho, p_sp = spearmanr(years, values)

    above = int(np.sum(model.resid > 0))
    below = int(np.sum(model.resid < 0))

    print(f"-- {label} --")
    print(f"  OLS:          y = {intercept:.3f} + ({slope:+.3f}) * (year - 2018)")
    print(f"                R^2 = {r2:.2f},  p = {p_ols:.3f}")
    low, high = model.conf_int()[1]
    print(f"                95% CI: {low:.3f} to {high:.3f} pp/year")
    print(f"                Residuals: {above} above / {below} below,  DW = {dw:.2f}")
    print(f"                Lag-1 residual correlation = {np.corrcoef(model.resid[:-1], model.resid[1:])[0, 1]:.3f}")
    print(f"  Mann-Kendall: tau = {mk_result.Tau:.3f},  p = {mk_result.p:.3f},  "
          f"trend = {mk_result.trend}")
    print(f"  Theil-Sen:    slope = {ts.slope:.3f}  "
          f"(95% CI: {ts.low_slope:.3f} to {ts.high_slope:.3f})")
    print(f"  Spearman:     rho = {rho:.3f},  p = {p_sp:.3f}")
    for lag in [1, 2, 3]:
        robust = model.get_robustcov_results(cov_type="HAC", maxlags=lag,
                                           kernel="bartlett", use_correction=False, use_t=True)
        low, high = robust.conf_int()[1]
        print(f"  Newey-West ({lag} lag): SE = {robust.bse[1]:.3f}, "
              f"95% CI: {low:.3f} to {high:.3f}, p = {robust.pvalues[1]:.6f}")
    for name, test in [("Hamed-Rao", mk.hamed_rao_modification_test),
                       ("Yue-Wang", mk.yue_wang_modification_test)]:
        for lag in [1, None]:
            result = test(values, lag=lag)
            print(f"  {name} ({'1 lag' if lag == 1 else 'all lags'}): p = {result.p:.6f}")
    print(f"  Trend-free pre-whitening: p = {mk.trend_free_pre_whitening_modification_test(values).p:.6f}")
    print()


for label, col in FIELDS.items():
    analyze_field(label, col)

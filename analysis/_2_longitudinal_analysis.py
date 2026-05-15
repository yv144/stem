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
    "Engineering & Technology": "ENGTECH",
    "Science & Mathematics":    "SCIMAT",
    "Non-STEM":                 "NONSTEM",
}


def analyze_field(label: str, col: str) -> None:
    values = df_share[col].values.astype(float)

    # OLS: female_share ~ year
    X     = sm.add_constant(years)
    model = sm.OLS(values, X).fit()
    slope     = model.params[1]
    intercept = model.params[0]
    r2        = model.rsquared
    p_ols     = model.pvalues[1]
    dw        = durbin_watson(model.resid)

    # Non-parametric tests
    mk_result = mk.original_test(values)
    ts        = theilslopes(values, years, alpha=0.01)   # 99% CI
    rho, p_sp = spearmanr(years, values)

    above = int(np.sum(model.resid > 0))
    below = int(np.sum(model.resid < 0))

    print(f"-- {label} --")
    print(f"  OLS:          y = {slope:+.3f} * year + ({intercept:.0f})")
    print(f"                R^2 = {r2:.2f},  p = {p_ols:.3f}")
    print(f"                Residuals: {above} above / {below} below,  DW = {dw:.2f}")
    print(f"  Mann-Kendall: tau = {mk_result.Tau:.3f},  p = {mk_result.p:.3f},  "
          f"trend = {mk_result.trend}")
    print(f"  Theil-Sen:    slope = {ts.slope:.3f}  "
          f"(99% CI: {ts.low_slope:.3f} to {ts.high_slope:.3f})")
    print(f"  Spearman:     rho = {rho:.3f},  p = {p_sp:.3f}")
    print()


for label, col in FIELDS.items():
    analyze_field(label, col)

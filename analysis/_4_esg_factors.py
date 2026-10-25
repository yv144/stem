"""
Spearman correlations between female STEM and Engineering/ICT shares and economic, social,
and governance indicators for Kazakhstan (2012–2024).

Reproduces §4.2 and Table 2, including first differences and adjustment for year.
Run from any directory: python analysis/_4_esg_factors.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr, t

from _0_preprocessing import YEARS, load_admissions, compute_female_share, load_indicators


def partial_spearman(a, b, years):
    X = np.column_stack([np.ones(len(years)), years - np.mean(years)])
    ranks = np.column_stack([rankdata(a), rankdata(b)])
    residuals = ranks - X @ np.linalg.lstsq(X, ranks, rcond=None)[0]
    if np.any(np.linalg.norm(residuals, axis=0) < 1e-9):
        return np.nan, np.nan
    rho = float(pearsonr(residuals[:, 0], residuals[:, 1]).statistic)
    df = len(years) - 3
    p = 0.0 if abs(rho) >= 1 else 2 * t.sf(abs(rho) * np.sqrt(df / (1 - rho**2)), df)
    return rho, p


df_share = compute_female_share(load_admissions())
df_indic = load_indicators().drop(columns="GDP (constant 2015 US$)")

df = df_share[["Year", "STEM", "ENGTECH"]].merge(df_indic, on="Year", validate="one_to_one")
if df["Year"].tolist() != YEARS or not np.isfinite(df.to_numpy(dtype=float)).all():
    raise ValueError("These checks require complete annual data for 2012–2024.")

years = df["Year"].to_numpy(dtype=float)
indicator_cols = list(df_indic.columns.drop("Year"))

for outcome in ["STEM", "ENGTECH"]:
    female_share = df[outcome].to_numpy(dtype=float)
    print(f"\nFemale share in {outcome} admissions: rho (p)\n")
    print(f"  {'Indicator':<55} {'Levels':>17} {'First differences':>17} {'Year-adjusted':>17}")
    print("  " + "-" * 109)

    for col in indicator_cols:
        indicator = df[col].to_numpy(dtype=float)
        results = [spearmanr(female_share, indicator),
                   spearmanr(np.diff(female_share), np.diff(indicator)),
                   partial_spearman(female_share, indicator, years)]
        cells = [f"{rho:+.3f} ({p:.4f})" if np.isfinite(rho) else "undefined"
                 for rho, p in results]
        print(f"  {col:<55} " + " ".join(f"{cell:>17}" for cell in cells))

print()
print("n = 13 annual observations; n = 12 first differences; two-sided, unadjusted p-values.")
print("Year-adjusted correlations are undefined when either rank series is collinear with year.")

"""
Trend analysis of the ten most popular Engineering & Technology majors (2012–2018).

The series ends at 2018 because a national curriculum reclassification in 2019
changed major codes, making pre- and post-2019 data incomparable at the major level.

Reproduces Table 1 of the paper.
Run from any directory: python analysis/_3_detailed_engtech.py
"""

import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import permutations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import pymannkendall as mk

DATA_DIR = Path(__file__).parent.parent / "data"

df = pd.read_csv(DATA_DIR / "_2_detailed_engtech.csv")

# First row is the ENGTECH aggregate total; remaining rows are individual majors
df_majors = df.iloc[1:].copy()

years     = np.array([2012, 2013, 2014, 2015, 2016, 2017, 2018], dtype=float)
year_cols = [str(int(y)) for y in years]
ranks = np.arange(1, len(years) + 1)
null_rho = 1 - 6 * ((np.array(list(permutations(ranks))) - ranks) ** 2).sum(axis=1) / (
    len(ranks) * (len(ranks) ** 2 - 1))


def round_share(value):
    return (Decimal(str(value)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "+"
    return ""


print(f"{'Major':<45} {'2012':>6} {'2018':>6} {'Δ pp':>7}  {'Spearman':>9}  {'MK tau':>7}")
print("-" * 90)

for _, row in df_majors.iterrows():
    values  = row[year_cols].astype(float).values
    share   = values * 100          # stored as decimals (0-1); convert to percent
    val_12  = round_share(row["2012"])
    val_18  = round_share(row["2018"])
    delta   = round_share(Decimal(str(row["2018"])) - Decimal(str(row["2012"])))

    if len(np.unique(share)) != len(share):
        raise ValueError("The exact permutation distribution requires seven distinct annual shares.")
    rho = spearmanr(years, share).statistic
    p_sp = np.mean(np.abs(null_rho) >= abs(rho) - 1e-12)
    mk_result = mk.original_test(share)

    sp_str = f"{rho:+.3f}{sig_stars(p_sp)}"
    mk_str = f"{mk_result.Tau:+.3f}{sig_stars(mk_result.p)}"

    print(f"  {row['Major']:<43} {val_12:5.1f}%  {val_18:5.1f}%  {delta:+6.1f}  "
          f"{sp_str:>10}  {mk_str:>8}")

print()
print("Note: + p<.10  * p<.05  ** p<.01  *** p<.001")
print("Spearman p-values use all 5,040 permutations; changes are percentage points.")

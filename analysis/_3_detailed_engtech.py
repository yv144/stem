"""
Trend analysis of the ten most popular Engineering & Technology majors (2012–2018).

The series ends at 2018 because a national curriculum reclassification in 2019
changed major codes, making pre- and post-2019 data incomparable at the major level.

Reproduces Table 2 of the paper.
Run from any directory: python analysis/_3_detailed_engtech.py
"""

import sys
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


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "+"
    return ""


print(f"{'Major':<45} {'2012':>6} {'2018':>6} {'Change':>7}  {'Spearman':>9}  {'MK tau':>7}")
print("-" * 90)

for _, row in df_majors.iterrows():
    values  = row[year_cols].astype(float).values
    share   = values * 100          # stored as decimals (0-1); convert to percent
    val_12  = share[0]
    val_18  = share[-1]
    delta   = val_18 - val_12

    rho, p_sp = spearmanr(years, share)
    mk_result = mk.original_test(share)

    sp_str = f"{rho:+.3f}{sig_stars(p_sp)}"
    mk_str = f"{mk_result.Tau:+.3f}{sig_stars(mk_result.p)}"

    print(f"  {row['Major']:<43} {val_12:5.1f}%  {val_18:5.1f}%  {delta:+5.1f}%  "
          f"{sp_str:>10}  {mk_str:>8}")

print()
print("Note: + p<.10  * p<.05  ** p<.01  *** p<.001")

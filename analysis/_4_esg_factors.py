"""
Spearman correlations between female STEM enrollment and economic, social,
and governance indicators for Kazakhstan (2012–2024).

Reproduces §4.2 and Table 1 of the paper.
Run from any directory: python analysis/_4_esg_factors.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scipy.stats import spearmanr

from _0_preprocessing import load_admissions, compute_female_share, load_indicators

df_share = compute_female_share(load_admissions())
df_indic = load_indicators()

df = df_share[["Year", "STEM"]].merge(df_indic, on="Year")
df = df.rename(columns={"STEM": "Female share in STEM (%)"})

female_share = df["Female share in STEM (%)"].values
indicator_cols = [c for c in df.columns if c not in ("Year", "Female share in STEM (%)")]

print("Spearman correlations with female share in STEM enrollment\n")
print(f"  {'Indicator':<55} {'rho':>6}  {'p':>6}")
print("  " + "-" * 72)

for col in indicator_cols:
    rho, p = spearmanr(female_share, df[col])
    sig = " *" if p < 0.05 else ""
    print(f"  {col:<55} {rho:+.3f}  {p:.3f}{sig}")

print()
print("  * p < 0.05")

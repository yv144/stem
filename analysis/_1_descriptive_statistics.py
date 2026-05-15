"""
Descriptive statistics for Kazakhstan STEM enrollment (2012–2024).

Reproduces the summary figures reported in §3.3 and the opening of §4.1.
Run from any directory: python analysis/_1_descriptive_statistics.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from _0_preprocessing import load_admissions, compute_female_share

df_admissions = load_admissions()
df_share = compute_female_share(df_admissions)

year_cols = [str(y) for y in range(2012, 2025)]


def _total(major: str, gender: str) -> dict:
    row = df_admissions[(df_admissions["major"] == major) & (df_admissions["gender"] == gender)]
    return row[year_cols].iloc[0].astype(float)


# ── Overall enrollment ──────────────────────────────────────────────────────────

all_combined = _total("ALL", "COMBINED")
all_women    = _total("ALL", "WOMEN")
total_all    = int(all_combined.sum())
total_women  = int(all_women.sum())
gpi_all      = total_women / (total_all - total_women)

print("=== Overall Enrollment (2012–2024) ===")
print(f"  Total admissions:          {total_all:,}")
print(f"  Female admissions:         {total_women:,}  ({total_women/total_all*100:.1f}%)")
print(f"  Gender Parity Index (GPI): {gpi_all:.2f}")
print()

# ── STEM enrollment ─────────────────────────────────────────────────────────────

stem_combined = _total("STEM", "COMBINED")
stem_women    = _total("STEM", "WOMEN")
total_stem    = int(stem_combined.sum())
total_stem_w  = int(stem_women.sum())
gpi_stem      = total_stem_w / (total_stem - total_stem_w)

print("=== STEM Enrollment (2012–2024) ===")
print(f"  Total STEM admissions:     {total_stem:,}")
print(f"  Female STEM admissions:    {total_stem_w:,}  ({total_stem_w/total_stem*100:.1f}%)")
print(f"  STEM GPI:                  {gpi_stem:.2f}")
print()

# ── 2024 enrollment breakdown ───────────────────────────────────────────────────

total_2024 = float(all_combined["2024"])
engtech_2024 = float(_total("ENGTECH", "COMBINED")["2024"])
scimat_2024  = float(_total("SCIMAT",  "COMBINED")["2024"])
nonstem_2024 = float(_total("NONSTEM", "COMBINED")["2024"])

print("=== 2024 Enrollment Breakdown ===")
print(f"  Engineering & Technology:  {engtech_2024:,.0f}  ({engtech_2024/total_2024*100:.1f}%)")
print(f"  Science & Mathematics:     {scimat_2024:,.0f}   ({scimat_2024/total_2024*100:.1f}%)")
print(f"  Non-STEM:                  {nonstem_2024:,.0f}  ({nonstem_2024/total_2024*100:.1f}%)")
print()

# ── Female share by field ───────────────────────────────────────────────────────

print("=== Female Share by Field ===")
fields = {
    "Engineering & Technology": "ENGTECH",
    "Science & Mathematics":    "SCIMAT",
    "Non-STEM":                 "NONSTEM",
    "All fields":               "ALL",
}

for label, col in fields.items():
    series = df_share[col].values.astype(float)
    years  = df_share["Year"].values
    val_2012 = df_share.loc[df_share["Year"] == 2012, col].iloc[0]
    val_2024 = df_share.loc[df_share["Year"] == 2024, col].iloc[0]
    print(f"  {label}")
    print(f"    2012: {val_2012:.1f}%   2024: {val_2024:.1f}%"
          f"   Mean ± SD: {np.mean(series):.1f}% ± {np.std(series, ddof=1):.1f}%")

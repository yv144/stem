# Supplementary Code: Female Enrollment in STEM Higher Education in Kazakhstan

Replication code for the paper. Covers the period 2012–2024 using data from the Bureau of National Statistics of Kazakhstan, the World Development Indicators (World Bank), and the Worldwide Governance Indicators (World Bank).

## Structure

```
data/
  _1_national_statistics.csv     # admissions by major, gender, and year
  _2_detailed_engtech.csv        # female share in top engineering majors (2012-2018)
  _3_development_indicators.csv  # World Development Indicators for Kazakhstan
  _4_governance_indicators.xlsx  # Worldwide Governance Indicators for Kazakhstan

analysis/
  _0_preprocessing.py            # data loading functions used by all scripts
  _1_descriptive_statistics.py   # overall enrollment and gender shares
  _2_longitudinal_analysis.py    # OLS, Mann-Kendall, Theil-Sen, Spearman by field 
  _3_detailed_engtech.py         # per-major trend table for engineering 
  _4_esg_factors.py              # correlations with economic and governance indicators 
```

## Setup

Requires Python 3.14+. Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Running

Run any script from the project root:

```bash
python analysis/_1_descriptive_statistics.py
python analysis/_2_longitudinal_analysis.py
python analysis/_3_detailed_engtech.py
python analysis/_4_esg_factors.py
```

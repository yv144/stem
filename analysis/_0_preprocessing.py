"""
Data loading and preprocessing for the Kazakhstan STEM gender study.

Each function returns a clean DataFrame ready for analysis.
Run from the stem/ directory or any location; paths are resolved relative to this file.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
YEARS = list(range(2012, 2025))

# ISCED-F 2013 broad-field groupings, as coded in _1_national_statistics.csv.
# Pre-2019 codes (5B-*) and post-2019 codes (SCI_05, IT_06, ENG_07) are mutually
# exclusive per year, so summing them gives the correct annual aggregate.
SCIMAT  = ["SCI_05", "5B-NAT"]
ENGTECH = ["IT_06", "ENG_07", "5B-TECH"]
STEM    = SCIMAT + ENGTECH

# WDI series codes and their readable labels for _3_development_indicators.csv
WDI_CODES = {
    "NY.GDP.MKTP.KD":    "GDP (constant 2015 US$)",
    "NY.GDP.PCAP.KD":    "GDP per capita (constant 2015 US$)",
    "NV.IND.MANF.KD":    "Manufacturing, value added (constant 2015 US$)",
    "NV.IND.TOTL.KD":    "Industry, value added (constant 2015 US$)",
    "IT.NET.USER.ZS":    "Internet users (% of population)",
    "SL.EMP.WORK.FE.ZS": "Female wage workers (% of female employment)",
    "SG.GEN.PARL.ZS":   "Women in parliament (%)",
}

# Column name in the WGI Excel sheets that holds the governance estimate
_WGI_ESTIMATE_COL = "Governance estimate (approx. -2.5 to +2.5)"
_WGI_COUNTRY_COL  = "Economy (code)"


def load_admissions() -> pd.DataFrame:
    """
    Return admissions data with computed aggregate rows for SCIMAT, ENGTECH, STEM, NONSTEM.

    Source: _1_national_statistics.csv
    Columns: major, gender, then one column per year (as strings, e.g. '2012').
    """
    df = pd.read_csv(DATA_DIR / "_1_national_statistics.csv").fillna(0)
    year_cols = [str(y) for y in YEARS]

    def _aggregate(name: str, components: list, gender: str) -> dict:
        mask = df["major"].isin(components) & (df["gender"] == gender)
        totals = df[mask][year_cols].sum()
        return {"major": name, "gender": gender, **totals.to_dict()}

    new_rows = []
    for group_name, components in [("SCIMAT", SCIMAT), ("ENGTECH", ENGTECH), ("STEM", STEM)]:
        for gender in ["COMBINED", "WOMEN"]:
            new_rows.append(_aggregate(group_name, components, gender))

    # NONSTEM = ALL − STEM
    stem_rows = {r["gender"]: r for r in new_rows if r["major"] == "STEM"}
    for gender in ["COMBINED", "WOMEN"]:
        all_vals  = df[(df["major"] == "ALL") & (df["gender"] == gender)][year_cols].values[0]
        stem_vals = pd.Series({y: stem_rows[gender][y] for y in year_cols}).values
        nonstem   = all_vals - stem_vals
        new_rows.append({"major": "NONSTEM", "gender": gender,
                         **dict(zip(year_cols, nonstem))})

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def compute_female_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return female share (%) per year for each field group.

    Input: output of load_admissions().
    Returns a DataFrame with columns: Year, ALL, ENGTECH, SCIMAT, STEM, NONSTEM.
    """
    year_cols = [str(y) for y in YEARS]
    groups = ["ALL", "ENGTECH", "SCIMAT", "STEM", "NONSTEM"]

    records = {}
    for group in groups:
        combined = df[(df["major"] == group) & (df["gender"] == "COMBINED")][year_cols].values[0]
        women    = df[(df["major"] == group) & (df["gender"] == "WOMEN")][year_cols].values[0]
        records[group] = women / combined * 100

    result = pd.DataFrame(records, index=pd.Index(YEARS, name="Year"))
    return result.reset_index().sort_values("Year").reset_index(drop=True)


def load_wdi() -> pd.DataFrame:
    """
    Return World Development Indicators for Kazakhstan, filtered to WDI_CODES.

    Source: _3_development_indicators.csv (World Bank Data 360 format).
    Returns a DataFrame with columns: Year, then one column per indicator (readable label).
    """
    df = pd.read_csv(DATA_DIR / "_3_development_indicators.csv")
    df = df[df["Series Code"].isin(WDI_CODES)].copy()
    df["Series Name"] = df["Series Code"].map(WDI_CODES)

    year_str_cols = [str(y) for y in YEARS]
    df = df[["Series Name"] + year_str_cols].set_index("Series Name")

    # Pivot: rows = years, columns = indicator labels
    result = df.T.reset_index().rename(columns={"index": "Year"})
    result["Year"] = result["Year"].astype(int)
    for col in result.columns[1:]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result.sort_values("Year").reset_index(drop=True)


def load_wgi() -> pd.DataFrame:
    """
    Return Worldwide Governance Indicators for Kazakhstan (2012–2024).

    Source: _4_governance_indicators.xlsx (World Bank WGI format).
    Sheets: rq (Regulatory Quality), ge (Government Effectiveness), rl (Rule of Law).
    Returns a DataFrame with columns: Year, Regulatory Quality, Government Effectiveness, Rule of Law.
    """
    sheets = {
        "rq": "Regulatory Quality",
        "ge": "Government Effectiveness",
        "rl": "Rule of Law",
    }

    frames = []
    for sheet, label in sheets.items():
        df = pd.read_excel(DATA_DIR / "_4_governance_indicators.xlsx", sheet_name=sheet)
        kaz = df[df[_WGI_COUNTRY_COL] == "KAZ"][["Year", _WGI_ESTIMATE_COL]].copy()
        kaz = kaz[(kaz["Year"] >= 2012) & (kaz["Year"] <= 2024)]
        kaz = kaz.rename(columns={_WGI_ESTIMATE_COL: label}).set_index("Year")
        frames.append(kaz)

    return pd.concat(frames, axis=1).reset_index().sort_values("Year").reset_index(drop=True)


def load_indicators() -> pd.DataFrame:
    """Return merged WDI and WGI data on Year."""
    return load_wdi().merge(load_wgi(), on="Year")

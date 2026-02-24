# Streamlit Web Application

import os
import re
import io
import json
import warnings
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# PAGE CONFIG

st.set_page_config(
    page_title="Data Quality Framework",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM CSS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background-color: #f5f7fa;
    color: #1e293b;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}
section[data-testid="stSidebar"] * {
    color: #334155 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #1e293b !important;
}

/* Headers */
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    color: #1e293b !important;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-card.hard { border-left: 4px solid #ef4444; }
.metric-card.soft { border-left: 4px solid #f59e0b; }
.metric-card.ok   { border-left: 4px solid #10b981; }
.metric-card.info { border-left: 4px solid #3b82f6; }

.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    color: #94a3b8;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    line-height: 1;
}
.metric-card.hard .metric-value { color: #ef4444; }
.metric-card.soft .metric-value { color: #d97706; }
.metric-card.ok   .metric-value { color: #10b981; }
.metric-card.info .metric-value { color: #3b82f6; }

/* Issue badge */
.badge-hard { background: #fee2e2; color: #b91c1c; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
.badge-soft { background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }

/* Decision banner */
.decision-blocked {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 1px solid #fca5a5;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    color: #b91c1c;
    letter-spacing: 0.04em;
    box-shadow: 0 1px 4px rgba(239,68,68,0.1);
}
.decision-proceed {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1px solid #86efac;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    color: #15803d;
    letter-spacing: 0.04em;
    box-shadow: 0 1px 4px rgba(16,185,129,0.1);
}

/* Section header */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #94a3b8;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* Issue row */
.issue-row {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.875rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.issue-check {
    font-family: 'IBM Plex Mono', monospace;
    color: #64748b;
    font-size: 0.78rem;
    margin-bottom: 0.25rem;
}
.issue-detail {
    color: #334155;
    line-height: 1.5;
}

/* Dimension tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9;
    border-bottom: 1px solid #e2e8f0;
    border-radius: 8px 8px 0 0;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.73rem;
    letter-spacing: 0.04em;
    color: #64748b;
    padding: 0.6rem 1rem;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #2563eb !important;
    border-bottom: 2px solid #2563eb !important;
    background: transparent !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.04em;
    background: #2563eb;
    border: 1px solid #1d4ed8;
    color: #ffffff;
    border-radius: 7px;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
    box-shadow: 0 1px 3px rgba(37,99,235,0.25);
}
.stButton > button:hover {
    background: #1d4ed8;
    color: #fff;
}

/* Download button */
.stDownloadButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    background: #ffffff;
    border: 1px solid #d1d5db;
    color: #374151;
    border-radius: 7px;
    transition: all 0.2s;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}
.stDownloadButton > button:hover {
    background: #f9fafb;
    border-color: #9ca3af;
}

/* Expander */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #475569 !important;
}

/* Number input, text input */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background: #ffffff !important;
    border-color: #d1d5db !important;
    color: #1e293b !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Progress bar */
.stProgress .st-bo { background-color: #2563eb !important; }

/* Alert */
.stAlert { border-radius: 8px !important; }

/* Title */
.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #1e293b;
    letter-spacing: -0.02em;
}
.app-title span {
    color: #2563eb;
}
.app-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #94a3b8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}
</style>
""", unsafe_allow_html=True)


# CONFIGURATION (sidebar-editable)

REQUIRED_COLUMNS = [
    "timevalue", "providerkey", "companynameofficial",
    "fiscalperiodend", "operationstatustype", "ipostatustype",
    "geonameen", "industrycode", "REVENUE", "unit_REVENUE",
]

VALID_OPERATION_STATUSES = {"ACTIVE", "INACTIVE", "DISSOLVED", "BANKRUPT", "RESTRUCTURING"}
VALID_IPO_STATUSES        = {"PUBLIC", "PRIVATE", "DELISTED", "PENDING"}
CURRENT_YEAR              = datetime.datetime.now().year

COUNTRY_CURRENCY_MAP = {
    "United Kingdom": "GBP", "India": "INR", "United States": "USD",
    "Germany": "EUR", "France": "EUR", "Japan": "JPY",
    "Sweden": "SEK", "Denmark": "DKK", "Indonesia": "IDR",
}

FISCALPERIODEND_REGEX = re.compile(r"^\d{2}-[A-Za-z]{3}$")
INDUSTRYCODE_REGEX    = re.compile(r"^\d{3,4}\s-\s.+$")
CURRENCY_REGEX        = re.compile(r"^[A-Z]{3}$")

VALID_MONTH_ABBREVS = {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"}
MONTH_MAX_DAYS = {
    "Jan":31,"Feb":29,"Mar":31,"Apr":30,"May":31,"Jun":30,
    "Jul":31,"Aug":31,"Sep":30,"Oct":31,"Nov":30,"Dec":31,
}

FLAG_PRIORITY = {"exact_repeat":10,"zero_revenue":8,"negative":8,"yoy_spike":5,"missing_revenue":3}

DIMENSION_COL_MAP = {
    "Completeness": "flag_completeness",
    "Validity":     "flag_validity",
    "Consistency":  "flag_consistency",
    "Uniqueness":   "flag_uniqueness",
    "Plausibility": "flag_llm",
}

LLM_MODEL = "gpt-4o-mini"


# QUALITY CHECK FUNCTIONS

def _issue(severity, dimension, check, detail, rows=None):
    return {"severity": severity, "dimension": dimension, "check": check,
            "detail": detail, "affected_rows": rows or []}


def check_completeness(df, revenue_null_pct_max=0.05):
    issues = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(_issue("HARD_REJECT","Completeness","missing_columns",
                             f"Required columns absent: {missing_cols}"))
        return issues
    null_name = df.index[df["companynameofficial"].isnull()].tolist()
    if null_name:
        issues.append(_issue("HARD_REJECT","Completeness","null_companynameofficial",
                             f"{len(null_name)} row(s) have no companynameofficial",null_name))
    rev_null_pct = df["REVENUE"].isnull().mean()
    if rev_null_pct > revenue_null_pct_max:
        issues.append(_issue("SOFT_FLAG","Completeness","high_revenue_null_pct",
                             f"REVENUE null rate = {rev_null_pct:.1%} (threshold {revenue_null_pct_max:.0%})"))
    active_no_rev = df.index[(df["operationstatustype"]=="ACTIVE") & df["REVENUE"].isnull()].tolist()
    if active_no_rev:
        issues.append(_issue("SOFT_FLAG","Completeness","active_missing_revenue",
                             f"{len(active_no_rev)} ACTIVE row(s) missing REVENUE",active_no_rev))
    public_no_rev = df.index[(df["ipostatustype"]=="PUBLIC") & df["REVENUE"].isnull()].tolist()
    if public_no_rev:
        issues.append(_issue("SOFT_FLAG","Completeness","public_missing_revenue",
                             f"{len(public_no_rev)} PUBLIC row(s) missing REVENUE",public_no_rev))
    rev_no_unit = df.index[df["REVENUE"].notna() & df["unit_REVENUE"].isnull()].tolist()
    unit_no_rev = df.index[df["unit_REVENUE"].notna() & df["REVENUE"].isnull()].tolist()
    if rev_no_unit:
        issues.append(_issue("SOFT_FLAG","Completeness","revenue_without_unit",
                             f"{len(rev_no_unit)} row(s) have REVENUE but no unit_REVENUE",rev_no_unit))
    if unit_no_rev:
        issues.append(_issue("SOFT_FLAG","Completeness","unit_without_revenue",
                             f"{len(unit_no_rev)} row(s) have unit_REVENUE but no REVENUE",unit_no_rev))
    return issues


def _parse_fiscalperiodend(val):
    if pd.isnull(val): return False, False
    if isinstance(val, (datetime.datetime, datetime.date)): return True, val.year > CURRENT_YEAR
    s = str(val).strip()
    if not FISCALPERIODEND_REGEX.match(s): return False, False
    day_str, month_abbr = s.split("-")
    day = int(day_str)
    if month_abbr not in VALID_MONTH_ABBREVS: return False, False
    if day < 1 or day > MONTH_MAX_DAYS.get(month_abbr, 31): return False, False
    return True, False


def check_validity(df, revenue_max=10e12):
    issues = []

    def valid_year(x):
        try: return (not pd.isnull(x)) and (x == int(x)) and (1900 <= int(x) <= CURRENT_YEAR)
        except: return False

    bad_tv = df.index[~df["timevalue"].apply(valid_year)].tolist()
    if bad_tv:
        issues.append(_issue("HARD_REJECT","Validity","invalid_timevalue",
                             f"{len(bad_tv)} row(s) with out-of-range or non-integer timevalue",bad_tv))

    parsed = df["fiscalperiodend"].apply(_parse_fiscalperiodend)
    bad_format  = df.index[~parsed.apply(lambda t: t[0])].tolist()
    future_date = df.index[parsed.apply(lambda t: t[0] and t[1])].tolist()
    if bad_format:
        issues.append(_issue("SOFT_FLAG","Validity","invalid_fiscalperiodend_format",
                             f"{len(bad_format)} row(s) with unrecognised fiscalperiodend (expected DD-Mon)",bad_format))
    if future_date:
        issues.append(_issue("SOFT_FLAG","Validity","fiscalperiodend_future_date",
                             f"{len(future_date)} row(s) have fiscalperiodend year > {CURRENT_YEAR}",future_date))

    bad_ic = df.index[~df["industrycode"].apply(
        lambda x: bool(INDUSTRYCODE_REGEX.match(str(x).strip())) if not pd.isnull(x) else False
    )].tolist()
    if bad_ic:
        issues.append(_issue("HARD_REJECT","Validity","invalid_industrycode_format",
                             f"{len(bad_ic)} row(s) with malformed industrycode",bad_ic))

    bad_ops = df.index[~df["operationstatustype"].isin(VALID_OPERATION_STATUSES)].tolist()
    if bad_ops:
        issues.append(_issue("HARD_REJECT","Validity","invalid_operationstatustype",
                             f"Unknown operationstatustype: {df.loc[bad_ops,'operationstatustype'].unique().tolist()}",bad_ops))

    bad_ipo = df.index[~df["ipostatustype"].isin(VALID_IPO_STATUSES)].tolist()
    if bad_ipo:
        issues.append(_issue("HARD_REJECT","Validity","invalid_ipostatustype",
                             f"Unknown ipostatustype: {df.loc[bad_ipo,'ipostatustype'].unique().tolist()}",bad_ipo))

    has_unit = df["unit_REVENUE"].notna()
    bad_cur = df.index[has_unit & ~df["unit_REVENUE"].apply(
        lambda x: bool(CURRENCY_REGEX.match(str(x).strip())) if pd.notna(x) else False)].tolist()
    if bad_cur:
        issues.append(_issue("SOFT_FLAG","Validity","invalid_currency_code",
                             f"{len(bad_cur)} row(s) with non-ISO unit_REVENUE",bad_cur))

    has_rev = df["REVENUE"].notna()
    neg_rev  = df.index[has_rev & (df["REVENUE"] < 0)].tolist()
    huge_rev = df.index[has_rev & (df["REVENUE"] > revenue_max)].tolist()
    if neg_rev:
        issues.append(_issue("SOFT_FLAG","Validity","negative_revenue",f"{len(neg_rev)} row(s) with REVENUE < 0",neg_rev))
    if huge_rev:
        issues.append(_issue("SOFT_FLAG","Validity","revenue_exceeds_cap",f"{len(huge_rev)} row(s) with REVENUE > {revenue_max:.0e}",huge_rev))
    return issues


def check_consistency(df, yoy_threshold=0.50):
    issues = []
    global_min_year = int(df["timevalue"].dropna().min())
    global_max_year = int(df["timevalue"].dropna().max())

    entity_attrs = ["companynameofficial","industrycode","fiscalperiodend","geonameen","ipostatustype"]
    for attr in entity_attrs:
        for pk, grp in df.groupby("providerkey"):
            unique_vals = grp[attr].dropna().unique()
            if len(unique_vals) > 1:
                sev = "HARD_REJECT" if attr in ("companynameofficial","industrycode") else "SOFT_FLAG"
                issues.append(_issue(sev,"Consistency",f"entity_{attr}_conflict",
                                     f"providerkey {pk} has multiple {attr} values: {unique_vals.tolist()}",
                                     grp.index.tolist()))

    has_both = df["unit_REVENUE"].notna() & df["geonameen"].notna()
    for pk, grp in df[has_both].groupby("providerkey"):
        country = grp["geonameen"].iloc[0]
        expected = COUNTRY_CURRENCY_MAP.get(country)
        if not expected: continue
        non_expected = [c for c in grp["unit_REVENUE"].dropna().unique() if c != expected]
        if non_expected:
            issues.append(_issue("SOFT_FLAG","Consistency","currency_country_mismatch",
                                 f"{grp['companynameofficial'].iloc[0]} (pk={pk}, {country}): expected {expected}, reports in {non_expected}",
                                 grp.index.tolist()))

    for pk, grp in df.groupby("providerkey"):
        grp_sorted = grp.sort_values("timevalue")
        company = grp_sorted["companynameofficial"].iloc[0]
        is_active = grp_sorted["operationstatustype"].eq("ACTIVE").all()
        is_public = grp_sorted["ipostatustype"].eq("PUBLIC").all()

        zero_rows = grp_sorted.index[grp_sorted["REVENUE"].notna() & (grp_sorted["REVENUE"]==0)].tolist()
        if zero_rows and is_active and is_public:
            zero_yrs = grp_sorted.loc[zero_rows,"timevalue"].tolist()
            issues.append(_issue("SOFT_FLAG","Consistency","zero_revenue_active_public",
                                 f"providerkey={pk} ({company}): REVENUE=0 in {zero_yrs} for ACTIVE PUBLIC",zero_rows))

        grp_rev = grp_sorted.dropna(subset=["REVENUE"])
        if len(grp_rev) < 2: continue
        rev_vals = grp_rev["REVENUE"].values
        yr_vals  = grp_rev["timevalue"].values
        idxs     = grp_rev.index.tolist()

        for i in range(1, len(rev_vals)):
            prev, curr = rev_vals[i-1], rev_vals[i]
            year_gap = int(yr_vals[i]) - int(yr_vals[i-1])
            if prev == 0: continue
            pct_change = (curr - prev) / abs(prev)
            adj_thresh = yoy_threshold * year_gap
            if abs(pct_change) > adj_thresh:
                issues.append(_issue("SOFT_FLAG","Consistency","revenue_yoy_spike",
                                     f"providerkey={pk} ({company}): {int(yr_vals[i-1])}→{int(yr_vals[i])} ({year_gap}-yr gap) {pct_change:+.0%} (threshold ±{adj_thresh:.0%})",
                                     [idxs[i-1], idxs[i]]))

    for pk, grp in df.groupby("providerkey"):
        grp_sorted = grp.sort_values("timevalue").dropna(subset=["REVENUE"])
        if len(grp_sorted) < 3: continue
        rev_list = grp_sorted["REVENUE"].values
        yr_list  = grp_sorted["timevalue"].values
        idx_list = grp_sorted.index.tolist()
        reported = set()
        for i in range(len(rev_list)):
            for j in range(i+2, len(rev_list)):
                key = (round(rev_list[i],2), int(yr_list[i]), int(yr_list[j]))
                if rev_list[i] == rev_list[j] and key not in reported:
                    reported.add(key)
                    issues.append(_issue("SOFT_FLAG","Consistency","exact_revenue_repeat_non_adjacent",
                                         f"providerkey={pk} ({grp_sorted['companynameofficial'].iloc[0]}): identical REVENUE {rev_list[i]:.4e} in {int(yr_list[i])} and {int(yr_list[j])}",
                                         [idx_list[i], idx_list[j]]))

    for pk, grp in df.groupby("providerkey"):
        years = sorted(grp["timevalue"].dropna().astype(int).tolist())
        if len(years) < 2: continue
        span_min = max(min(years), global_min_year)
        span_max = min(max(years), global_max_year)
        missing  = sorted(set(range(span_min, span_max+1)) - set(years))
        if missing:
            issues.append(_issue("SOFT_FLAG","Consistency","missing_years_in_series",
                                 f"providerkey={pk} ({grp['companynameofficial'].iloc[0]}): gap year(s) {missing} within {span_min}–{span_max}",
                                 grp.index.tolist()))
    return issues


def check_uniqueness(df):
    issues = []
    dupes = df[df.duplicated(subset=["providerkey","timevalue"], keep=False)]
    if not dupes.empty:
        issues.append(_issue("HARD_REJECT","Uniqueness","duplicate_primary_key",
                             f"{len(dupes)} row(s) share a (providerkey, timevalue) pair",dupes.index.tolist()))
    exact_dupes = df[df.duplicated(keep=False)]
    if not exact_dupes.empty:
        issues.append(_issue("HARD_REJECT","Uniqueness","exact_duplicate_rows",
                             f"{len(exact_dupes)} row(s) are completely identical",exact_dupes.index.tolist()))
    name_to_keys = df.dropna(subset=["companynameofficial"]).groupby("companynameofficial")["providerkey"].nunique()
    for name in name_to_keys[name_to_keys > 1].index.tolist():
        keys = df[df["companynameofficial"]==name]["providerkey"].unique().tolist()
        issues.append(_issue("SOFT_FLAG","Uniqueness","company_name_maps_to_multiple_keys",
                             f"'{name}' maps to multiple providerkeys: {keys}"))
    return issues


def check_plausibility_llm(df, api_key=None, max_candidates=20):
    issues = []
    if not api_key:
        issues.append(_issue("SOFT_FLAG","Plausibility","llm_skipped",
                             "No OpenAI API key provided – LLM plausibility check skipped."))
        return issues
    try:
        from openai import OpenAI
    except ImportError:
        issues.append(_issue("SOFT_FLAG","Plausibility","llm_skipped","openai package not installed."))
        return issues

    # Build candidates
    scored = []
    for pk, grp in df.groupby("providerkey"):
        grp_s = grp.sort_values("timevalue")
        company  = grp_s["companynameofficial"].iloc[0]
        industry = grp_s["industrycode"].iloc[0]
        currency = grp_s["unit_REVENUE"].dropna().iloc[0] if grp_s["unit_REVENUE"].notna().any() else "?"
        country  = grp_s["geonameen"].iloc[0]
        rev_rows = grp_s.dropna(subset=["REVENUE"])[["timevalue","REVENUE"]]
        series   = [{"year": int(r.timevalue),"revenue": float(r.REVENUE)} for _,r in rev_rows.iterrows()]
        flags = []; score = 0
        neg = [r for r in series if r["revenue"] < 0]
        if neg: flags.append(f"Negative revenue: {neg}"); score += 8
        if grp_s["operationstatustype"].eq("ACTIVE").all() and grp_s["ipostatustype"].eq("PUBLIC").all():
            zero = [r for r in series if r["revenue"]==0]
            if zero: flags.append(f"Zero revenue (ACTIVE PUBLIC): {zero}"); score += 8
            null_yrs = grp_s.loc[grp_s["REVENUE"].isnull(),"timevalue"].tolist()
            if null_yrs: flags.append(f"REVENUE missing in years {null_yrs}"); score += 3
        rev_vals = [r["revenue"] for r in series]; yr_vals = [r["year"] for r in series]
        for i in range(len(rev_vals)):
            for j in range(i+2, len(rev_vals)):
                if rev_vals[i]==rev_vals[j]: flags.append(f"Identical REVENUE {rev_vals[i]:.4e} in {yr_vals[i]} and {yr_vals[j]}"); score += 10
        for i in range(1, len(series)):
            prev,curr = series[i-1]["revenue"],series[i]["revenue"]
            gap = series[i]["year"]-series[i-1]["year"]
            if prev!=0:
                chg = (curr-prev)/abs(prev)
                if abs(chg)>0.50*gap: flags.append(f"YoY {series[i-1]['year']}→{series[i]['year']} ({gap}-yr): {chg:+.0%}"); score += 5
        if flags:
            scored.append({"_score":score,"providerkey":str(pk),"company":company,"country":country,
                           "currency":currency,"industry":industry,"revenue_series":series,"flags":flags})
    scored.sort(key=lambda x: x["_score"], reverse=True)
    top = scored[:max_candidates]
    for c in top: del c["_score"]
    if not top: return issues

    SYSTEM_PROMPT = """You are a senior data quality analyst specialising in financial time-series data.
For each company record, evaluate whether flagged anomalies represent genuine data quality issues or plausible business events.
For EACH company output a JSON object with keys: "providerkey","company","verdict" (PLAUSIBLE/SUSPICIOUS/LIKELY_ERROR),"confidence" (HIGH/MEDIUM/LOW),"reasoning" (1-3 sentences),"recommended_action".
Return ONLY a valid JSON array — no markdown, no extra text."""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=LLM_MODEL, temperature=0,
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":json.dumps(top, indent=2)}])
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?","",raw); raw = re.sub(r"\n?```$","",raw.strip()).strip()
        verdicts = json.loads(raw)
    except Exception as e:
        issues.append(_issue("SOFT_FLAG","Plausibility","llm_error",f"LLM call failed: {e}"))
        return issues

    for v in verdicts:
        verdict  = v.get("verdict","UNKNOWN")
        severity = "HARD_REJECT" if verdict=="LIKELY_ERROR" else "SOFT_FLAG"
        pk = v.get("providerkey")
        matching = df.index[df["providerkey"].astype(str)==str(pk)].tolist()
        issues.append(_issue(severity,"Plausibility",f"llm_verdict_{verdict.lower()}",
                             f"[LLM|{verdict}|confidence={v.get('confidence')}] {v.get('company')} (pk={pk}): {v.get('reasoning')} → {v.get('recommended_action')}",
                             matching))
    return issues


def apply_flags_to_df(df, all_issues):
    df = df.copy()
    for col in DIMENSION_COL_MAP.values(): df[col] = ""
    row_flags = defaultdict(lambda: defaultdict(list))
    for iss in all_issues:
        col = DIMENSION_COL_MAP.get(iss["dimension"], f"flag_{iss['dimension'].lower()}")
        label = f"[{iss['severity']}] {iss['check']}"
        for idx in iss["affected_rows"]: row_flags[idx][col].append(label)
    for idx, dim_dict in row_flags.items():
        for col, labels in dim_dict.items(): df.at[idx, col] = " | ".join(labels)
    def row_severity(idx):
        has_hard = any(iss["severity"]=="HARD_REJECT" and idx in iss["affected_rows"] for iss in all_issues)
        has_soft = any(iss["severity"]=="SOFT_FLAG"   and idx in iss["affected_rows"] for iss in all_issues)
        if has_hard: return "HARD_REJECT"
        if has_soft: return "SOFT_FLAG"
        return ""
    df["flag_severity"]  = df.index.map(row_severity)
    df["flag_ANY_ISSUE"] = df["flag_severity"].map(lambda x: "YES" if x in ("HARD_REJECT","SOFT_FLAG") else "NO")
    return df


# STREAMLIT UI

# Header
st.markdown('<div class="app-title">DATA QUALITY <span>FRAMEWORK</span></div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Financial Revenue Dataset · 5-Dimension Quality Check</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# SIDEBAR 
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown('<div class="section-header">Thresholds</div>', unsafe_allow_html=True)

    revenue_null_pct_max = st.slider(
        "Max REVENUE null rate (%)", 0, 20, 5, 1,
        help="Dataset-level threshold for acceptable null revenue ratio"
    ) / 100.0

    revenue_max = st.number_input(
        "Revenue sanity cap ($)", value=10_000_000_000_000,
        step=1_000_000_000_000, format="%d",
        help="Rows with REVENUE exceeding this value are flagged"
    )

    yoy_threshold = st.slider(
        "YoY change threshold (% per year)", 10, 200, 50, 5,
        help="Max acceptable year-over-year revenue change per year of gap"
    ) / 100.0

    st.markdown('<div class="section-header">LLM Plausibility Check</div>', unsafe_allow_html=True)
    openai_key = st.text_input(
        "OpenAI API Key", type="password",
        placeholder="sk-...",
        help="Optional: enables GPT-4o-mini plausibility analysis"
    )
    llm_max_candidates = st.slider("Max LLM candidates", 5, 50, 20, 5) if openai_key else 20

    st.markdown('<div class="section-header">Checks to Run</div>', unsafe_allow_html=True)
    run_completeness = st.checkbox("1. Completeness", value=True)
    run_validity     = st.checkbox("2. Validity",     value=True)
    run_consistency  = st.checkbox("3. Consistency",  value=True)
    run_uniqueness   = st.checkbox("4. Uniqueness",   value=True)
    run_plausibility = st.checkbox("5. Plausibility (LLM)", value=bool(openai_key))

    st.markdown("---")
    st.markdown("""
    <div style='font-family: IBM Plex Mono, monospace; font-size: 0.65rem; color: #94a3b8;'>
    Supports .xlsx, .xls, .csv<br><br>
    Required columns:<br>
    timevalue · providerkey<br>
    companynameofficial<br>
    fiscalperiodend<br>
    operationstatustype<br>
    ipostatustype · geonameen<br>
    industrycode · REVENUE<br>
    unit_REVENUE
    </div>
    """, unsafe_allow_html=True)

# UPLOAD 
uploaded = st.file_uploader(
    "Drop your dataset here",
    type=["xlsx","xls","csv"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.markdown("""
    <div style='background:#ffffff;border:1px dashed #cbd5e1;border-radius:10px;
    padding:3rem;text-align:center;margin-top:1rem;box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.85rem;color:#94a3b8;'>
            ↑ Upload a dataset to begin quality analysis<br><br>
            <span style='color:#cbd5e1;font-size:0.75rem;'>Accepted formats: .xlsx  .xls  .csv</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# LOAD DATA 
@st.cache_data(show_spinner=False)
def load_file(file_bytes, file_name):
    ext = os.path.splitext(file_name)[1].lower()
    buf = io.BytesIO(file_bytes)
    if ext == ".csv":
        return pd.read_csv(buf)
    else:
        return pd.read_excel(buf)

with st.spinner("Loading dataset…"):
    df_raw = load_file(uploaded.read(), uploaded.name)

st.markdown(f"""
<div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#94a3b8;margin-bottom:1rem;'>
📂 {uploaded.name} &nbsp;·&nbsp; {len(df_raw):,} rows &nbsp;·&nbsp; {len(df_raw.columns)} columns
</div>
""", unsafe_allow_html=True)

# RUN CHECKS 
if st.button("▶  Run Quality Checks", type="primary"):
    all_issues = []

    progress_bar = st.progress(0, text="Running checks…")

    if run_completeness:
        progress_bar.progress(10, text="[1/5] Completeness…")
        all_issues.extend(check_completeness(df_raw, revenue_null_pct_max))

    if run_validity:
        progress_bar.progress(30, text="[2/5] Validity…")
        all_issues.extend(check_validity(df_raw, revenue_max))

    if run_consistency:
        progress_bar.progress(55, text="[3/5] Consistency…")
        all_issues.extend(check_consistency(df_raw, yoy_threshold))

    if run_uniqueness:
        progress_bar.progress(75, text="[4/5] Uniqueness…")
        all_issues.extend(check_uniqueness(df_raw))

    if run_plausibility:
        progress_bar.progress(85, text="[5/5] Plausibility (LLM)…")
        all_issues.extend(check_plausibility_llm(
            df_raw,
            api_key=openai_key or None,
            max_candidates=llm_max_candidates
        ))

    progress_bar.progress(100, text="Complete!")

    df_flagged = apply_flags_to_df(df_raw, all_issues)

    # Store in session state
    st.session_state["all_issues"]  = all_issues
    st.session_state["df_flagged"]  = df_flagged
    st.session_state["run_ts"]      = datetime.datetime.now().isoformat()

# RESULTS 
if "all_issues" not in st.session_state:
    st.stop()

all_issues = st.session_state["all_issues"]
df_flagged = st.session_state["df_flagged"]

hard_rejects = [x for x in all_issues if x["severity"]=="HARD_REJECT"]
soft_flags   = [x for x in all_issues if x["severity"]=="SOFT_FLAG"]
flagged_rows = df_flagged[df_flagged["flag_ANY_ISSUE"]=="YES"]

# DECISION BANNER 
if hard_rejects:
    st.markdown("""
    <div class="decision-blocked">
    ⛔ &nbsp; INGESTION BLOCKED &nbsp; — &nbsp; resolve HARD_REJECT issues before loading data
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="decision-proceed">
    ✅ &nbsp; PROCEED WITH FLAGS &nbsp; — &nbsp; no hard blockers detected; review soft flags
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# METRIC CARDS 
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card info">
        <div class="metric-label">Total Issues</div>
        <div class="metric-value">{len(all_issues)}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card hard">
        <div class="metric-label">Hard Rejects</div>
        <div class="metric-value">{len(hard_rejects)}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-label-card metric-card soft">
        <div class="metric-label">Soft Flags</div>
        <div class="metric-value">{len(soft_flags)}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    pct = len(flagged_rows)/len(df_flagged)*100 if len(df_flagged) else 0
    card_class = "ok" if pct == 0 else ("hard" if pct > 50 else "soft")
    st.markdown(f"""<div class="metric-card {card_class}">
        <div class="metric-label">Rows Flagged</div>
        <div class="metric-value">{len(flagged_rows)}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# DIMENSION BREAKDOWN  
st.markdown('<div class="section-header">Issues by Dimension</div>', unsafe_allow_html=True)

dim_tabs = st.tabs(["All"] + list(DIMENSION_COL_MAP.keys()))

def render_issues(issues_list):
    if not issues_list:
        st.markdown("<span style='color:#94a3b8;font-family:IBM Plex Mono,monospace;font-size:0.8rem;'>No issues found in this dimension.</span>", unsafe_allow_html=True)
        return
    for iss in issues_list:
        badge_class = "badge-hard" if iss["severity"]=="HARD_REJECT" else "badge-soft"
        badge_text  = "🔴 HARD_REJECT" if iss["severity"]=="HARD_REJECT" else "🟡 SOFT_FLAG"
        n_rows = len(iss["affected_rows"])
        row_txt = f"· {n_rows} row(s)" if n_rows else ""
        st.markdown(f"""
        <div class="issue-row">
            <div class="issue-check">
                <span class="{badge_class}">{badge_text}</span>&nbsp;&nbsp;
                <span style='color:#2563eb'>{iss['dimension']}</span>&nbsp;›&nbsp;{iss['check']}
                &nbsp;<span style='color:#cbd5e1;font-size:0.7rem;'>{row_txt}</span>
            </div>
            <div class="issue-detail">{iss['detail']}</div>
        </div>""", unsafe_allow_html=True)

with dim_tabs[0]:
    render_issues(all_issues)

for i, (dim, col) in enumerate(DIMENSION_COL_MAP.items()):
    with dim_tabs[i+1]:
        dim_issues = [x for x in all_issues if x["dimension"]==dim]
        n_flagged  = (df_flagged[col]!="").sum()
        st.markdown(f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#94a3b8;'>{n_flagged} rows flagged in this dimension</span>", unsafe_allow_html=True)
        render_issues(dim_issues)

# FLAGGED DATA TABLE 
st.markdown('<div class="section-header">Flagged Rows</div>', unsafe_allow_html=True)

show_all = st.checkbox("Show all rows (including clean)", value=False)
display_df = df_flagged if show_all else flagged_rows

if display_df.empty:
    st.success("No rows flagged — dataset is clean!")
else:
    # Highlight severity
    def color_severity(val):
        if val == "HARD_REJECT": return "background-color:#fee2e2;color:#b91c1c"
        if val == "SOFT_FLAG":   return "background-color:#fef3c7;color:#92400e"
        return ""

    styled = display_df.style.applymap(color_severity, subset=["flag_severity"])
    st.dataframe(styled, use_container_width=True, height=400)

# DOWNLOADS 
st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)

col_dl1, col_dl2, col_dl3 = st.columns(3)

with col_dl1:
    excel_buf = io.BytesIO()
    df_flagged.to_excel(excel_buf, index=False)
    excel_buf.seek(0)
    st.download_button(
        "📥 Download Flagged Excel",
        data=excel_buf,
        file_name="quality_flagged.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with col_dl2:
    report = {
        "run_timestamp":      st.session_state["run_ts"],
        "dataset_shape":      {"rows": len(df_raw), "columns": len(df_raw.columns)},
        "total_issues":       len(all_issues),
        "hard_reject_count":  len(hard_rejects),
        "soft_flag_count":    len(soft_flags),
        "ingestion_decision": "BLOCKED" if hard_rejects else "PROCEED_WITH_FLAGS",
        "issues":             all_issues,
    }
    st.download_button(
        "📋 Download JSON Report",
        data=json.dumps(report, indent=2, default=str),
        file_name="quality_report.json",
        mime="application/json",
    )

with col_dl3:
    flagged_only_buf = io.BytesIO()
    flagged_rows.to_excel(flagged_only_buf, index=False)
    flagged_only_buf.seek(0)
    st.download_button(
        "⚠️ Download Flagged Rows Only",
        data=flagged_only_buf,
        file_name="quality_flagged_rows_only.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

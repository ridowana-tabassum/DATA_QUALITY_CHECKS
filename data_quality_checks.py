import os
import re
import sys
import json
import warnings
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# CONFIGURATION

INPUT_FILE  = "CaseStudy_Quality_sample25.xlsx"
OUTPUT_FILE = "CaseStudy_Quality_sample25_flagged.xlsx"
REPORT_FILE = "quality_report.json"

REQUIRED_COLUMNS = [
    "timevalue", "providerkey", "companynameofficial",
    "fiscalperiodend", "operationstatustype", "ipostatustype",
    "geonameen", "industrycode", "REVENUE", "unit_REVENUE",
]

VALID_OPERATION_STATUSES = {"ACTIVE", "INACTIVE", "DISSOLVED", "BANKRUPT", "RESTRUCTURING"}
VALID_IPO_STATUSES        = {"PUBLIC", "PRIVATE", "DELISTED", "PENDING"}
CURRENT_YEAR              = datetime.datetime.now().year

# Revenue thresholds
REVENUE_YOY_THRESHOLD_PER_YEAR = 0.50   # 50% per year (gap-adjusted)
REVENUE_MAX                    = 10e12  # 10 trillion sanity cap
REVENUE_NULL_PCT_MAX           = 0.05   # dataset-level null threshold

# LLM settings
LLM_MAX_CANDIDATES = 20
LLM_MODEL          = "gpt-4o-mini"

# Country → expected currency
COUNTRY_CURRENCY_MAP = {
    "United Kingdom": "GBP",
    "India":          "INR",
    "United States":  "USD",
    "Germany":        "EUR",
    "France":         "EUR",
    "Japan":          "JPY",
    "Sweden":         "SEK",
    "Denmark":        "DKK",
    "Indonesia":      "IDR",
}

# Regex patterns
FISCALPERIODEND_REGEX = re.compile(r"^\d{2}-[A-Za-z]{3}$")
INDUSTRYCODE_REGEX    = re.compile(r"^\d{3,4}\s-\s.+$")
CURRENCY_REGEX        = re.compile(r"^[A-Z]{3}$")

VALID_MONTH_ABBREVS = {
    "Jan","Feb","Mar","Apr","May","Jun",
    "Jul","Aug","Sep","Oct","Nov","Dec",
}
MONTH_MAX_DAYS = {
    "Jan":31,"Feb":29,"Mar":31,"Apr":30,"May":31,"Jun":30,
    "Jul":31,"Aug":31,"Sep":30,"Oct":31,"Nov":30,"Dec":31,
}

# Priority scores for LLM candidate selection
FLAG_PRIORITY = {
    "exact_repeat":    10,
    "zero_revenue":     8,
    "negative":         8,
    "yoy_spike":        5,
    "missing_revenue":  3,
}


# LOAD DATA

def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .xlsx, .xls, or .csv.")
    print(f"Loaded {len(df)} rows × {len(df.columns)} columns.")
    return df


# INTERNAL ISSUE HELPER

def _issue(severity, dimension, check, detail, rows=None):
    return {
        "severity":      severity,
        "dimension":     dimension,
        "check":         check,
        "detail":        detail,
        "affected_rows": rows or [],
    }



# CHECK 1 – COMPLETENESS

def check_completeness(df: pd.DataFrame):
    issues = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(_issue("HARD_REJECT", "Completeness", "missing_columns",
                             f"Required columns absent: {missing_cols}"))
        return issues

    null_name = df.index[df["companynameofficial"].isnull()].tolist()
    if null_name:
        issues.append(_issue("HARD_REJECT", "Completeness", "null_companynameofficial",
                             f"{len(null_name)} row(s) have no companynameofficial", null_name))

    rev_null_pct = df["REVENUE"].isnull().mean()
    if rev_null_pct > REVENUE_NULL_PCT_MAX:
        issues.append(_issue("SOFT_FLAG", "Completeness", "high_revenue_null_pct",
                             f"REVENUE null rate = {rev_null_pct:.1%} "
                             f"(threshold {REVENUE_NULL_PCT_MAX:.0%})"))

    active_no_rev = df.index[
        (df["operationstatustype"] == "ACTIVE") & df["REVENUE"].isnull()
    ].tolist()
    if active_no_rev:
        issues.append(_issue("SOFT_FLAG", "Completeness", "active_missing_revenue",
                             f"{len(active_no_rev)} ACTIVE row(s) missing REVENUE",
                             active_no_rev))

    public_no_rev = df.index[
        (df["ipostatustype"] == "PUBLIC") & df["REVENUE"].isnull()
    ].tolist()
    if public_no_rev:
        issues.append(_issue("SOFT_FLAG", "Completeness", "public_missing_revenue",
                             f"{len(public_no_rev)} PUBLIC row(s) missing REVENUE",
                             public_no_rev))

    rev_no_unit = df.index[df["REVENUE"].notna() & df["unit_REVENUE"].isnull()].tolist()
    unit_no_rev = df.index[df["unit_REVENUE"].notna() & df["REVENUE"].isnull()].tolist()
    if rev_no_unit:
        issues.append(_issue("SOFT_FLAG", "Completeness", "revenue_without_unit",
                             f"{len(rev_no_unit)} row(s) have REVENUE but no unit_REVENUE",
                             rev_no_unit))
    if unit_no_rev:
        issues.append(_issue("SOFT_FLAG", "Completeness", "unit_without_revenue",
                             f"{len(unit_no_rev)} row(s) have unit_REVENUE but no REVENUE",
                             unit_no_rev))

    flagged = sum(len(i["affected_rows"]) for i in issues)
    print(f"[Check 1] Completeness: {len(issues)} issue type(s) found, "
          f"{flagged} total row references.")
    return issues



# CHECK 2 – VALIDITY

def _parse_fiscalperiodend(val):
    if pd.isnull(val):
        return False, False
    if isinstance(val, (datetime.datetime, datetime.date)):
        return True, val.year > CURRENT_YEAR
    s = str(val).strip()
    if not FISCALPERIODEND_REGEX.match(s):
        return False, False
    day_str, month_abbr = s.split("-")
    day = int(day_str)
    if month_abbr not in VALID_MONTH_ABBREVS:
        return False, False
    if day < 1 or day > MONTH_MAX_DAYS.get(month_abbr, 31):
        return False, False
    return True, False


def check_validity(df: pd.DataFrame):
    issues = []

    def valid_year(x):
        try:
            return (not pd.isnull(x)) and (x == int(x)) and (1900 <= int(x) <= CURRENT_YEAR)
        except Exception:
            return False

    bad_tv = df.index[~df["timevalue"].apply(valid_year)].tolist()
    if bad_tv:
        issues.append(_issue("HARD_REJECT", "Validity", "invalid_timevalue",
                             f"{len(bad_tv)} row(s) with out-of-range or non-integer timevalue",
                             bad_tv))

    parsed      = df["fiscalperiodend"].apply(_parse_fiscalperiodend)
    bad_format  = df.index[~parsed.apply(lambda t: t[0])].tolist()
    future_date = df.index[parsed.apply(lambda t: t[0] and t[1])].tolist()
    if bad_format:
        issues.append(_issue("SOFT_FLAG", "Validity", "invalid_fiscalperiodend_format",
                             f"{len(bad_format)} row(s) with unrecognised fiscalperiodend "
                             "(expected DD-Mon, e.g. 31-May)", bad_format))
    if future_date:
        issues.append(_issue("SOFT_FLAG", "Validity", "fiscalperiodend_future_date",
                             f"{len(future_date)} row(s) have fiscalperiodend year > {CURRENT_YEAR}",
                             future_date))

    def valid_industry(x):
        if pd.isnull(x):
            return False
        return bool(INDUSTRYCODE_REGEX.match(str(x).strip()))

    bad_ic = df.index[~df["industrycode"].apply(valid_industry)].tolist()
    if bad_ic:
        issues.append(_issue("HARD_REJECT", "Validity", "invalid_industrycode_format",
                             f"{len(bad_ic)} row(s) with malformed industrycode "
                             r"(expected: \d{3,4} - Description)", bad_ic))

    bad_ops = df.index[~df["operationstatustype"].isin(VALID_OPERATION_STATUSES)].tolist()
    if bad_ops:
        vals = df.loc[bad_ops, "operationstatustype"].unique().tolist()
        issues.append(_issue("HARD_REJECT", "Validity", "invalid_operationstatustype",
                             f"Unknown operationstatustype value(s): {vals}", bad_ops))

    bad_ipo = df.index[~df["ipostatustype"].isin(VALID_IPO_STATUSES)].tolist()
    if bad_ipo:
        vals = df.loc[bad_ipo, "ipostatustype"].unique().tolist()
        issues.append(_issue("HARD_REJECT", "Validity", "invalid_ipostatustype",
                             f"Unknown ipostatustype value(s): {vals}", bad_ipo))

    has_unit = df["unit_REVENUE"].notna()
    bad_cur = df.index[
        has_unit & ~df["unit_REVENUE"].apply(
            lambda x: bool(CURRENCY_REGEX.match(str(x).strip())) if pd.notna(x) else False
        )
    ].tolist()
    if bad_cur:
        issues.append(_issue("SOFT_FLAG", "Validity", "invalid_currency_code",
                             f"{len(bad_cur)} row(s) with non-ISO unit_REVENUE value", bad_cur))

    has_rev  = df["REVENUE"].notna()
    neg_rev  = df.index[has_rev & (df["REVENUE"] < 0)].tolist()
    huge_rev = df.index[has_rev & (df["REVENUE"] > REVENUE_MAX)].tolist()
    if neg_rev:
        issues.append(_issue("SOFT_FLAG", "Validity", "negative_revenue",
                             f"{len(neg_rev)} row(s) with REVENUE < 0", neg_rev))
    if huge_rev:
        issues.append(_issue("SOFT_FLAG", "Validity", "revenue_exceeds_cap",
                             f"{len(huge_rev)} row(s) with REVENUE > {REVENUE_MAX:.0e}", huge_rev))

    flagged = sum(len(i["affected_rows"]) for i in issues)
    print(f"[Check 2] Validity: {len(issues)} issue type(s) found, "
          f"{flagged} total row references.")
    return issues



# CHECK 3 – CONSISTENCY

def check_consistency(df: pd.DataFrame):
    issues = []

    global_min_year = int(df["timevalue"].dropna().min())
    global_max_year = int(df["timevalue"].dropna().max())

    # 3-A  Entity attributes must be stable per providerkey
    entity_attrs = ["companynameofficial", "industrycode", "fiscalperiodend",
                    "geonameen", "ipostatustype"]
    for attr in entity_attrs:
        for pk, grp in df.groupby("providerkey"):
            unique_vals = grp[attr].dropna().unique()
            if len(unique_vals) > 1:
                sev = "HARD_REJECT" if attr in ("companynameofficial", "industrycode") else "SOFT_FLAG"
                issues.append(_issue(
                    sev, "Consistency", f"entity_{attr}_conflict",
                    f"providerkey {pk} has multiple {attr} values: {unique_vals.tolist()}",
                    grp.index.tolist(),
                ))

    # 3-B  Currency vs country mismatch — one flag per company
    has_both = df["unit_REVENUE"].notna() & df["geonameen"].notna()
    for pk, grp in df[has_both].groupby("providerkey"):
        country  = grp["geonameen"].iloc[0]
        expected = COUNTRY_CURRENCY_MAP.get(country)
        if not expected:
            continue
        non_expected = [c for c in grp["unit_REVENUE"].dropna().unique() if c != expected]
        if non_expected:
            issues.append(_issue(
                "SOFT_FLAG", "Consistency", "currency_country_mismatch",
                f"{grp['companynameofficial'].iloc[0]} (pk={pk}, {country}): "
                f"expected {expected}, reports in {non_expected}",
                grp.index.tolist(),
            ))

    # 3-C  YoY spike with gap-adjusted threshold + zero-revenue flag
    for pk, grp in df.groupby("providerkey"):
        grp_sorted = grp.sort_values("timevalue")
        company    = grp_sorted["companynameofficial"].iloc[0]
        is_active  = grp_sorted["operationstatustype"].eq("ACTIVE").all()
        is_public  = grp_sorted["ipostatustype"].eq("PUBLIC").all()

        # Zero revenue for ACTIVE PUBLIC is anomalous
        zero_rows = grp_sorted.index[
            grp_sorted["REVENUE"].notna() & (grp_sorted["REVENUE"] == 0)
        ].tolist()
        if zero_rows and is_active and is_public:
            zero_yrs = grp_sorted.loc[zero_rows, "timevalue"].tolist()
            issues.append(_issue(
                "SOFT_FLAG", "Consistency", "zero_revenue_active_public",
                f"providerkey={pk} ({company}): REVENUE = 0 in year(s) {zero_yrs} "
                "for an ACTIVE PUBLIC company",
                zero_rows,
            ))

        # Gap-adjusted YoY threshold
        grp_rev  = grp_sorted.dropna(subset=["REVENUE"])
        if len(grp_rev) < 2:
            continue
        rev_vals = grp_rev["REVENUE"].values
        yr_vals  = grp_rev["timevalue"].values
        idxs     = grp_rev.index.tolist()

        for i in range(1, len(rev_vals)):
            prev, curr = rev_vals[i - 1], rev_vals[i]
            year_gap   = int(yr_vals[i]) - int(yr_vals[i - 1])
            if prev == 0:
                continue
            pct_change         = (curr - prev) / abs(prev)
            adjusted_threshold = REVENUE_YOY_THRESHOLD_PER_YEAR * year_gap
            if abs(pct_change) > adjusted_threshold:
                issues.append(_issue(
                    "SOFT_FLAG", "Consistency", "revenue_yoy_spike",
                    f"providerkey={pk} ({company}): "
                    f"{int(yr_vals[i-1])}→{int(yr_vals[i])} ({year_gap}-yr gap) "
                    f"revenue changed {pct_change:+.0%} "
                    f"(threshold ±{adjusted_threshold:.0%}, {prev:.2e}→{curr:.2e})",
                    [idxs[i - 1], idxs[i]],
                ))

    # 3-D  Exact revenue repeat across non-adjacent years (copy-paste error indicator)
    for pk, grp in df.groupby("providerkey"):
        grp_sorted = grp.sort_values("timevalue").dropna(subset=["REVENUE"])
        if len(grp_sorted) < 3:
            continue
        rev_list = grp_sorted["REVENUE"].values
        yr_list  = grp_sorted["timevalue"].values
        idx_list = grp_sorted.index.tolist()
        reported = set()
        for i in range(len(rev_list)):
            for j in range(i + 2, len(rev_list)):
                key = (round(rev_list[i], 2), int(yr_list[i]), int(yr_list[j]))
                if rev_list[i] == rev_list[j] and key not in reported:
                    reported.add(key)
                    issues.append(_issue(
                        "SOFT_FLAG", "Consistency", "exact_revenue_repeat_non_adjacent",
                        f"providerkey={pk} ({grp_sorted['companynameofficial'].iloc[0]}): "
                        f"identical REVENUE {rev_list[i]:.4e} in years "
                        f"{int(yr_list[i])} and {int(yr_list[j])} (possible copy error)",
                        [idx_list[i], idx_list[j]],
                    ))

    # 3-E  Missing years in series (within company's own span)
    for pk, grp in df.groupby("providerkey"):
        years    = sorted(grp["timevalue"].dropna().astype(int).tolist())
        if len(years) < 2:
            continue
        span_min = max(min(years), global_min_year)
        span_max = min(max(years), global_max_year)
        missing  = sorted(set(range(span_min, span_max + 1)) - set(years))
        if missing:
            issues.append(_issue(
                "SOFT_FLAG", "Consistency", "missing_years_in_series",
                f"providerkey={pk} ({grp['companynameofficial'].iloc[0]}): "
                f"gap year(s) {missing} within reported span {span_min}–{span_max}",
                grp.index.tolist(),
            ))

    flagged = sum(len(i["affected_rows"]) for i in issues)
    print(f"[Check 3] Consistency: {len(issues)} issue type(s) found, "
          f"{flagged} total row references.")
    return issues



# CHECK 4 – UNIQUENESS

def check_uniqueness(df: pd.DataFrame):
    issues = []

    dupes = df[df.duplicated(subset=["providerkey", "timevalue"], keep=False)]
    if not dupes.empty:
        issues.append(_issue(
            "HARD_REJECT", "Uniqueness", "duplicate_primary_key",
            f"{len(dupes)} row(s) share a (providerkey, timevalue) pair",
            dupes.index.tolist(),
        ))

    exact_dupes = df[df.duplicated(keep=False)]
    if not exact_dupes.empty:
        issues.append(_issue(
            "HARD_REJECT", "Uniqueness", "exact_duplicate_rows",
            f"{len(exact_dupes)} row(s) are completely identical",
            exact_dupes.index.tolist(),
        ))

    name_to_keys = df.dropna(subset=["companynameofficial"]).groupby(
        "companynameofficial"
    )["providerkey"].nunique()
    for name in name_to_keys[name_to_keys > 1].index.tolist():
        keys = df[df["companynameofficial"] == name]["providerkey"].unique().tolist()
        issues.append(_issue(
            "SOFT_FLAG", "Uniqueness", "company_name_maps_to_multiple_keys",
            f"'{name}' maps to multiple providerkeys: {keys}",
        ))

    flagged = sum(len(i["affected_rows"]) for i in issues)
    print(f"[Check 4] Uniqueness: {len(issues)} issue type(s) found, "
          f"{flagged} total row references.")
    return issues


# CHECK 5 – PLAUSIBILITY (LLM-assisted via OpenAI GPT)

def _build_llm_candidates(df: pd.DataFrame):
    scored = []

    for pk, grp in df.groupby("providerkey"):
        grp_s    = grp.sort_values("timevalue")
        company  = grp_s["companynameofficial"].iloc[0]
        industry = grp_s["industrycode"].iloc[0]
        currency = (grp_s["unit_REVENUE"].dropna().iloc[0]
                    if grp_s["unit_REVENUE"].notna().any() else "?")
        country  = grp_s["geonameen"].iloc[0]

        rev_rows = grp_s.dropna(subset=["REVENUE"])[["timevalue", "REVENUE"]]
        series   = [{"year": int(r.timevalue), "revenue": float(r.REVENUE)}
                    for _, r in rev_rows.iterrows()]

        flags = []
        score = 0

        neg = [r for r in series if r["revenue"] < 0]
        if neg:
            flags.append(f"Negative revenue: {neg}")
            score += FLAG_PRIORITY["negative"]

        if (grp_s["operationstatustype"].eq("ACTIVE").all() and
                grp_s["ipostatustype"].eq("PUBLIC").all()):
            zero = [r for r in series if r["revenue"] == 0]
            if zero:
                flags.append(f"Zero revenue (ACTIVE PUBLIC): {zero}")
                score += FLAG_PRIORITY["zero_revenue"]
            null_yrs = grp_s.loc[grp_s["REVENUE"].isnull(), "timevalue"].tolist()
            if null_yrs:
                flags.append(f"REVENUE missing in years {null_yrs}")
                score += FLAG_PRIORITY["missing_revenue"]

        rev_vals = [r["revenue"] for r in series]
        yr_vals  = [r["year"]    for r in series]
        for i in range(len(rev_vals)):
            for j in range(i + 2, len(rev_vals)):
                if rev_vals[i] == rev_vals[j]:
                    flags.append(
                        f"Identical REVENUE {rev_vals[i]:.4e} "
                        f"in {yr_vals[i]} and {yr_vals[j]}"
                    )
                    score += FLAG_PRIORITY["exact_repeat"]

        for i in range(1, len(series)):
            prev, curr = series[i - 1]["revenue"], series[i]["revenue"]
            gap = series[i]["year"] - series[i - 1]["year"]
            if prev != 0:
                chg = (curr - prev) / abs(prev)
                if abs(chg) > REVENUE_YOY_THRESHOLD_PER_YEAR * gap:
                    flags.append(
                        f"YoY {series[i-1]['year']}→{series[i]['year']} "
                        f"({gap}-yr gap): {chg:+.0%} ({prev:.2e}→{curr:.2e})"
                    )
                    score += FLAG_PRIORITY["yoy_spike"]

        if flags:
            scored.append({
                "_score":         score,
                "providerkey":    str(pk),
                "company":        company,
                "country":        country,
                "currency":       currency,
                "industry":       industry,
                "revenue_series": series,
                "flags":          flags,
            })

    scored.sort(key=lambda x: x["_score"], reverse=True)
    top = scored[:LLM_MAX_CANDIDATES]
    for c in top:
        del c["_score"]
    return top


SYSTEM_PROMPT = """
You are a senior data quality analyst specialising in financial time-series data.
You will receive a JSON array of company records pre-flagged by automated rule checks.
For each record, evaluate whether the flagged anomalies represent genuine data quality
issues (wrong extraction, copy-paste errors, decimal shifts, implausible values) or
plausible business events (divestiture, acquisition, one-off loss, FX translation,
Covid-19 impact in 2020, spin-off).

For EACH company output a JSON object with these exact keys:
  "providerkey"        : the identifier provided
  "company"            : company name
  "verdict"            : one of ["PLAUSIBLE", "SUSPICIOUS", "LIKELY_ERROR"]
  "confidence"         : one of ["HIGH", "MEDIUM", "LOW"]
  "reasoning"          : 1-3 sentence explanation referencing the specific flag(s)
  "recommended_action" : e.g. "Accept", "Flag for manual review", "Reject and re-extract"

Guidance:
  - A 2020 revenue drop is often PLAUSIBLE (Covid-19).
  - An exact revenue repeat 4 years apart after a decline is LIKELY_ERROR.
  - A 99% drop to near-zero is SUSPICIOUS unless the company divested its main business.
  - Zero revenue for an investment trust (industry 6430) is PLAUSIBLE.

Return ONLY a valid JSON array — no markdown, no extra text.
""".strip()


def check_plausibility_llm(df: pd.DataFrame, api_key=None):
    issues = []

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        issues.append(_issue(
            "SOFT_FLAG", "Plausibility", "llm_skipped",
            "No OPENAI_API_KEY found – LLM plausibility check skipped. "
            "Set the environment variable to enable it.",
        ))
        print("[Check 5] Plausibility: LLM check skipped (no API key).")
        return issues

    try:
        from openai import OpenAI
    except ImportError:
        issues.append(_issue(
            "SOFT_FLAG", "Plausibility", "llm_skipped",
            "openai package not installed. Run: pip install openai",
        ))
        print("[Check 5] Plausibility: openai package not found – skipping.")
        return issues

    candidates = _build_llm_candidates(df)
    if not candidates:
        print("[Check 5] Plausibility: No candidates flagged for LLM review.")
        return issues

    client = OpenAI(api_key=api_key)
    print(f"[Check 5] Plausibility: Sending {len(candidates)} candidate(s) "
          f"(top {LLM_MAX_CANDIDATES} by severity) to {LLM_MODEL} …")

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(candidates, indent=2)},
            ],
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip()).strip()
        if not raw:
            issues.append(_issue("SOFT_FLAG", "Plausibility", "llm_parse_error",
                                 "GPT returned an empty response"))
            return issues
        verdicts = json.loads(raw)
    except json.JSONDecodeError as e:
        issues.append(_issue("SOFT_FLAG", "Plausibility", "llm_parse_error",
                             f"GPT response was not valid JSON: {e}"))
        return issues
    except Exception as e:
        issues.append(_issue("SOFT_FLAG", "Plausibility", "llm_api_error",
                             f"OpenAI API call failed: {e}"))
        return issues

    for v in verdicts:
        verdict  = v.get("verdict", "UNKNOWN")
        severity = "HARD_REJECT" if verdict == "LIKELY_ERROR" else "SOFT_FLAG"
        pk       = v.get("providerkey")
        matching = df.index[df["providerkey"].astype(str) == str(pk)].tolist()
        issues.append(_issue(
            severity, "Plausibility", f"llm_verdict_{verdict.lower()}",
            detail=(
                f"[LLM | {verdict} | confidence={v.get('confidence')}] "
                f"{v.get('company')} (pk={pk}): {v.get('reasoning')} "
                f"→ {v.get('recommended_action')}"
            ),
            rows=matching,
        ))

    flagged = sum(len(i["affected_rows"]) for i in issues)
    print(f"[Check 5] Plausibility: {len(issues)} LLM verdict(s) issued, "
          f"{flagged} total row references.")
    return issues


# APPLY FLAGS TO DATAFRAME (Excel output columns)

DIMENSION_COL_MAP = {
    "Completeness": "flag_completeness",
    "Validity":     "flag_validity",
    "Consistency":  "flag_consistency",
    "Uniqueness":   "flag_uniqueness",
    "Plausibility": "flag_llm",
}


def apply_flags_to_df(df: pd.DataFrame, all_issues: list) -> pd.DataFrame:
    df = df.copy()

    # Initialize flag columns
    for col in DIMENSION_COL_MAP.values():
        df[col] = ""

    # Collect per-row, per-dimension messages
    row_flags = defaultdict(lambda: defaultdict(list))
    for iss in all_issues:
        col = DIMENSION_COL_MAP.get(iss["dimension"], f"flag_{iss['dimension'].lower()}")
        label = f"[{iss['severity']}] {iss['check']}"
        for idx in iss["affected_rows"]:
            row_flags[idx][col].append(label)

    # Write back to DataFrame
    for idx, dim_dict in row_flags.items():
        for col, labels in dim_dict.items():
            df.at[idx, col] = " | ".join(labels)

    # Overall severity column
    def row_severity(idx):
        has_hard = any(
            iss["severity"] == "HARD_REJECT" and idx in iss["affected_rows"]
            for iss in all_issues
        )
        has_soft = any(
            iss["severity"] == "SOFT_FLAG" and idx in iss["affected_rows"]
            for iss in all_issues
        )
        if has_hard:
            return "HARD_REJECT"
        if has_soft:
            return "SOFT_FLAG"
        return ""

    df["flag_severity"]  = df.index.map(row_severity)
    df["flag_ANY_ISSUE"] = df["flag_severity"].map(
        lambda x: "YES" if x in ("HARD_REJECT", "SOFT_FLAG") else "NO"
    )

    return df


# SUMMARY REPORT

def print_summary(all_issues: list, df: pd.DataFrame):
    hard_rejects = [x for x in all_issues if x["severity"] == "HARD_REJECT"]
    soft_flags   = [x for x in all_issues if x["severity"] == "SOFT_FLAG"]

    print("\n" + "═" * 70)
    print("  QUALITY FLAG SUMMARY")
    print("═" * 70)
    print(f"  Total issue types   : {len(all_issues)}")
    print(f"  🔴 HARD_REJECT      : {len(hard_rejects)}  (block insert)")
    print(f"  🟡 SOFT_FLAG        : {len(soft_flags)}  (insert but mark)")

    flagged_rows = df[df["flag_ANY_ISSUE"] == "YES"]
    print(f"\n  Rows flagged (any)  : {len(flagged_rows)} / {len(df)}")

    for dim, col in DIMENSION_COL_MAP.items():
        n = (df[col] != "").sum()
        print(f"  {dim:<16}: {n} rows")

    if hard_rejects:
        print("\n  ⛔  DATASET BLOCKED – resolve HARD_REJECT issues before ingestion.")
    else:
        print("\n  ✅  Dataset may proceed to ingestion (review SOFT_FLAGs).")
    print("═" * 70)


# MAIN

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File not found: {INPUT_FILE}")
        sys.exit(1)

    df = load_data(INPUT_FILE)

    # Run all checks
    all_issues = []
    all_issues.extend(check_completeness(df))
    all_issues.extend(check_validity(df))
    all_issues.extend(check_consistency(df))
    all_issues.extend(check_uniqueness(df))
    all_issues.extend(check_plausibility_llm(df))

    # Apply flags and write Excel output
    df_flagged = apply_flags_to_df(df, all_issues)
    df_flagged.to_excel(OUTPUT_FILE, index=False)
    print(f"\nFlagged Excel saved → {OUTPUT_FILE}")

    # Save JSON report
    hard_rejects = [x for x in all_issues if x["severity"] == "HARD_REJECT"]
    report = {
        "run_timestamp":      datetime.datetime.now().isoformat(),
        "dataset_shape":      {"rows": len(df), "columns": len(df.columns)},
        "total_issues":       len(all_issues),
        "hard_reject_count":  len(hard_rejects),
        "soft_flag_count":    len(all_issues) - len(hard_rejects),
        "ingestion_decision": "BLOCKED" if hard_rejects else "PROCEED_WITH_FLAGS",
        "issues":             all_issues,
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Quality report saved → {REPORT_FILE}")

    print_summary(all_issues, df_flagged)


if __name__ == "__main__":
    main()

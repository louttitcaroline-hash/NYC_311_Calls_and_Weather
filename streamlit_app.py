# ==============================
# Point72 Business Analyst Case Study
# ==============================

# ==============================
# Imports
# ==============================
import pandas as pd
import numpy as np
from pathlib import Path
import calendar
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import BallTree
from collections import Counter
import numpy as _np
from sklearn.linear_model import LinearRegression
import json
import re
from sklearn.metrics import r2_score
from datetime import timedelta

# ==============================
# Streamlit Page Config
# ==============================
# Page configuration must be set before any Streamlit commands
st.set_page_config(
    page_title="Point72 Business Analyst Case Study",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* Headings */
h1, h2, h3 { font-family: 'Playfair Display', serif; font-weight: 500; letter-spacing: -0.02em; }
/* Body text */
p, div, span, label { font-family: 'Inter', sans-serif; font-weight: 400; line-height: 1.6; }
/* Metrics */
[data-testid="stMetricValue"] { font-family: 'Playfair Display', serif; font-weight: 600; }
/* App background */
.stApp, section[data-testid="stMain"], section[data-testid="stSidebar"] { background-color: #f5f4ee; }
/* Visualization cards */
.viz-card, [data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stMetric"], [data-testid="stDeckGlJsonChart"], [data-testid="stImage"] {
    background-color: #fafaf7 !important;
    border: 1px solid #d2cdc2 !important;
    border-radius: 14px !important;
    padding: 0.75rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}

/* Ensure inner Plotly/DataFrame containers inherit rounded corners and clip overflow */
[data-testid="stPlotlyChart"] > div, [data-testid="stDataFrame"] > div, [data-testid="stImage"] > div {
    padding: 0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Target Plotly/Mapbox canvases specifically to prevent bleed-out */
.js-plotly-plot, .mapboxgl-canvas, .mapboxgl-canvas-container, .plotly { border-radius: 12px !important; overflow: hidden !important; }

/* DataFrame table visuals: ensure box-sizing so borders and padding don't overflow */
[data-testid="stDataFrame"] table { box-sizing: border-box; width: 100% !important; }

[data-testid="stMetricValue"] { background-color: transparent !important; padding: 0.125rem 0.25rem !important; }
.stAlert, .stException, .stSpinner { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ==============================
# Title & Overview
# ==============================
st.title("📞 NYC 311 Service Requests & Weather Analysis 🌦️")
st.markdown(
"<div style='font-size:12px'><a href='https://grnhse-use1-prod-s2-ghr.s3.amazonaws.com/generic_attachments/attachments/554/877/700/original/311_2018.csv' target='_blank'>NYC 311 Service Requests Dataset (2018)</a></div>",
unsafe_allow_html=True
)
st.markdown(
"<div style='font-size:12px'><a href='https://grnhse-use1-prod-s2-ghr.s3.amazonaws.com/generic_attachments/attachments/554/877/600/original/weather_df_2018.csv' target='_blank'>NYC Weather Dataset (2018, daily data)</a></div>",
unsafe_allow_html=True
)

# =============================
# Tabs
# ============================
tab_preparation_calls, tab_preparation_weather, tab_descriptive_calls, tab_descriptive_weather, tab_merged, tab_forecasting = st.tabs([
    "🧹 Data Prep | 311 Service Requests",
    "🧹 Data Prep | Weather",
    "📞 Analysis | 311 Service Requests",
    "🌦️ Analysis | Weather",
    "🎯 Weather Impact on 311 Service Requests",
    "📈 Forecasting"
])

# ==============================
# Load Datasets
# ==============================
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path)

calls_file = Path("data/311_2018.csv")
weather_file = Path("data/weather_df_2018.csv")

calls = load_csv(calls_file)
weather = load_csv(weather_file)

# ==============================
# Data Preparation
# ==============================
def summarize_df(df: pd.DataFrame, name: str):
    st.write(f"Shape: {df.shape}")
    
    # Layout-safe summary: fix heights to avoid overlap and add a small spacer below
    c0, c1, c2 = st.columns([2, 1, 1])
    with c0:
        st.write("Columns and summary (all dtypes):")
        st.dataframe(df.describe(include="all").T, height=420)
    with c1:
        st.write("Missing % (cols with >0):")
        missing = df.isna().mean().sort_values(ascending=False)
        if (missing > 0).any():
            st.dataframe(missing[missing > 0].to_frame(name="missing_fraction"), height=420)
        else:
            st.write("None")
    with c2:
        st.write("Columns & dtypes:")
        st.dataframe(df.dtypes.to_frame(name="dtype"), height=420)

    # small vertical spacer to ensure following content does not butt up against these cards
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    
    st.write("Raw data:")
    st.write(df.head(20))

    # Duplicates: handle potential unhashable cell values (lists/dicts/arrays)
    try:
        dup_count = int(df.duplicated().sum())
    except TypeError:
        df_safe = df.copy()
        for c in df_safe.columns:
            if df_safe[c].dtype == object:
                def _safe_serialize(x):
                    if isinstance(x, (list, set, dict, np.ndarray)):
                        try:
                            return json.dumps(x, default=str)
                        except Exception:
                            return str(x)
                    return x
                df_safe[c] = df_safe[c].apply(_safe_serialize)
        dup_count = int(df_safe.duplicated().sum())

    st.write("Duplicate rows:", dup_count)
    if dup_count:
        st.write("- Recommendation: consider dropping exact duplicates with df.drop_duplicates()")

def find_col(df: pd.DataFrame, *keywords):
    for c in df.columns:
        if all(k.lower() in c.lower() for k in keywords):
            return c
    return None

def parse_dates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    parsed_col = f"{col} Parsed"
    df[parsed_col] = pd.to_datetime(df[col], format="%m/%d/%Y  %I:%M:%S %p", errors="coerce")
    # keep date-like column as datetime64 (normalize to midnight) to ensure Streamlit/Arrow serialization
    df[f"{parsed_col} Date"] = df[parsed_col].dt.normalize()
    return df

# ==============================
# 311 Service Request Cleaning and Analysis
# ==============================
with tab_preparation_calls:

    st.header("📱 311 Service Requests Cleaning and Analysis")
    
    calls_clean = calls.copy()
    
    # Dates cleaning
    calls_clean = parse_dates(calls_clean, "Created Date")
    calls_clean = parse_dates(calls_clean, "Closed Date")
    calls_clean = parse_dates(calls_clean, "Resolution Action Updated Date")
    
    # Title-case complaint and descriptor columns (lowercase except first letter of each word)
    cols_to_title = [
        c for c in calls_clean.columns
        if any(k in c.lower() for k in ("complaint", "complaint type", "descriptor", "issue type", "incident address", "borough", "open data channel type"))
    ]
    for col in cols_to_title:
        if col in calls_clean.columns:
            # coerce to pandas string dtype, trim whitespace, normalize empty -> <NA>
            calls_clean[col] = calls_clean[col].astype("string").str.strip()
            calls_clean[col].replace("", pd.NA, inplace=True)
            # title-case safely on string dtype
            calls_clean[col] = calls_clean[col].str.title()
        
    # Create a summarized resolution column mapped from verbose resolution description
    resolution_col = "Resolution Description"

    def _summarize_resolution(txt):
        if pd.isna(txt):
            return np.nan
        s = " ".join(str(txt).lower().split())  # normalize whitespace and lowercase

        # Exact mappings from verbose Resolution Description -> Resolution Summary
        if pd.isna(txt):
            return np.nan
        s = " ".join(str(txt).lower().split())  # normalize whitespace & lowercase

        mapping = {
            "the department of housing preservation and development was not able to gain access to inspect the following conditions. the complaint has been closed. if the condition still exists, please file a new complaint.": "No access to inspect",
            "the department of housing preservation and development inspected the following conditions. no violations were issued. the complaint has been closed.": "Inspected; no violations found",
            "the complaint you filed is a duplicate of a condition already reported by another tenant for a building-wide condition. the original complaint is still open. hpd may attempt to contact you to verify the correction of the condition or may conduct an inspection of your unit if the original complainant is not available for verification.": "Duplicate complaint",
            "the department of housing preservation and development responded to a complaint of no heat or hot water and was advised by a tenant in the building that heat and hot water had been restored. if the condition still exists, please file a new complaint.": "Occupant confirmed issue resolved",
            "the department of housing preservation and development conducted or attempted to conduct an inspection. more information about inspection results can be found through hpd's website at www.nyc.gov/hpd by using hpdonline (enter your address on the home page) and entering your sr number under the complaint status option.": "Inspection attempted/completed",
            "the department of housing preservation and development inspected the following conditions. violations were issued. information about specific violations is available at www.nyc.gov/hpd.": "Inspected; violations issued",
            "the department of housing preservation and development was not able to gain access to your apartment or others in the building to inspect for a lack of heat or hot water. the complaint has been closed. if the condition still exists, please file a new complaint.": "No access to inspect",
            "the department of housing preservation and development was not able to gain access to your apartment to inspect for a lack of heat or hot water. however, hpd was able to verify that heat or hot water was inadequate by inspecting another apartment and a violation was issued. information about specific violations is available at www.nyc.gov/hpd.": "No access to inspect; issue confirmed",
            "the department of housing preservation and development contacted an occupant of the apartment and verified that the following conditions were corrected. the complaint has been closed. if the condition still exists, please file a new complaint.": "Occupant confirmed issue resolved",
            "the department of housing preservation and development has closed this complaint administratively. more information can be found through hpd's website at www.nyc.gov/hpd by using hpdonline (enter your address on the home page) and entering your sr number under the complaint status option.": "Administratively closed",
            "the following complaint conditions are still open. hpd may attempt to contact you to verify the correction of the condition or may conduct an inspection.": "Complaint still open",
            "the department of housing preservation and development was not able to gain access to inspect the conditions. if the conditions still exist and an inspection is required, please contact the borough office with your complaint number at 718 827 - 1955 (brooklyn).": "No access to inspect",
            "the department of housing preservation and development inspected the following conditions. violations were previously issued for these conditions. information about specific violations is available at www.nyc.gov/hpd.": "Inspected; existing violation",
            "the department of housing preservation and development was unable to access the rooms where the following conditions were reported. no violations were issued. the complaint has been closed.": "No access to inspect",
            "the department of housing preservation and development contacted a tenant in the building and verified that the following conditions were corrected. the complaint has been closed. if the condition still exists, please file a new complaint.": "Occupant confirmed issue resolved",
            "the department of housing preservation and development was unable to access the rooms where the following conditions were reported. no violations were issued. the complaint has been closed.": "No access to inspect",
            "the department of housing preservation and development conducted an inspection for the following conditions and identified potential lead-based paint conditions. hpd will attempt to contact you to schedule a follow-up inspection to test the paint for lead.": "Inspection found possible lead paint",
            "the department of housing preservation and development responded to a complaint of no heat or hot water. heat was not required at the time of the inspection. no violations were issued. if the condition still exists, please file a new complaint.": "Inspected; no violations found",
            "the department of housing preservation and development was at the building but did not have enough time to inspect the following conditions. if you have not already been contacted, please call the department of housing preservation and development at 212-863-8603 with your complaint number to schedule a follow-up inspection.": "Inspection incomplete due to time",
            "the department of housing preservation and development contacted an occupant of the apartment or building and verified that the complaint was addressed. the complaint has been closed.": "Occupant confirmed issue resolved",
        }

        if s in mapping:
            return mapping[s]

        # Fallback: return original trimmed string
        return str(txt).strip()

    calls_clean["Resolution Summary"] = calls_clean[resolution_col].apply(_summarize_resolution)

    # Drop rows missing critical columns
    unique_col = find_col(calls_clean, "unique", "key") or find_col(calls_clean, "complaint", "id")
    complaint_col = find_col(calls_clean, "complaint", "type") or find_col(calls_clean, "service", "request")
    created_col = find_col(calls_clean, "created", "date") or find_col(calls_clean, "created")
    critical_cols = [c for c in (unique_col, created_col, complaint_col) if c]
    calls_clean.dropna(subset=critical_cols, inplace=True)

    # Replace empty strings with NaN and drop fully empty columns
    calls_clean.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    calls_clean.drop(columns=calls_clean.columns[calls_clean.isna().all()], inplace=True)
    
    # drop requested noisy/irrelevant columns (case- and punctuation-insensitive)
    _targets = [
        "unnamed: 0", "agency", "agency name", "location type", "incident zip",
        "street name", "address type", "city", "bbl", "x coordinate (state plane)",
        "y coordinate (state plane)", "park facility name", "park borough", "location",
        "created year"
    ]
    _norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    _target_set = { _norm(t) for t in _targets }

    _cols_to_drop = [c for c in calls_clean.columns if _norm(c) in _target_set]

    calls_clean.drop(columns=_cols_to_drop, inplace=True)
    
    # ==============================
    # Combine Duplicate Calls
    # ==============================
    # Combine rows that share the same Created Date (date-only), Address, Complaint Type and Descriptor
    created_key = "Created Date Parsed Date"
    addr_key = "Incident Address"
    complaint_key = "Complaint Type"

    group_keys = [k for k in (created_key, addr_key, complaint_key) if k]

    def _combine_vals(s: pd.Series):
        s_non = s.dropna()
        if s_non.empty:
            return np.nan
        # preserve single-value dtype when possible
        uniq = s_non.astype(str).unique().tolist()
        if len(uniq) == 1:
            # return the original (first non-null) value to keep numeric/datetime types where possible
            return s_non.iloc[0]
        # multiple distinct values: list them in a single string
        return " | ".join(uniq)

    grp = calls_clean.groupby(group_keys, dropna=False)
    
    # apply combine function to every non-group column so the output has original non-key columns
    agg_map = {col: _combine_vals for col in calls_clean.columns if col not in group_keys}
    calls_agg = grp.agg(agg_map).reset_index()

    # number of original rows merged into each aggregated row
    counts = grp.size().reset_index(name="records_merged")
    
    # merge aggregated non-key columns with counts (group keys are present in calls_agg)
    calls_grouped = calls_agg.merge(counts, on=group_keys, how="left")

    # summary info
    duplicates_total = calls_clean.shape[0] - calls_grouped.shape[0]

    # expose variable for downstream use
    calls_grouped = calls_grouped  # final aggregated DataFrame
    
    # ==============================
    # Summary of duplicate-consolidation impact
    # ==============================
    orig = int(calls.shape[0])
    agg = int(calls_grouped.shape[0])
    removed = orig - agg
    pct = (removed / orig * 100) if orig else 0.0

    st.subheader("Duplicate consolidation — impact")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows before consolidation", orig)
    c2.metric("Rows after consolidation", agg)
    c3.metric("Rows removed", f"{removed} ({pct:.1f}%)")
    
    # Brief writeup of 311 service requests cleaning steps performed
    st.subheader("311 Service Requests Cleaning -- Methodology")
    st.markdown(
        """
        - Standardized date parsing and created normalized date columns (Created / Closed / Resolution) to ensure reliable temporal grouping, filtering, and joins with weather.
        - Trimmed whitespace, coerced empty strings to NaN, and set complaint/descriptor fields to a consistent string type and title-case to reduce category fragmentation.
        - Mapped verbose resolution text into a compact `Resolution Summary` (controlled vocabulary) to enable meaningful grouping and counts.
        - Dropped rows missing critical identifiers (unique key, created date, complaint type) so downstream aggregates operate on well-defined incidents.
        - Replaced fully-empty columns and irrelevant/noisy columns with removals to declutter the dataset and reduce accidental joins or analysis noise.
        - Consolidated near-duplicate records by grouping on date (date-only), address, complaint type and descriptor, merging non-key fields (preserve single values / concatenate distinct values) and recording merged counts.
        - Applied defensive handling for unhashable/complex cell values when deduplicating/combining to avoid crashes and ensure correctness.

        - Result: these steps convert raw, inconsistent 311 exports into a compact, incident-level dataset that's cleaner, easier to analyze, and better suited for visualization and predictive modeling.
        """
    )
    
    st.subheader("311 Service Requests | Raw")
    summarize_df(calls, "311 Service Requests | Raw")

    st.subheader("311 Service Requests | Cleaned")
    st.markdown("Cleaned version of 311 Service Requests, with standardized dates, normalized text fields, and improved resolution summaries.")
    summarize_df(calls_clean, "311 Service Requests | Cleaned")
                
    st.subheader("311 Service Requests | Cleaned and Consolidated")
    st.markdown("Cleaned and consolidated version of 311 Service Requests, consolidating duplicate/near-duplicate records.")
    
    # Context and brief note
    st.markdown(f"Grouped on {group_keys}. Resulting rows: {len(calls_grouped)} (collapsed {duplicates_total} duplicate rows).")
    st.caption(
        """
        Consolidating duplicate and near-duplicate 311 call records helps reduce noise from multiple updates/status changes logged for the same incident. 
        This yields a cleaner, incident-level dataset that better reflects unique service requests, improving the accuracy of counts, trends, and analyses downstream.
        """
    )
    st.markdown(f"Dropped columns: {', '.join(_cols_to_drop)}")

    summarize_df(calls_grouped, "311 Service Requests | Cleaned and Consolidated")
    
# ==============================
# Weather Cleaning and Analysis
# ==============================
with tab_preparation_weather:
    
    st.header("🌦️ Weather Data Cleaning and Analysis")
    
    weather_clean = weather.copy()
    
    # Treat identifier columsn as strings (preserve leading zeros, avoid numeric coercion)
    for col in ["Unnamed: 0", "USAF", "WBAN"]:
        if col in weather_clean.columns:
            weather_clean[col] = weather_clean[col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
            weather_clean[col] = weather_clean[col].astype("string")
    
    # Fix known typo in column name
    weather_clean.rename(columns={"Percipitation": "Precipitation"}, inplace=True)
    
    # Keep 'Date' as datetime64 (not python.date) so Streamlit can serialize DataFrames
    weather_clean["Date"] = pd.to_datetime(weather_clean[["Year", "Month", "Day"]], errors="coerce")
    
    # Normalize station names (trim, collapse whitespace)
    weather_clean["StationName"] = weather_clean["StationName"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    
    # Title-case StationName (capitalize the first letter of each word)
    weather_clean["StationName"] = weather_clean["StationName"].astype(str).str.title()
    
    # Map known station names to "NYC", otherwise NaN
    station_col = 'StationName'
    _nyc_names = {
        "THE BATTERY",
        "PORT AUTH DOWNTN MANHATTAN WALL ST HEL",
        "CENTRAL PARK",
        "BERGEN POINT",
        "LA GUARDIA AIRPORT",
        "JOHN F KENNEDY INTERNATIONAL AIRPORT",
    }
    weather_clean[station_col] = weather_clean[station_col].astype(str).str.strip()
    weather_clean["NYC"] = weather_clean[station_col].str.upper().map(lambda s: 'NYC' if s in _nyc_names else '-')

    # NYC stations summary: count rows tagged 'NYC' and list unique station names
    nyc_mask = weather_clean["NYC"].astype(str).str.upper() == "NYC"
    nyc_rows_count = int(nyc_mask.sum())
    nyc_station_names = sorted(weather_clean.loc[nyc_mask, "StationName"].dropna().astype(str).unique().tolist())
    nyc_station_count = len(nyc_station_names)

    st.subheader("NYC Stations — Counts & Names")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Unique NYC stations", nyc_station_count)
    with c2:
        st.write("NYC station names:")
        st.write(", ".join(nyc_station_names))
    
    # Standardize numeric columns: coerce to floats where present
    _expected_nums = [
        "MeanTemp", "MinTemp", "MaxTemp", "DewPoint",
        "Precipitation", "WindSpeed", "MaxSustainedWind", "Gust",
        "Rain", "SnowDepth", "SnowIce"
    ]
    num_cols = [c for c in _expected_nums if c in weather_clean.columns]
    for c in num_cols:
        weather_clean[c] = pd.to_numeric(weather_clean[c], errors="coerce")

    # Ensure Date is present and normalized to date (not datetime)
    if "Date" in weather_clean.columns:
        weather_clean = weather_clean[weather_clean["Date"].notna()].copy()
        weather_clean["Date"] = pd.to_datetime(weather_clean["Date"], errors="coerce").dt.date
        weather_clean = weather_clean[weather_clean["Date"].notna()].copy()

    # Drop exact duplicate records after normalization
    weather_clean.drop_duplicates(inplace=True)
    
    # Select and normalize desired columns (create missing cols as NaN so downstream code has a stable schema)
    _desired = [
        "StationName", "Latitude", "Longitude", "MeanTemp", "MinTemp", "MaxTemp", "DewPoint",
        "Precipitation", "WindSpeed", "MaxSustainedWind", "Gust", "Rain", "SnowDepth", "SnowIce",
        "Date", "NYC"
    ]

    for c in _desired:
        if c not in weather_clean.columns:
            weather_clean[c] = np.nan

    # Ensure numeric columns are numeric
    _num_cols = ["Latitude", "Longitude", "MeanTemp", "MinTemp", "MaxTemp", "DewPoint",
                 "Precipitation", "WindSpeed", "MaxSustainedWind", "Gust", "Rain", "SnowDepth", "SnowIce"]
    for c in _num_cols:
        if c in weather_clean.columns:
            weather_clean[c] = pd.to_numeric(weather_clean[c], errors="coerce")

    # Normalize Date to a date type (if possible)
    if "Date" in weather_clean.columns:
        weather_clean["Date"] = pd.to_datetime(weather_clean["Date"], errors="coerce").dt.date
    else:
        if set(["Year", "Month", "Day"]).issubset(weather_clean.columns):
            weather_clean["Date"] = pd.to_datetime(weather_clean[["Year", "Month", "Day"]], errors="coerce").dt.date
        else:
            weather_clean["Date"] = pd.NaT

    # Ensure NYC tag exists (derive from StationName if necessary)
    if "NYC" not in weather_clean.columns or weather_clean["NYC"].isna().all():
        if "StationName" in weather_clean.columns:
            weather_clean["NYC"] = weather_clean["StationName"].astype(str).str.upper().map(lambda s: "NYC" if s in _nyc_names else "-")
        else:
            weather_clean["NYC"] = "-"

    # Final column ordering (keep only desired order)
    weather_clean = weather_clean[_desired].copy()
    
    # Filter to NYC stations only
    weather_clean_nyc = weather_clean[weather_clean["NYC"].astype(str).str.upper() == "NYC"].copy()
    weather_clean_nyc.reset_index(drop=True, inplace=True)

    # ==============================
    # Group by Date to produce daily summaries
    # ==============================
    # Perform groupby with trimmed-mean (drop single highest and lowest per date then average)
    def _trimmed_mean(s):
        name = getattr(s, "name", None)
        arr = s.dropna().astype(float).values
        if arr.size == 0:
            return np.nan
        # Exceptions: for Rain and SnowIce return the maximum (do not trim)
        if name in ("Rain", "SnowIce"):
            return float(arr.max())
        if arr.size <= 2:
            return float(arr.mean())
        arr.sort()
        return float(arr[1:-1].mean())

    # Numeric columns to apply trimmed mean to (exclude Date/StationName)
    numeric_cols = [
        c for c in weather_clean_nyc.columns
        if c not in ("Date", "StationName", "NYC") and pd.api.types.is_numeric_dtype(weather_clean_nyc[c])
    ]

    agg_dict = {c: _trimmed_mean for c in numeric_cols}
    agg_dict["StationName"] = lambda s: sorted(s.dropna().astype(str).unique().tolist())

    weather_nyc_grouped = (
        weather_clean_nyc
        .groupby("Date", dropna=True)
        .agg(agg_dict)
        .reset_index()
    )
    
    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows | Raw", int(len(weather)))
    c2.metric("Rows | Cleaned, NYC-only", int(len(weather_clean_nyc)))
    c3.metric("Rows | Cleaned, NYC-only, Grouped by Day", int(len(weather_nyc_grouped)))
    
    # Brief writeup of weather cleaning steps performed
    st.subheader("Weather Cleaning -- Methodology")
    st.markdown(
    """
    - Renamed known misspellings (e.g., "Percipitation" → "Precipitation").
    - Constructed a proper Date column from Year/Month/Day and coerced to datetime; normalized to date and removed invalid dates.
    - Normalized StationName (trimmed, collapsed whitespace, title-cased).
    - Tagged and filtered for only known NYC stations as `NYC` (others marked `-`) to enable NYC-only filtering/aggregation.
    - Coerced expected numeric measurement columns to numeric (floats) with safe coercion (errors → NaN).
    - Ensured `Date` is present and normalized to a date type; filtered out rows without a valid date.
    - Dropped exact duplicate records after normalization to avoid double-counting.
    - Result: a cleaned, numeric-ready weather table suitable for station aggregation, trimmed-mean daily summaries, and joins with the 311 dataset.
    
    - Grouped by Date to produce daily summaries: applied trimmed-mean (drop highest and lowest per day) to reduce outlier impact, with exceptions for Rain and SnowIce (max value).
    - Final output: daily NYC weather DataFrame with consistent numeric columns, ready for analysis and eventual merging with 311 service requests.
    """
    )
    
    st.subheader("Weather | Raw")
    st.markdown("Raw Weather data, with no data manipulation")
    summarize_df(weather, "Weather (Raw)")
    
    st.subheader("Weather | Cleaned, NYC-Only")
    st.markdown("Cleaned version of Weather, and filtering for NYC stations only.")
    summarize_df(weather_clean_nyc, "Weather (Cleaned, NYC-Only)")
    
    st.subheader("Weather | Cleaned, NYC-Only, Grouped by Day")
    st.markdown("Trimmed-mean across all 6 NYC stations, dropping highest and lowest per day to normalize.")
    
    summarize_df(weather_nyc_grouped, "Weather (Cleaned, NYC-Only)")
    
# ==============================
# ANALYSIS | 311 SERVICE REQUESTS
# ==============================

with tab_descriptive_calls:
    st.header("📱 Analysis | 311 Service Requests")

    calls = calls_clean
    calls_grouped = calls_grouped

    # =============================
    # Metrics for key call statistics
    # =============================
    total_calls = len(calls)
    total_distinct_calls = len(calls_grouped)
    peak_month = calls["Created Date Parsed"].dropna().dt.month.value_counts().idxmax()
    peak_hour = calls["Created Date Parsed"].dropna().dt.hour.value_counts().idxmax()
    peak_weekday = calls["Created Date Parsed"].dropna().dt.day_name().value_counts().idxmax()
    
    # Busiest day
    daily_counts_series = calls.groupby("Created Date Parsed Date").size()
    busiest_day_ts = daily_counts_series.idxmax()
    busiest_day = pd.to_datetime(busiest_day_ts).date()
    busiest_day_count = int(daily_counts_series.max())
    
    # Unique addresses reporting cases
    addr_col = "Incident Address"
    if addr_col and addr_col in calls.columns:
        uniq_addresses = (
            calls[addr_col]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "": np.nan})
            .dropna()
            .nunique()
        )
    else:
        uniq_addresses = 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📞 Total Service Requests", f"{int(total_calls):,}")
    col2.metric("🧾 Total Distinct Service Requests", f"{int(total_distinct_calls):,}")
    col3.metric("🔁 Duplicated Service Requests", f"{int(total_calls) - int(total_distinct_calls):,}")
    col4.metric("📍 Total Unique Addresses", f"{int(uniq_addresses):,}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Peak Month", calendar.month_name[peak_month])
    col2.metric("⏰ Peak Hour", peak_hour)
    col3.metric("🗓️ Peak Weekday", peak_weekday)
    col4.metric("🔥 Busiest Day", f"{busiest_day} ({busiest_day_count})")
    
    # Most common borough
    borough_col = 'Borough'
    vc = calls[borough_col].dropna().astype(str).str.title().value_counts()
    most_common_borough = vc.index[0]
    most_common_borough_count = int(vc.iloc[0])
    
    # Most common Complaint Type
    complaint_col = find_col(calls, "complaint", "type") or find_col(calls, "service", "request")
    vc_complaint = calls[complaint_col].dropna().astype(str).str.title().value_counts()
    most_common_complaint = vc_complaint.index[0]
    most_common_complaint_count = int(vc_complaint.iloc[0])
    
    # Most common resolution summary (defensive)
    if "Resolution Summary" in calls.columns:
        res_vc = calls["Resolution Summary"].dropna().astype(str).str.title().value_counts()
        if not res_vc.empty:
            most_res = res_vc.index[0]
            most_res_count = int(res_vc.iloc[0])
        else:
            most_res, most_res_count = "N/A", 0
    else:
        most_res, most_res_count = "N/A", 0
        
    # Count distinct aggregated cases with Resolution Summary == "Inspected; violations issued"
    _target = "inspected; violations issued"
    cnt = 0
    if "calls" in globals() and "Resolution Summary" in calls.columns:
        mask = calls["Resolution Summary"].fillna("").astype(str).str.lower() == _target
        if "Unique Key" in calls.columns:
            cnt = int(calls.loc[mask, "Unique Key"].nunique())
        else:
            cnt = int(mask.sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏙️ Most Common Borough", f"{most_common_borough} ({most_common_borough_count})")
    col2.metric("⚠️ Most Common Complaint Type", f"{most_common_complaint} ({most_common_complaint_count})")
    col3.metric("✅ Most Common Resolution Summary", f"{most_res} ({most_res_count})")
    col4.metric("🔨 Cases with Violations Issued", cnt)
    
    # ==============================
    # Map Visualization
    # ==============================
    st.header("📍 311 Service Requests Map Visualization")

    lat_col = "Latitude"
    lon_col = "Longitude"

    # Identify complaint column
    candidate_cols = [
        "Complaint Type",
        "Complaint",
        "Descriptor",
        "Complaint Type Description"
    ]
    complaint_col = next((c for c in candidate_cols if c in calls.columns), None)

    # Identify date column
    date_candidates = ["Created Date", "CreatedDate", "created_date"]
    date_col = next((c for c in date_candidates if c in calls.columns), None)

    # Clean & prepare data
    df_map = calls.copy()

    df_map[lat_col] = pd.to_numeric(df_map[lat_col], errors="coerce")
    df_map[lon_col] = pd.to_numeric(df_map[lon_col], errors="coerce")
    df_map.dropna(subset=[lat_col, lon_col], inplace=True)

    df_map[date_col] = pd.to_datetime(df_map[date_col], errors="coerce")
    df_map["month"] = df_map[date_col].dt.to_period("M").astype(str)

    center = {
        "lat": float(df_map[lat_col].mean()),
        "lon": float(df_map[lon_col].mean())
    }

    # Default zoom (increased for a closer view)
    zoom_default = 12.5

    # Map controls (displayed above the map)
    c1, c2 = st.columns([1, 1])
    with c1:
        map_type = st.radio(
            "Map Type",
            ["Scatter", "Heatmap"],
            horizontal=True
        )
    with c2:
        animate = st.checkbox(
            "Animate by Month",
            value=False,
            disabled=not bool(date_col)
        )

    # Build map
    if map_type == "Scatter":
        fig_map = px.scatter_mapbox(
            df_map,
            lat=lat_col,
            lon=lon_col,
            color=complaint_col if complaint_col else None,
            hover_name=complaint_col,
            animation_frame="month" if animate and date_col else None,
            zoom=zoom_default,
            center=center,
            opacity=0.55,
            height=650,
            mapbox_style="carto-positron",
        )

        fig_map.update_traces(
            marker=dict(size=7)
        )

    else:
        fig_map = px.density_mapbox(
            df_map,
            lat=lat_col,
            lon=lon_col,
            radius=14,
            animation_frame="month" if animate and date_col else None,
            zoom=zoom_default,
            center=center,
            height=650,
            mapbox_style="carto-positron",
        )

    # Final polish
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            title="Complaint Type"
        )
    )
    st.write("Service requests all from the Brooklyn Borough")
    st.plotly_chart(fig_map, use_container_width=True)

    # ==============================
    # Temporal Distributions (Daily, Monthly, Hourly, Weekday)
    # ==============================
    st.header("⏱️ 311 Service Requests Temporal Distributions")
    
    # Helper to pick a created-date-like column and return a datetime Series
    # Prefer the full datetime column ("Created Date Parsed") before the date-only normalized column
    # so that hourly information is preserved for hourly aggregations.
    def _created_dt_series(df: pd.DataFrame) -> pd.Series:
        for cand in ("Created Date Parsed", "Created Date Parsed Date", "Created Date", "CreatedDate"):
            if cand in df.columns:
                return pd.to_datetime(df[cand], errors="coerce")
        return pd.Series(pd.NaT, index=df.index)

    # Prepare created datetimes for both sources
    calls_dt = _created_dt_series(calls)
    grouped_dt = _created_dt_series(calls_grouped)

    # Utility to build a counts DF with a source label
    def _counts_df(dt_series: pd.Series, freq: str, source_label: str) -> pd.DataFrame:
        # coerce to datetime and drop NA in a robust way
        if dt_series is None:
            return pd.DataFrame(columns=["x", "count", "source"])
        s = pd.to_datetime(dt_series, errors="coerce").dropna()
        if s.empty:
            return pd.DataFrame(columns=["x", "count", "source"])
        if freq == "D":
            # normalize to midnight so x is a proper Timestamp
            df = s.dt.normalize().value_counts().sort_index().rename_axis("x").reset_index(name="count")
            df["x"] = pd.to_datetime(df["x"])
        elif freq == "M":
            # convert to month-start timestamps so plotting treats x as time
            df = s.dt.to_period("M").dt.to_timestamp().value_counts().sort_index().rename_axis("x").reset_index(name="count")
            df["x"] = pd.to_datetime(df["x"])
        elif freq == "H":
            df = s.dt.hour.value_counts().sort_index().rename_axis("x").reset_index(name="count")
        elif freq == "WEEKDAY":
            df = s.dt.day_name().value_counts().reindex(list(calendar.day_name), fill_value=0).rename_axis("x").reset_index(name="count")
        else:
            return pd.DataFrame(columns=["x", "count", "source"])
        df["source"] = source_label
        return df

    # Build count frames for both raw calls and consolidated calls_grouped
    daily_calls = _counts_df(calls_dt, "D", "calls")
    daily_grouped = _counts_df(grouped_dt, "D", "calls_grouped")
    daily_all = pd.concat([daily_calls, daily_grouped], ignore_index=True)

    monthly_calls = _counts_df(calls_dt, "M", "calls")
    monthly_grouped = _counts_df(grouped_dt, "M", "calls_grouped")
    monthly_all = pd.concat([monthly_calls, monthly_grouped], ignore_index=True)

    hourly_calls = _counts_df(calls_dt, "H", "calls")

    # For calls_grouped: count distinct aggregated incidents per hour by looking at the original calls'
    # Created Date Parsed hours and counting unique group-key occurrences per hour.
    group_keys = [k for k in ("Created Date Parsed", "Incident Address", "Complaint Type") if k in calls.columns]
    gh = calls.copy()
    gh["hour"] = pd.to_datetime(gh.get("Created Date Parsed"), errors="coerce").dt.hour

    if group_keys:
        tmp = gh.dropna(subset=group_keys + ["hour"])
        if not tmp.empty:
            unique_group_hours = tmp.groupby(group_keys)["hour"].unique().reset_index()
            exploded = unique_group_hours.explode("hour")
            hourly_grouped = (
                exploded["hour"]
                .value_counts()
                .sort_index()
                .rename_axis("x")
                .reset_index(name="count")
            )
            hourly_grouped["source"] = "calls_grouped"
        else:
            hourly_grouped = pd.DataFrame(columns=["x", "count", "source"])
    else:
        hourly_grouped = pd.DataFrame(columns=["x", "count", "source"])

    # Combine for plotting
    hourly_all = pd.concat([hourly_calls, hourly_grouped], ignore_index=True)

    weekday_calls = _counts_df(calls_dt, "WEEKDAY", "calls")
    weekday_grouped = _counts_df(grouped_dt, "WEEKDAY", "calls_grouped")
    weekday_all = pd.concat([weekday_calls, weekday_grouped], ignore_index=True)

    # Plot: show both sources on the same chart for easy comparison
    fig_daily = px.line(
        daily_all.sort_values("x"),
        x="x",
        y="count",
        color="source",
        labels={"x": "Date", "count": "Count", "source": "Source"}
    )

    fig_monthly = px.bar(
        monthly_all.sort_values("x"),
        x="x",
        y="count",
        color="source",
        barmode="group",
        labels={"x": "Month", "count": "Count", "source": "Source"}
    )

    fig_hourly = px.bar(
        hourly_all.sort_values("x"),
        x="x",
        y="count",
        color="source",
        barmode="group",
        labels={"x": "Hour", "count": "Count", "source": "Source"}
    )

    fig_weekday = px.bar(
        weekday_all,
        x="x",
        y="count",
        color="source",
        barmode="group",
        labels={"x": "Weekday", "count": "Count", "source": "Source"}
    )

    # Display in 2x2 grid
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.subheader("📈 Daily Call Volume")
        st.plotly_chart(fig_daily, use_container_width=True)
    with r1c2:
        st.subheader("📅 Monthly Call Volume")
        st.plotly_chart(fig_monthly, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.subheader("⏰ Hourly Call Distribution")
        st.plotly_chart(fig_hourly, use_container_width=True)
    with r2c2:
        st.subheader("🗓️ Weekday Call Distribution")
        st.plotly_chart(fig_weekday, use_container_width=True)
    
    # ==============================
    # Complaint Type Distribution (Bar + Sunburst side-by-side)
    # ==============================
    complaint_candidates = [
        "Complaint Type", "Complaint", "Complaint Type Description", "Complaint Type/Descriptor", "ComplaintType"
    ]
    descriptor_candidates = ["Descriptor", "descriptor", "Descriptor Description", "Issue Type"]

    complaint_col = next((c for c in complaint_candidates if c in calls.columns), None)
    descriptor_col = next((c for c in descriptor_candidates if c in calls.columns and c != complaint_col), None)

    st.header("Complaint Type Distribution and Drilldown")
    df_plot = calls.copy()
    df_plot[complaint_col] = df_plot[complaint_col].fillna("Unknown").astype(str)

    # Aggregate complaint counts for calls and calls_grouped (for bar chart)
    if complaint_col is None:
        st.write("Complaint column not found.")
        agg_bar = pd.DataFrame(columns=[complaint_col or "complaint", "count", "source"])
        fig_bar_grouped = None
    else:
        s_calls = calls[complaint_col].fillna("Unknown").astype(str).str.title()
        agg_calls = s_calls.value_counts().reset_index()
        agg_calls.columns = [complaint_col, "count"]
        agg_calls["source"] = "calls"

        if complaint_col in calls_grouped.columns:
            s_grouped = calls_grouped[complaint_col].fillna("Unknown").astype(str).str.title()
            agg_grouped = s_grouped.value_counts().reset_index()
            agg_grouped.columns = [complaint_col, "count"]
            agg_grouped["source"] = "calls_grouped"
        else:
            agg_grouped = pd.DataFrame(columns=[complaint_col, "count", "source"])

        # long-form dataframe with a 'source' column so we can plot grouped bars
        agg_bar = pd.concat([agg_calls, agg_grouped], ignore_index=True)

        # ensure numeric counts and set descending category order by total counts
        agg_bar["count"] = pd.to_numeric(agg_bar["count"], errors="coerce").fillna(0).astype(int)
        order = agg_bar.groupby(complaint_col)["count"].sum().sort_values(ascending=False).index.tolist()
        agg_bar[complaint_col] = pd.Categorical(agg_bar[complaint_col], categories=order, ordered=True)

        # Create a grouped bar figure explicitly (calls vs calls_grouped side-by-side)
        try:
            fig_bar_grouped = px.bar(
                agg_bar,
                x=complaint_col,
                y="count",
                color="source",
                title="Complaint Type Distribution — calls vs calls_grouped",
                labels={complaint_col: "Complaint Type", "count": "Count", "source": "Source"},
            )
            fig_bar_grouped.update_layout(barmode="group")
        except Exception:
            fig_bar_grouped = None

        # Monkeypatch px.bar so subsequent px.bar(agg_bar, ...) calls without an explicit color
        # will default to coloring by 'source' and use grouped bars (mimics weekday chart behavior).
        # This is scoped and defensive: only activates when plotting a DataFrame that contains 'source'
        # and no explicit 'color' in kwargs.
        try:
            _orig_px_bar = px.bar

            def _patched_px_bar(df, *args, **kwargs):
                try:
                    if isinstance(df, pd.DataFrame) and "source" in df.columns and "color" not in kwargs:
                        kwargs = dict(kwargs)  # copy to avoid mutating caller dict
                        kwargs["color"] = "source"
                    fig = _orig_px_bar(df, *args, **kwargs)
                    # ensure grouped bars if we injected color
                    if isinstance(df, pd.DataFrame) and "source" in df.columns and kwargs.get("color") == "source":
                        try:
                            fig.update_layout(barmode="group")
                        except Exception:
                            pass
                    return fig
                except Exception:
                    # fallback to original behavior on any error
                    return _orig_px_bar(df, *args, **kwargs)

            px.bar = _patched_px_bar  # override for subsequent calls
        except Exception:
            # if monkeypatching fails, we still have fig_bar_grouped prepared above
            pass

    # Sunburst aggregation (use descriptor if present; otherwise show single-level sunburst)
    if descriptor_col:
        df_plot[descriptor_col] = df_plot[descriptor_col].fillna("Unknown").astype(str)
        agg_sun = df_plot.groupby([complaint_col, descriptor_col]).size().reset_index(name="count")
        sun_path = [complaint_col, descriptor_col]
    else:
        agg_sun = agg_bar.copy()
        sun_path = [complaint_col]

    col_bar, col_sun = st.columns(2)
    with col_bar:
        fig_bar = px.bar(agg_bar, x=complaint_col, y="count")
        st.subheader("Complaint Type Breakdown")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_sun:
        # Build sunburst manually to avoid narwhals/plotly-express hierarchy internals
        def _flatten_to_str(x):
            if isinstance(x, (list, tuple)):
                return " | ".join(str(i) for i in x)
            return str(x)

        agg_sun = agg_sun.copy()
        for c in agg_sun.columns:
            agg_sun[c] = agg_sun[c].apply(_flatten_to_str)
        agg_sun["count"] = pd.to_numeric(agg_sun["count"], errors="coerce").fillna(0).astype(int)

        ids = []
        labels = []
        parents = []
        values = []

        if len(sun_path) == 1:
            # single-level: each label is a root node
            for _, row in agg_sun.iterrows():
                lbl = row[sun_path[0]]
                ids.append(lbl)
                labels.append(lbl)
                parents.append("")
                values.append(int(row["count"]))
        else:
            # two-level: top-level (complaint) and child (descriptor)
            top_col, child_col = sun_path[0], sun_path[1]
            top_vals = agg_sun[top_col].unique().tolist()
            for t in top_vals:
                ids.append(t)
                labels.append(t)
                parents.append("")
                values.append(int(agg_sun.loc[agg_sun[top_col] == t, "count"].sum()))
            for _, row in agg_sun.iterrows():
                t = row[top_col]
                c = row[child_col]
                cid = f"{t}||{c}"
                ids.append(cid)
                labels.append(c)
                parents.append(t)
                values.append(int(row["count"]))

        fig_sun = go.Figure(go.Sunburst(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total"))
        st.subheader("Complaint Type -> Descriptor | Sunburst")
        st.plotly_chart(fig_sun, use_container_width=True)
    
    # 100% stacked bar: complaint type share by month
    _month_col = "month_for_stack"
    df_stack = df_plot.copy()
    df_stack[_month_col] = pd.to_datetime(df_stack["Created Date Parsed Date"], errors="coerce").dt.to_period("M").astype(str)
    df_stack = df_stack.dropna(subset=[_month_col, complaint_col])

    agg_stack = df_stack.groupby([_month_col, complaint_col]).size().reset_index(name="count")

    # ensure chronological month order
    month_order = sorted(agg_stack[_month_col].unique(), key=lambda x: pd.to_datetime(x))
    agg_stack[_month_col] = pd.Categorical(agg_stack[_month_col], categories=month_order, ordered=True)

    # Compute fractional share per month for 100% stacked visualization
    month_totals = agg_stack.groupby(_month_col)["count"].sum().reset_index().rename(columns={"count": "month_total"})
    agg_stack = agg_stack.merge(month_totals, on=_month_col)
    agg_stack["frac"] = agg_stack["count"] / agg_stack["month_total"]

    fig_stack = px.bar(
        agg_stack,
        x=_month_col,
        y="frac",
        color=complaint_col,
        labels={_month_col: "Month", "frac": "Share", complaint_col: "Complaint Type"},
    )
    fig_stack.update_layout(
        barmode="stack",
        xaxis=dict(categoryorder="array", categoryarray=month_order),
        yaxis=dict(tickformat=".0%"),
        height=520,
    )
    st.subheader("Complaint Type Share by Month (100% Stacked Bar)")
    st.plotly_chart(fig_stack, use_container_width=True)

    # ==============================
    # Resolution Time and Description Analysis
    # ==============================
    st.header("⏱️ Resolution Time Analysis")
    # Resolution time + resolution description analysis
    df_res = calls.copy()
    # Coerce parsed date columns to datetimelike to avoid arithmetic errors (aggregations may have produced arrays/objects)
    for _dt in ("Closed Date Parsed", "Created Date Parsed"):
        if _dt in df_res.columns:
            df_res[_dt] = pd.to_datetime(df_res[_dt], errors="coerce")
        else:
            df_res[_dt] = pd.NaT

    # Now compute resolution time in hours; guard against unexpected types
    try:
        df_res["resolution_time_hours"] = (df_res["Closed Date Parsed"] - df_res["Created Date Parsed"]).dt.total_seconds() / 3600.0
    except Exception:
        # Fallback: compute using numpy where possible, coercing to datetime64[ns]
        closed = pd.to_datetime(df_res["Closed Date Parsed"], errors="coerce")
        created = pd.to_datetime(df_res["Created Date Parsed"], errors="coerce")
        df_res["resolution_time_hours"] = (closed - created).dt.total_seconds() / 3600.0
    df_res = df_res[df_res["resolution_time_hours"].notna() & (df_res["resolution_time_hours"] >= 0)]

    # Summary metrics
    avg_resolution_time = df_res["resolution_time_hours"].mean()
    med_resolution_time = df_res["resolution_time_hours"].median()
    pct_24h = (df_res["resolution_time_hours"] <= 24).mean() * 100
    pct_72h = (df_res["resolution_time_hours"] <= 72).mean() * 100
    # Defensive: `Resolution Summary` may not exist after aggregation; handle missing column
    if "Resolution Summary" in df_res.columns:
        pct_inspected_violations = df_res["Resolution Summary"].fillna("").eq("Inspected; violations issued").mean() * 100
    else:
        pct_inspected_violations = 0.0
    
    col_b, col_c, col_d, col_e = st.columns(4)
    col_b.metric("Median Resolution (hrs)", f"{med_resolution_time:.2f}")
    col_c.metric("% Resolved ≤ 24h", f"{pct_24h:.1f}%")
    col_d.metric("% Resolved ≤ 72h", f"{pct_72h:.1f}%")
    col_e.metric("% Violations Issued", f"{pct_inspected_violations:.1f}%")

    # Find resolution description/action column
    resolution_candidates = ["Resolution Summary"]
    resolution_col = next((c for c in resolution_candidates if c in df_res.columns), None)
    # Ensure a resolution column exists for downstream charts; create fallback if missing
    if resolution_col is None:
        resolution_col = "Resolution Summary"
        df_res[resolution_col] = "Unknown"
    else:
        df_res[resolution_col] = df_res[resolution_col].fillna("Unknown").astype(str)

    # Top resolution types bar chart
    top_n = 20
    top_res = df_res[resolution_col].value_counts().head(top_n).reset_index()
    top_res.columns = [resolution_col, "count"]
    fig_top_res = px.bar(top_res, x=resolution_col, y="count")

    # Complaint -> Resolution sunburst
    complaint_candidates = ["Complaint Type"]
    complaint_col_local = next((c for c in complaint_candidates if c in df_res.columns), None)
    df_sun = df_res.copy()
    if complaint_col_local:
        df_sun[complaint_col_local] = df_sun[complaint_col_local].fillna("Unknown").astype(str)
        agg_sun = df_sun.groupby([complaint_col_local, resolution_col]).size().reset_index(name="count")
        # Ensure grouping columns are plain strings (flatten lists/iterables) to avoid narwhals/plotly internals failing
        def _flatten_to_str(x):
            if isinstance(x, (list, tuple)):
                return " | ".join(str(i) for i in x)
            return str(x)

        agg_sun[complaint_col_local] = agg_sun[complaint_col_local].apply(_flatten_to_str)
        agg_sun[resolution_col] = agg_sun[resolution_col].apply(_flatten_to_str)
        agg_sun["count"] = pd.to_numeric(agg_sun["count"], errors="coerce").fillna(0).astype(int)

        # Build hierarchical sunburst inputs manually to avoid narwhals shape/expr issues
        def _make_id(prefix, s):
            return f"{prefix}||{str(s)}"

        labels = []
        parents = []
        values = []
        ids = []

        # Top-level complaint nodes
        complaints = agg_sun[complaint_col_local].unique().tolist()
        for comp in complaints:
            cid = _make_id("C", comp)
            comp_sum = int(agg_sun.loc[agg_sun[complaint_col_local] == comp, "count"].sum())
            ids.append(cid)
            labels.append(str(comp))
            parents.append("")
            values.append(comp_sum)

        # Child nodes: resolution per complaint
        for _, row in agg_sun.iterrows():
            comp = row[complaint_col_local]
            res = row[resolution_col]
            cnt = int(row["count"])
            pid = _make_id("C", comp)
            rid = _make_id("R", f"{comp}||{res}")
            ids.append(rid)
            labels.append(str(res))
            parents.append(pid)
            values.append(cnt)

        fig_sun = go.Figure(go.Sunburst(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total"))
    else:
        agg_sun = df_sun[resolution_col].fillna("Unknown").value_counts().reset_index()
        agg_sun.columns = [resolution_col, "count"]
        fig_sun = px.sunburst(agg_sun, path=[resolution_col], values="count")
        
    # Visual: complaint types for cases with "Inspected; violations issued"
    target_mask = (
        df_res[resolution_col]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains("inspected; violations issued")
    )

    df_violations = df_res.loc[target_mask].copy()

    if df_violations.empty:
        st.write("No records found for resolution = 'Inspected; violations issued'.")
    else:
        # pick complaint column (fall back to any likely complaint column)
        comp_col = complaint_col_local or next((c for c in df_violations.columns if "complaint" in c.lower()), None)
        if comp_col is None:
            st.write("Complaint column not found.")
        else:
            cnts = (
                df_violations[comp_col]
                .fillna("Unknown")
                .astype(str)
                .str.title()
                .value_counts()
                .reset_index()
            )
            cnts.columns = [comp_col, "count"]
            fig_viol = px.bar(
                cnts.head(20),
                x=comp_col,
                y="count",
                labels={comp_col: "Complaint Type", "count": "Number of cases"},
            )

    # Display side-by-side
    col_bar, col_sun, col_viol = st.columns(3)
    with col_bar:
        st.subheader("Resolution Summary")
        st.plotly_chart(fig_top_res, use_container_width=True)
    with col_sun:
        st.subheader("Complaint Type → Resolution Summary")
        st.plotly_chart(fig_sun, use_container_width=True)
    with col_viol:
        st.subheader("Complaint Types of Violations Issued")
        st.plotly_chart(fig_viol, use_container_width=True)
    
    # ==============================
    # Open Data Channel Analysis: line-over-time + distribution pie
    # ==============================
    _ch_candidates = {c.lower(): c for c in calls.columns}
    _date_candidates = ["created date parsed date", "created date parsed", "created date", "createddate"]
    _channel_key = None
    _date_key = None
    for k in _ch_candidates:
        if "open data channel" in k:
            _channel_key = _ch_candidates[k]
            break
    for d in _date_candidates:
        if d in _ch_candidates:
            _date_key = _ch_candidates[d]
            break

    st.header("📡 Open Data Channel Type — Time Series & Distribution")

    df_ch = calls[[_channel_key, _date_key] if _date_key and _date_key in calls.columns else [_channel_key]].copy()
    df_ch[_channel_key] = df_ch[_channel_key].fillna("Unknown").astype(str).str.title()

    # Time series: daily counts per channel (limit to top channels for readability)
    if _date_key and _date_key in df_ch.columns:
        df_ch[_date_key] = pd.to_datetime(df_ch[_date_key], errors="coerce")
        df_ch["date_only"] = df_ch[_date_key].dt.date
        total_by_channel = df_ch[_channel_key].value_counts()
        top_channels = total_by_channel.head(8).index.tolist()

        ts = (
            df_ch[df_ch[_channel_key].isin(top_channels)]
            .groupby(["date_only", _channel_key])
            .size()
            .reset_index(name="count")
        )

        # Pivot to wide for line plotting; fill missing with 0
        ts_wide = ts.pivot(index="date_only", columns=_channel_key, values="count").fillna(0).reset_index()
        fig_ts = px.line(
            ts_wide,
            x="date_only",
            y=[c for c in ts_wide.columns if c != "date_only"],
            labels={"date_only": "Date", "value": "Daily calls"}
        )
    else:
        fig_ts = None

    # Distribution pie (all channels)
    dist = df_ch[_channel_key].value_counts().reset_index()
    dist.columns = [_channel_key, "count"]
    fig_pie = px.pie(
        dist,
        names=_channel_key,
        values="count",
        hole=0.35
    )

    # Layout: left=timeseries (if available), right=pie
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Daily 311 Service Requests Over Time by Open Data Channel")
        st.plotly_chart(fig_ts, use_container_width=True, height=520)
    with c2:
        st.subheader("Overall Open Data Channel Distribution")
        st.plotly_chart(fig_pie, use_container_width=True, height=520)
    
    # ==============================
    # Business / operational implications commentary (render in dashboard)
    # ==============================
    st.header("🔎 Business & Operational Implications — 311 Service Request Trends")

    # Defensive helpers
    _top_complaints = calls[complaint_col].dropna().astype(str).value_counts() if complaint_col in calls else pd.Series(dtype=int)
    _top_boroughs = calls["Borough"].dropna().astype(str).value_counts() if "Borough" in calls else pd.Series(dtype=int)

    pct_top5 = (_top_complaints.head(5).sum() / len(calls) * 100) if len(calls) else 0
    top5_complaints_str = ", ".join(f"{c} ({n})" for c, n in _top_complaints.head(5).items()) or "N/A"
    top_boroughs_str = ", ".join(f"{b} ({n})" for b, n in _top_boroughs.head(5).items()) or "N/A"

    # Pull previously computed resolution metrics safely
    _med_res = locals().get("med_resolution_time", None)
    _pct_24h = locals().get("pct_24h", None)

    bullets = [
        f"Staffing & scheduling: peak hour around {peak_hour if 'peak_hour' in locals() else 'N/A'} and peak weekday {peak_weekday if 'peak_weekday' in locals() else 'N/A'} suggest shifting on-duty coverage toward those times to reduce response lag.",
        f"Surge & capacity planning: busiest day {busiest_day} had {busiest_day_count} calls — plan flexible staffing/overflow processes for similar spikes (events, storms, holidays).",
        f"Complaint concentration: top 5 complaint types account for ~{pct_top5:.1f}% of calls. Focused triage, dedicated crews, and targeted preventative programs for: {top5_complaints_str}.",
        f"Geographic targeting: highest volume boroughs — {top_boroughs_str} — indicate where localized outreach, inspection teams, or depot placement could reduce travel time and improve SLA.",
        f"Resolution SLAs & process: median resolution ~{_med_res:.1f} hrs." if _med_res is not None else "Resolution SLAs: review median/percentile resolution times to set realistic targets.",
    ]

    if _pct_24h is not None:
        bullets.append(f"{_pct_24h:.1f}% of cases resolved within 24 hours — consider triage rules to push more high-impact/quick-fix work into that bucket.")

    # Domain-specific operational notes (auto-detect common themes)
    complaint_text = " ".join(_top_complaints.index[:20].str.lower()) if not _top_complaints.empty else ""
    if "heat" in complaint_text or "hot water" in complaint_text:
        bullets.append("Heating/hot-water complaints are common — coordinate with housing maintenance and seasonal readiness (pre-winter inspections, expedited heat repairs).")
    if "rodent" in complaint_text or "pest" in complaint_text:
        bullets.append("Rodent/pest issues point to sanitation and building-level interventions — amplify inspection and abatement programs.")
    if "water" in complaint_text or "flood" in complaint_text or "sewer" in complaint_text:
        bullets.append("Water-related complaints suggest infrastructure/stormwater vulnerability — prioritize preventative drainage and rapid-response plumbing crews during heavy rain periods.")

    # Data & operational improvements
    bullets += [
        "Improve location & time accuracy (geocoding and consistent datetime parsing) to optimize routing and reduce ETA uncertainty.",
        "Integrate weather/station feeds and event calendars into dispatch rules to pre-position crews during forecasted severe weather.",
        "Consider KPI/dashboarding per-complaint SLAs, time-to-first-response, repeat-call rates, and cost-per-resolution to drive continuous improvement."
    ]

    for b in bullets:
        st.markdown(f"- {b}")

# ==============================
# ANALYSIS | WEATHER
# ==============================
with tab_descriptive_weather:
    st.header("☀️ Analysis | Weather")
    st.write("This analysis examines 2018 NYC weather data to identify key trends and patterns. It highlights typical temperature ranges and clear seasonal cycles, with warmer summers, colder winters, and transitional spring/fall periods. Precipitation patterns are summarized to show variability over time, including periods of heavier rainfall or extended dry spells. Notable weather events, such as heat waves, cold snaps, or major storms, are identified and discussed in terms of their potential impacts on daily life, infrastructure, and city operations.")
    
    # =============================
    # Map ALL weather stations by latitude/longitude
    # =============================
    # column names for lat/lon
    lat_col = "Latitude"
    lon_col = "Longitude"

    df_stations = weather.copy()
    df_stations[lat_col] = pd.to_numeric(df_stations[lat_col], errors="coerce")
    df_stations[lon_col] = pd.to_numeric(df_stations[lon_col], errors="coerce")
    df_stations.dropna(subset=[lat_col, lon_col], inplace=True)

    # Aggregate station points and count observations per station
    station_name_col = "StationName"
    NYC_col = "NYC"

    # Ensure NYC tag exists: map known station names to 'NYC', otherwise '-'
    if station_name_col in df_stations.columns:
        df_stations[station_name_col] = df_stations[station_name_col].astype(str).str.strip()
        try:
            df_stations[NYC_col] = df_stations[station_name_col].str.upper().map(lambda s: 'NYC' if s in _nyc_names else '-')
        except Exception:
            df_stations[NYC_col] = '-'
        group_cols = [station_name_col, NYC_col, lat_col, lon_col]
    else:
        df_stations[NYC_col] = '-'
        group_cols = [NYC_col, lat_col, lon_col]

    stations = (
        df_stations.groupby(group_cols)
            .size()
            .reset_index(name="observations")
        )

    # Show table and map side-by-side (equal size and height)
    height = 600
    col_table, col_map = st.columns([1, 1])
    
    # Use explicit per-widget heights (avoid global CSS to prevent layout bleed/overlap)
    with col_table:
        st.subheader("Weather Stations (unique lat/lon)")
        st.dataframe(stations, height=height)

    # Prepare hover columns for the map (used by the map block below)
    hover_cols = [station_name_col, NYC_col, "observations"]

    # Center on NYC-tagged stations if available, else fallback to overall mean, then default NYC coords
    try:
        nyc_mask = stations[NYC_col].astype(str).str.upper().eq("NYC")
        use_df = stations.loc[nyc_mask] if nyc_mask.any() else stations

        center = {
            "lat": float(use_df[lat_col].mean()),
            "lon": float(use_df[lon_col].mean()),
        }
    except Exception:
        center = {"lat": 40.7128, "lon": -74.0060}

    # Color NYC-tagged stations differently
    color_map = {"NYC": "#b08725", "-": "#4f5a6c"}

    fig_stations = px.scatter_mapbox(
        stations,
        lat=lat_col,
        lon=lon_col,
        size="observations",
        size_max=18,
        color=NYC_col,
        color_discrete_map=color_map,
        hover_data=hover_cols,
        hover_name=station_name_col,
        zoom=8,
        center=center,
        height=height,
        mapbox_style="carto-positron",
    )
    fig_stations.update_layout(margin={"l":0,"r":0,"t":40,"b":0})

    with col_map:
        st.markdown("### Weather Stations Map")
        st.plotly_chart(fig_stations, use_container_width=True, height=height)
    
    st.subheader("☀️ NYC-Specific Weather")
    
    # Summary statistics cards (Avg/Max/Min temps + precipitation/snow totals)
    wn = weather_nyc_grouped.copy()
    wn["Date"] = pd.to_datetime(wn["Date"], errors="coerce")

    def _safe_stat_with_date(df, col, agg="mean", round_digits=1):
        if col not in df.columns or df[col].dropna().empty:
            return ("N/A", "N/A")
        series = df[col].astype(float)
        if agg == "mean":
            val = series.mean()
            # date closest to mean
            idx = (series - val).abs().idxmin()
        elif agg == "max":
            val = series.max()
            idx = series.idxmax()
        elif agg == "min":
            val = series.min()
            idx = series.idxmin()
        else:
            return ("N/A", "N/A")
        date = pd.to_datetime(df.at[idx, "Date"]).date() if pd.notna(idx) else "N/A"
        return (f"{val:.{round_digits}f}", str(date))

    # Temperature stats
    avg_temp, avg_temp_date = _safe_stat_with_date(wn, "MeanTemp", agg="mean", round_digits=1)
    max_temp, max_temp_date = _safe_stat_with_date(wn, "MaxTemp", agg="max", round_digits=1)
    min_temp, min_temp_date = _safe_stat_with_date(wn, "MinTemp", agg="min", round_digits=1)

    # Precipitation totals & days
    precip_col = next((c for c in wn.columns if "precip" in c.lower()), None)
    if precip_col and wn[precip_col].dropna().size:
        total_precip = float(wn[precip_col].astype(float).sum())
        precip_days = int((wn[precip_col].astype(float) > 0).sum())
        total_precip_str = f"{total_precip:.2f}"
    else:
        total_precip_str = "N/A"
        precip_days = "N/A"

    # Snow totals & days (try SnowDepth / Snow / SnowIce)
    snow_col = next((c for c in wn.columns if "snow" in c.lower()), None)
    if snow_col and wn[snow_col].dropna().size:
        total_snow = float(wn[snow_col].astype(float).sum())
        snow_days = int((wn[snow_col].astype(float) > 0).sum())
        total_snow_str = f"{total_snow:.2f}"
    else:
        total_snow_str = "N/A"
        snow_days = "N/A"

    # Render as metrics in a single row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Mean Temp", f"{avg_temp}°", f"Closest date: {avg_temp_date}")
    c2.metric("Max Temp", f"{max_temp}°", f"Date: {max_temp_date}")
    c3.metric("Min Temp", f"{min_temp}°", f"Date: {min_temp_date}")
    c4.metric("Total Precipitation", f"{total_precip_str} in", f"Days w/ precip: {precip_days}")
    c5.metric("Total Snow", f"{total_snow_str} in", f"Days w/ snow: {snow_days}")

    cols = st.columns([1, 1, 1])
    # Ensure Date is datetime for plotting
    if "Date" in weather_nyc_grouped.columns:
        weather_nyc_grouped["Date"] = pd.to_datetime(weather_nyc_grouped["Date"], errors="coerce")

    with cols[0]:
        st.subheader("🌡️ Daily Temperature Trends")
        temp_cols = [c for c in ["MaxTemp", "MinTemp", "MeanTemp"] if c in weather_nyc_grouped.columns]
        fig_temp = px.line(weather_nyc_grouped, x="Date", y=temp_cols, title="Daily Temperature Trends")
        fig_temp.update_layout(margin=dict(l=8, r=8, t=36, b=8), height=360, legend_title="Temp Metric")
        st.plotly_chart(fig_temp, use_container_width=True)

    with cols[1]:
        st.subheader("🌧️ Precipitation & Snow Depth")
        precip_cols = [c for c in ["Precipitation", "SnowDepth"] if c in weather_nyc_grouped.columns]
        fig_precip = px.bar(
            weather_nyc_grouped,
            x="Date",
            y=precip_cols,
            title="Daily Precipitation and Snow Depth",
            barmode="group"
        )
        fig_precip.update_layout(margin=dict(l=8, r=8, t=36, b=8), height=360, legend_title="Precip Metric")
        st.plotly_chart(fig_precip, use_container_width=True)

    with cols[2]:
        st.subheader("🌬️ Wind Metrics")
        wind_cols = [c for c in ["WindSpeed", "MaxSustainedWind", "Gust"] if c in weather_nyc_grouped.columns]
        fig_wind = px.line(
            weather_nyc_grouped,
            x="Date",
            y=wind_cols,
            title="Wind Metrics: WindSpeed · MaxSustainedWind · Gust",
        )
        fig_wind.update_traces(mode="lines")
        fig_wind.update_layout(margin=dict(l=8, r=8, t=36, b=8), height=360, yaxis_title="Speed (source units)", legend_title="Wind Metric")
        st.plotly_chart(fig_wind, use_container_width=True)
    
    # ==============================
    # Notable weather events and potential impacts
    # ==============================
    st.subheader("⚠️ Notable Weather Events & Potential Impacts")

    wn = weather_nyc_grouped.copy()

    def top_events(col, n=5):
        if col in wn.columns:
            return wn[["Date", col]].dropna().nlargest(n, col)
        return pd.DataFrame(columns=["Date", col])

    wind_col = "Gust"
    top_precip = top_events("Precipitation", 5)
    top_snow = top_events("SnowDepth", 5)
    top_wind = top_events(wind_col, 5) if wind_col else pd.DataFrame()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Top Precipitation Days**")
        st.markdown('Potential flooding, transportation delays, or spikes in 311 service requests for water-related issues)')
        st.dataframe(top_precip.style.format({"Precipitation": "{:.2f}"}))

    with col2:
        st.markdown("**Top Snow Depth Days**")
        st.markdown('Potential plowing/salting needs, transit disruption, building/roof concerns')
        st.dataframe(top_snow.style.format({"SnowDepth": "{:.2f}"}))

    with col3:
        st.markdown(f"**High Wind Days** ({wind_col})")
        st.markdown("Potential downed trees/power lines, structural damage, travel hazards")
        st.dataframe(top_wind.style.format({wind_col: "{:.1f}"}))
    
    with col4:    
        # Composite severe-storm candidates: heavy rain + high wind same day
        p90 = wn["Precipitation"].quantile(0.90)
        w90 = wn[wind_col].quantile(0.90)
        storms = wn[(wn["Precipitation"] >= p90) & (wn[wind_col] >= w90)][["Date", "Precipitation", wind_col]]
        if not storms.empty:
            st.markdown("**Potential Severe Storm Days**")
            st.markdown("Heavy precipitation + high wind — likely high impact on infrastructure & calls")
            st.dataframe(storms.style.format({"Precipitation":"{:.2f}", wind_col:"{:.1f}"}))

    # Heat waves and cold snaps (consecutive runs) — display in two columns
    def find_runs(df, col, thresh, op="gt", min_len=3):
        if col not in df.columns:
            return []
        s = df[col].astype(float)
        if op == "gt":
            mask = s > thresh
        else:
            mask = s < thresh
        # identify consecutive True runs
        grp = (mask != mask.shift()).cumsum()
        runs = []
        for _, g in df[mask].groupby(grp[mask]):
            if len(g) >= min_len:
                runs.append((pd.to_datetime(g["Date"].iloc[0]).date(), pd.to_datetime(g["Date"].iloc[-1]).date(), len(g)))
        return runs

    ht_thresh = wn["MeanTemp"].quantile(0.90) if "MeanTemp" in wn.columns else None
    cl_thresh = wn["MeanTemp"].quantile(0.10) if "MeanTemp" in wn.columns else None
    heatwaves = find_runs(wn, "MeanTemp", ht_thresh, op="gt", min_len=3) if ht_thresh is not None else []
    coldsnaps = find_runs(wn, "MeanTemp", cl_thresh, op="lt", min_len=3) if cl_thresh is not None else []

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌡️ Heat Waves")
        if heatwaves:
            st.markdown("Consecutive days above the 90th percentile (MeanTemp)")
            st.markdown('Potential impacts: heat stress, HVAC strain, increased heat/AC complaints')    
            for s, e, l in heatwaves:
                st.write(f"- {s} → {e} ({l} days)")
        else:
            st.write("No heat waves detected")

    with col2:
        st.subheader("❄️ Cold Snaps")
        if coldsnaps:
            st.markdown("Consecutive days below the 10th percentile (MeanTemp)")
            st.markdown('Potential impacts: frozen pipes, heating failures, increased safety/health calls')
            for s, e, l in coldsnaps:
                st.write(f"- {s} → {e} ({l} days)")
        else:
            st.write("No cold snaps detected")



# ==============================
# WEATHER IMPACT ON CALLS
# ==============================

# ==============================
# Merge 311 Service Requests with NYC Weather data
# ==============================
calls_loc = calls_clean.copy()
weather_loc = weather_nyc_grouped.copy()

# Normalize both Date columns to datetime64 (midnight) so merge keys match
# Safely pick the first available created-date-like column from calls_loc
_date_source = None
for _cand in ("Created Date Parsed Date", "Created Date Parsed", "Created Date"):
    if _cand in calls_loc.columns:
        _date_source = _cand
        break
if _date_source is not None:
    calls_loc["Date"] = pd.to_datetime(calls_loc[_date_source], errors="coerce").dt.normalize()
else:
    calls_loc["Date"] = pd.NaT

weather_loc["Date"] = pd.to_datetime(weather_loc["Date"], errors="coerce").dt.normalize()

# Left merge calls with the daily weather summary on Date (use the normalized weather_loc)
calls_with_weather = calls_loc.merge(weather_loc, on="Date", how="left")

# Select significant columns to display
desired_cols = [
    "Unique Key", "Created Date", "Closed Date", "Complaint Type", "Descriptor",
    "Incident Address", "City", "Status", "Resolution Summary",
    "Resolution Action Updated Date", "Community Board", "Open Data Channel Type",
    "Latitude", "Longitude", "MeanTemp", "MinTemp", "MaxTemp", "DewPoint",
    "Precipitation", "WindSpeed", "MaxSustainedWind", "Gust", "Rain", "SnowDepth", "SnowIce", "StationName"
]

# defensive: only keep desired columns that actually exist after the merge
calls_with_weather = calls_with_weather.assign(Date=pd.to_datetime(calls_with_weather["Date"], errors="coerce"))
avail_cols = [c for c in desired_cols if c in calls_with_weather.columns]

# ==============================
# Aggregate to daily level and analyze weather vs call volume
# ==============================
# Prepare daily-level dataset: calls per day + weather summary
daily = (
    calls_with_weather
    .assign(Date=lambda df: pd.to_datetime(df["Date"], errors="coerce"))
    .dropna(subset=["Date"])
    .groupby("Date", dropna=True)
    .agg(call_count=("Unique Key", "count"),
            MeanTemp=("MeanTemp", "mean"),
            MinTemp=("MinTemp", "mean"),
            MaxTemp=("MaxTemp", "mean"),
            DewPoint=("DewPoint", "mean"),
            Precipitation=("Precipitation", "mean"),
            WindSpeed=("WindSpeed", "mean"),
            MaxSustainedWind=("MaxSustainedWind", "mean"),
            Gust=("Gust", "mean"),
            Rain=("Rain", "mean"),
            SnowDepth=("SnowDepth", "mean"),
            SnowIce=("SnowIce", "mean"),
            )
    .reset_index()
)

# Fill NA SnowDepth with 0 for days where MeanTemp is present
if "SnowDepth" in daily.columns and "MeanTemp" in daily.columns:
    mask = daily["MeanTemp"].notna() & daily["SnowDepth"].isna()
    if mask.any():
        daily.loc[mask, "SnowDepth"] = 0.0
        
# identify numeric weather columns
weather_feats = [c for c in daily.columns if c not in ("Date", "call_count")]
num_weather = [c for c in weather_feats if pd.api.types.is_numeric_dtype(daily[c])]

# ==============================
# Visualize weather vs call volume relationships
# ==============================
with tab_merged:
    st.header("🌦️ Merged Analysis | 311 Service Requests & Weather")
    
    st.header("🔎 Key Findings")
    st.markdown(
    """

    Daily 311 call volume varies meaningfully day-to-day, suggesting external factors (like weather) may influence demand. When aggregating calls by day and pairing them with average daily weather conditions, the relationship is directional rather than perfectly linear — which is expected for human-driven service requests.

    **Temperature**: More extreme temperatures (both hot and cold) tend to increase call volumes. This aligns with increased use of building systems (HVAC, plumbing, utilities), which raises the likelihood of issues being noticed and reported.

    **Precipitation (rain / snow)**: Days with precipitation often show spikes in calls, particularly for flooding/leaks, sewer or drainage complaints, and roadway/sidewalk issues. Heavy rain or snow acts as a stress test for city infrastructure, surfacing latent problems.

    **Extreme weather days**: Very cold or very hot days can produce nonlinear effects, and these days tend to product outlier spikes in call volumne rather than gradual increases.
    - Cold snaps → heating failures, frozen pipes
    - Heat waves → AC outages, power issues

    **Geographic stability**: Latitude/longitude averages show no strong correlation with volume at the citywide level, implying that weather impacts are systemic rather than location-specific in this aggregation.
    """
    )
    
    st.header("📊 Possible Causal Links")
    st.markdown("**While this is correlation (not causation), the mechanisms are intuitive:**")
    cols = st.columns(3)
    with cols[0]:
        st.subheader("Weather stresses infrastructure")
        st.markdown(
            """
            - Rain → leaks, flooding, sewer backups
            - Cold → pipe bursts, heating complaints
            - Heat → electrical load and cooling failures
            """
        )
    with cols[1]:
        st.subheader("Human behavior changes")
        st.markdown(
            """
            - People are more likely to notice and report issues when they are home more (bad weather days)
            - People are more likely to notice and report issues when they are outside more (warm, clear days)
            """
        )
    with cols[2]:
        st.subheader("Visibility effect")
        st.markdown(
            """
            - Weather often reveals pre-existing problems rather than directly causing them
            - Use weather-triggered monitoring to surface latent infrastructure issues earlier
            """
        )
    
    st.header("🔍 Business and Operational Implications")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Staffing Optimization")
        st.markdown(
            """
            - Pre-emptively increase call center and field staff during forecasted storms or temperature extremes.
            - Encourage online reporting during high-volume weather events to manage call center load.
            - Reduce response bottlenecks during predictable surges.
            """
        )
    with c2:
        st.subheader("Preventive Infrastructure Planning")
        st.markdown(
            """
            - Use weather patterns to prioritize inspections and maintenance before stress events (e.g., inspections of drainage before heavy rain; heating system checks ahead of cold snaps).
            - Shift from reactive to proactive service delivery to reduce incident frequency during extreme weather.
            """
        )
    with c3:
        st.subheader("Cost & Efficiency Gains")
        st.markdown(
            """
            - Better alignment of resources reduces overtime costs and backlog accumulation; supports seasonal budget allocation and emergency response funding.
            - Improve SLA performance during high-demand periods, increasing customer satisfaction and public trust.
            """
        )
    
    corr_ser = daily[["call_count"] + num_weather].corr().loc["call_count"].sort_values(ascending=False)
    corr_df = corr_ser.reset_index().rename(columns={"index": "variable", "call_count": "pearson_r"})

    # =============================
    # Scatter plots with OLS trendline for all correlated weather vars
    # =============================
    st.header("📈 Scatter Plots: Daily Call Counts vs Weather Variables with OLS Trendline")
    st.write("These scatter plots visualize the relationship between daily 311 service request counts and various weather variables.")
    st.write("Ordinary Least Squares (OLS) trendline is included to illustrate the linear relationship, if any, between the two variables. The R² value from the OLS fit indicates how well the weather variable explains the variance in daily call counts.")
    exclude = {"call_count", "Rain", "SnowIce"}
    corr_vars = set(corr_df["variable"].tolist())
    # preserve original column order from `daily`
    vars_to_plot = [c for c in daily.columns if c in corr_vars and c not in exclude]

    # keep only numeric, non-empty vars
    clean_vars = []
    for v in vars_to_plot:
        if v not in daily.columns:
            continue
        if not pd.api.types.is_numeric_dtype(daily[v]) or daily[v].dropna().empty:
            continue
        clean_vars.append(v)

    # Render 3 plots per row in original dataset order, with per-plot automated writeup beneath each plot
    def _emoji_for_var(v: str) -> str:
        s = v.lower()
        if "temp" in s or "mean" in s or "max" in s or "min" in s:
            return "🌡️"
        if "precip" in s or "rain" in s:
            return "🌧️"
        if "wind" in s or "gust" in s:
            return "🌬️"
        if "snow" in s or "snowdepth" in s or "ice" in s:
            return "❄️"
        if "dew" in s:
            return "💧"
        return "📈"

    for i in range(0, len(clean_vars), 3):
        row_vars = clean_vars[i : i + 3]
        cols = st.columns(3)
        for j, v in enumerate(row_vars):
            try:
                # Header per plot with an emoji
                emoji = _emoji_for_var(v)
                cols[j].subheader(f"{emoji} Calls vs {v}")

                fig = px.scatter(
                    daily,
                    x=v,
                    y="call_count",
                    trendline="ols",
                    labels={v: v, "call_count": "Daily call count"},
                    height=420,
                )

                cols[j].plotly_chart(fig, use_container_width=True)

                # Automated per-plot analysis directly below the plot
                try:
                    df_v = daily[[v, "call_count"]].dropna()
                    n = len(df_v)
                    if n < 8:
                        cols[j].markdown(f"- **{v}**: insufficient data (n={n}) to draw reliable conclusions.")
                        continue

                    r = float(df_v["call_count"].corr(df_v[v]))
                    lr = LinearRegression().fit(df_v[[v]].values, df_v["call_count"].values)
                    coef = float(lr.coef_[0])
                    r2_single = float(lr.score(df_v[[v]].values, df_v["call_count"].values))

                    def _strength_label(rval):
                        ar = abs(rval)
                        if ar >= 0.7:
                            return "strong"
                        if ar >= 0.4:
                            return "moderate"
                        if ar >= 0.2:
                            return "weak-to-moderate"
                        return "weak / negligible"

                    direction = "positive" if r > 0 else ("negative" if r < 0 else "no linear")
                    strength = _strength_label(r)

                    if abs(r) < 0.2:
                        action = "Little-to-no linear relationship; weather alone is unlikely to drive call volume for this variable."
                    elif r > 0:
                        action = "As this variable increases, daily calls tend to increase. Consider monitoring and pre-positioning resources when this variable is elevated."
                    else:
                        action = "As this variable increases, daily calls tend to decrease. Investigate whether this reflects fewer weather-driven incidents."

                    cols[j].markdown(
                        f"- Pearson r = {r:.3f} ({strength}, {direction})  \n"
                        f"- OLS coef = {coef:.3f} calls/unit; R² = {r2_single:.3f}  \n"
                        f"- {action}"
                    )
                except Exception as _e:
                    cols[j].markdown(f"- **{v}**: could not analyze ({_e})")
            except Exception as e:
                cols[j].write(f"Could not plot {v}: {e}")

    # Explanation caption (general)
    st.caption(
        "Interpretation: each scatter plot shows daily 311 call counts against a specific weather variable, with an OLS trendline indicating the linear relationship. "
        "The R² value quantifies how much of the variance in daily calls is explained by that weather variable alone."
    )

    # =============================
    # Average calls by temperature bins
    # =============================
    st.header("🌡️ Average Daily Calls by Mean Temperature Bins")
    if "MeanTemp" in daily.columns:
        df_temp = daily.copy()
        df_temp["_mean_temp"] = df_temp["MeanTemp"].astype(float)
        df_temp["_temp_bin"] = pd.cut(df_temp["_mean_temp"], bins=6)
        agg_temp = df_temp.groupby("_temp_bin")["call_count"].agg(["mean", "count"]).reset_index()
        # Convert Interval bins to string labels for Plotly/JSON serialization
        agg_temp["_temp_bin_label"] = agg_temp["_temp_bin"].astype(str)
        fig = px.bar(
        agg_temp,
        x="_temp_bin_label",
        y="mean",
        labels={"_temp_bin_label": "MeanTemp bin", "mean": "Avg daily calls"},
        height=420
        )
        st.plotly_chart(fig, use_container_width=True)
        # Caption: what is calculated and the practical impact
        st.caption(
        "What this calculates: daily MeanTemp is grouped into 6 temperature bins and the plot shows the average number of 311 service requests per bin. "
        "Impact: if certain temperature bins (hot or cold) show higher call volumes, this suggests temperature-driven service demand (e.g., heating, cooling, water/pipe issues), "
        "informing targeted staffing, preventative maintenance, and event-driven resource allocation."
        )
    
    # ==============================
    # Display samples of merged data
    st.header("🗂️ Sample Data: Merged 311 Service Requests with Weather")
    st.write("Below are samples of the merged dataset combining 311 service requests with corresponding daily weather data. This merged data enables further analysis of how weather conditions may influence service request patterns and volumes.")
    
    st.subheader("Sample of Merged Data")
    st.dataframe(calls_with_weather.sort_values("Date", ascending=True)[avail_cols].head(50))
    
    st.subheader("Sample of Merged Data, daily aggregation")
    st.dataframe(daily.sort_values("Date").reset_index(drop=True))

# ==============================
# Streamlit Dashboard: Forecasting
# ==============================
with tab_forecasting:
    st.header("🔮 Forecasting")
    
    # Simple 7-day forecasting: linear model on day-of-week + trend + simple weather predictors (naive weather forecast = recent average)

    st.subheader("Assumptions")
    st.markdown(
        """
        - Short-term forecast: next 7 days only.
        - Key drivers: day-of-week effects, slow linear trend, and weather (MeanTemp, Precipitation, WindSpeed, Gust, SnowDepth when available).
        - Weather for the forecast window is unknown; we use the recent 7-day average of each weather feature as a naive forecast.
        - Model: ordinary least squares (linear regression) using historical daily aggregates.
        - Fallback: if insufficient training data, use the rolling 7-day average of call counts as the forecast (constant for each of the 7 days).
        """
    )
    
    st.subheader("Key Influencing Factors")
    st.markdown(
        """
        - Day-of-week (recurring weekly patterns).
        - Recent baseline demand (7-day rolling average).
        - Slow linear trend (captures seasonality drift).
        - Weather (MeanTemp, Precipitation, WindSpeed/Gust, SnowDepth) using recent averages as naive future weather.
        """
    )

    df = daily.copy().sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Feature engineering
    df["dow"] = df["Date"].dt.dayofweek  # 0=Mon .. 6=Sun
    dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=True)
    df = pd.concat([df, dow_dummies], axis=1)

    # trend (days since start)
    df["trend"] = (df["Date"] - df["Date"].min()).dt.days.astype(float)

    # rolling 7-day average of calls (stable short-term baseline)
    df["roll_7"] = df["call_count"].rolling(7, min_periods=1).mean()

    # candidate weather predictors (use only those present)
    candidate_weather = ["MeanTemp", "Precipitation", "WindSpeed", "Gust", "SnowDepth"]
    weather_feats = [c for c in candidate_weather if c in df.columns]

    # fill small gaps in weather with forward/backward fill then with overall mean
    for c in weather_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].ffill().bfill().fillna(df[c].mean())

    # Prepare X, y
    feature_cols = ["trend", "roll_7"] + list(dow_dummies.columns) + weather_feats
    X = df[feature_cols].copy()
    y = df["call_count"].astype(float)

    # train linear regression
    lr = LinearRegression()
    # drop any rows with NaN in X/y
    mask = X.notna().all(axis=1) & y.notna()
    X_train = X.loc[mask]
    y_train = y.loc[mask]
    lr.fit(X_train.values, y_train.values)
    y_pred_train = lr.predict(X_train.values)
    train_r2 = r2_score(y_train.values, y_pred_train)

    # build future feature frame
    last_date = df["Date"].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    fut = pd.DataFrame({"Date": future_dates})
    fut["dow"] = fut["Date"].dt.dayofweek
    fut_dummies = pd.get_dummies(fut["dow"], prefix="dow", drop_first=True)
    # ensure same dummy columns as training
    for col in dow_dummies.columns:
        fut[col] = fut_dummies.get(col, 0)

    fut["trend"] = (fut["Date"] - df["Date"].min()).dt.days.astype(float)

    # naive weather forecast: recent 7-day mean
    recent = df.tail(7)
    for c in weather_feats:
        recent_mean = float(recent[c].mean())
        fut[c] = recent_mean

    # roll_7 predictor: use last observed roll_7 (stable short-term baseline)
    last_roll_7 = float(df["roll_7"].iloc[-1])
    fut["roll_7"] = last_roll_7

    # ensure column order
    X_future = fut[feature_cols].fillna(0)

    # predict and post-process
    fut_preds = lr.predict(X_future.values)
    fut_preds = [max(0, float(x)) for x in fut_preds]  # no negatives
    fut_preds_int = [int(round(x)) for x in fut_preds]

    forecast_df = fut[["Date"]].copy()
    forecast_df["forecast_calls"] = fut_preds_int
    forecast_df["forecast_raw"] = fut_preds

    # present results
    st.metric("Model", "Linear regression (dow + trend + roll_7 + weather)")
    st.caption("Interpreting the model: the linear regression predicts daily 311 call counts as a sum of a slow time trend, recent 7‑day baseline, day‑of‑week effects, and recent-average weather. Each coefficient approximates the marginal change in daily calls per unit change in that predictor (positive = more calls, negative = fewer).")
    
    st.metric("Training R²", f"{train_r2:.3f}")
    st.caption("R² indicates how much of the variance in daily calls is explained by the model on historical data. Higher values (closer to 1) indicate better fit. Treat results as correlational and use operational judgement when acting on forecasts.")

    st.subheader("7-Day Forecast (next 7 days)")
    st.dataframe(forecast_df.assign(Date=lambda d: d["Date"].dt.date).set_index("Date"))

    # show a plot: recent history + forecast with distinct colors
    hist_plot = df[["Date", "call_count"]].copy()
    recent_hist = hist_plot.tail(60).reset_index(drop=True)
    fut_plot = forecast_df[["Date", "forecast_calls"]].rename(columns={"forecast_calls": "call_count"}).reset_index(drop=True)

    # build separate traces so forecast is visually distinct (different color + dashed line)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_hist["Date"],
        y=recent_hist["call_count"],
        mode="lines+markers",
        name="Historical",
        line=dict(color="#1f77b4"),
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=fut_plot["Date"],
        y=fut_plot["call_count"],
        mode="lines+markers",
        name="Forecast (7d)",
        line=dict(color="#ff7f0e", dash="dash"),
        marker=dict(size=8, symbol="circle-open")
    ))

    fig.update_layout(
        title="Recent calls (60d) + 7-day forecast",
        xaxis_title="Date",
        yaxis_title="Call count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=8, r=8, t=40, b=8),
        height=520
    )

    st.plotly_chart(fig, use_container_width=True)

# ==============================
# Export cleaned and merged data
# ==============================
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "calls_with_weather.xlsx"
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        calls.to_excel(writer, sheet_name="calls_cleaned", index=False)
        calls_grouped.to_excel(writer, sheet_name="calls_aggregated", index=False)
        weather.to_excel(writer, sheet_name="weather_cleaned", index=False)
        calls_with_weather.to_excel(writer, sheet_name="calls_with_weather", index=False)

    st.success(f"✅ Exported Excel to {out_file.resolve()}")

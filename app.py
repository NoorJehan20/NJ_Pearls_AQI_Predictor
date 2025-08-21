# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from datetime import datetime, date, timedelta, timezone
import plotly.express as px
import plotly.graph_objects as go

# =========================
# CONFIG — EDIT THESE 3
# =========================
GITHUB_USER = "NoorJehan20"
GITHUB_REPO = "aqi-pm25-collector"
GITHUB_BRANCH = "main"              # or "master"
GITHUB_OUTPUT_DIR = "outputs"        # folder in your repo where CSVs are saved

CITY_NAME = "Karachi, Pakistan"
LOCAL_TZ_OFFSET = "+05:00"          # for display only in headers

# =========================
# STREAMLIT PAGE SETUP
# =========================
st.set_page_config(page_title=f"{CITY_NAME} AQI Dashboard", layout="wide")
st.markdown(
    """
    <style>
      .hero-aqi {
        padding: 18px 22px; border-radius: 18px;
        background: linear-gradient(135deg, #111827, #1f2937); color: #fff;
        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
      }
      .subtle {
        color: #9CA3AF; font-size: 0.9rem;
      }
      .card {
        padding: 16px; border-radius: 16px; background: #ffffff;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        height: 100%;
      }
      .divider { border-top: 1px solid #e5e7eb; margin: 12px 0 18px; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# UTILITIES
# =========================
AQI_BANDS = [
    (0, 50,  "Good",                        "🟢", "#d1fae5"),
    (51,100, "Moderate",                    "🟡", "#fef9c3"),
    (101,150,"Unhealthy (Sensitive)",      "🟠", "#ffedd5"),
    (151,200,"Unhealthy",                   "🔴", "#fee2e2"),
    (201,300,"Very Unhealthy",              "🟣", "#ede9fe"),
    (301,500,"Hazardous",                   "⚫", "#f5f5f5")
]

WHO_LIMITS = {   # daily or short-term guideline ballparks
    "pm25": 15,   # µg/m³ (24h guideline)
    "no2": 25,    # µg/m³ (24h guideline)
    "so2": 40,    # µg/m³ (24h guideline) - note: WHO has 24h 40 for SO2 used by many dashboards
    "o3": 100,    # µg/m³ (8h avg guideline)
    "co": 4       # mg/m³ (24h guideline)
}

def get_aqi_band(aqi: float):
    if pd.isna(aqi): return ("Unknown", "❔", "#f3f4f6")
    aqi = float(aqi)
    for low, high, label, icon, color in AQI_BANDS:
        if low <= aqi <= high:
            return (label, icon, color)
    return ("Unknown", "❔", "#f3f4f6")

def pick_col(df: pd.DataFrame, options, default=np.nan):
    for c in options:
        if c in df.columns:
            return df[c]
    return pd.Series([default]*len(df))

def pick_scalar(latest: pd.Series, options, default=np.nan):
    for c in options:
        if c in latest.index:
            return latest[c]
    return default

def parse_date_in_name(name: str):
    # catches e.g., daily_aqi_2025-08-17.csv OR aqi_2025-08-20.csv OR realtime_aqi_2025-08-20_*.csv
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except:
            return None
    return None

@st.cache_data(ttl=600)
def list_repo_csvs(user, repo, branch, dir_path):
    """List CSV files in a GitHub folder via API."""
    url = f"https://api.github.com/repos/{user}/{repo}/contents/{dir_path}?ref={branch}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    items = r.json()
    csvs = []
    for it in items:
        if it.get("type") == "file" and it.get("name","").lower().endswith(".csv"):
            csvs.append({
                "name": it["name"],
                "download_url": it["download_url"],
                "date_in_name": parse_date_in_name(it["name"])  # may be None
            })
    return csvs

@st.cache_data(ttl=600)
def fetch_latest_csv_from_github():
    """Pick the newest by date in filename; if none have date, just take lexicographically last."""
    csvs = list_repo_csvs(GITHUB_USER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_OUTPUT_DIR)
    if not csvs:
        raise RuntimeError("No CSV files found in the GitHub output folder.")

    dated = [c for c in csvs if c["date_in_name"] is not None]
    if dated:
        dated.sort(key=lambda x: x["date_in_name"])
        latest = dated[-1]
    else:
        # fallback: lexical sort by name as a rough approximation
        csvs.sort(key=lambda x: x["name"])
        latest = csvs[-1]

    df = pd.read_csv(latest["download_url"])
    return latest["name"], df

def coerce_datetime(df: pd.DataFrame):
    # Try to coerce a few likely datetime columns
    for c in ["datetime", "time", "timestamp"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c])
                return c
            except:
                pass
    # if none found, make a synthetic sequential time index (fallback)
    df["datetime"] = pd.to_datetime(pd.Series(range(len(df))), unit="h", origin="unix")
    return "datetime"

# =========================
# LOAD DATA (GitHub OR Upload)
# =========================
st.title(f"🌍 {CITY_NAME} — Air Quality Dashboard")
st.caption("A clean, single-page overview: Predicted AQI, weather, pollutants, forecasts, comparisons.")

left, right = st.columns([0.65, 0.35])
with right:
    st.markdown("**Data Source**")
    source = st.radio("Choose input:", ["GitHub (auto-latest)", "Upload CSV"], index=0)
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV exported by your pipeline", type=["csv"])

try:
    if uploaded:
        df = pd.read_csv(uploaded)
        src_name = uploaded.name
    else:
        src_name, df = fetch_latest_csv_from_github()
except Exception as e:
    st.error(f"Could not load data from GitHub: {e}")
    uploaded = st.file_uploader("As a fallback, upload a CSV", type=["csv"], key="fallback")
    if uploaded:
        df = pd.read_csv(uploaded)
        src_name = uploaded.name
    else:
        st.stop()

time_col = coerce_datetime(df)
df.sort_values(by=time_col, inplace=True)
df.reset_index(drop=True, inplace=True)

# =========================
# FLEXIBLE COLUMN MAPPING
# =========================
# AQI
df["predicted_aqi"] = pick_col(df, ["Predicted_AQI", "predicted_aqi", "aqi_pred", "aqi_predicted"])
df["waqi_aqi"]      = pick_col(df, ["Live_AQI_used", "waqi_aqi", "aqi_live", "live_aqi"])

# Weather (common names in your Kaggle/VC output)
df["temperature"] = pick_col(df, ["temp", "temperature"])
df["humidity"]    = pick_col(df, ["humidity", "rh"])
df["wind_speed"]  = pick_col(df, ["windspeed", "wind_speed"])
df["pressure"]    = pick_col(df, ["pressure"])
df["visibility"]  = pick_col(df, ["visibility"])
df["precip"]      = pick_col(df, ["precip", "precipitation"])

# Pollutants (if available)
for pol in ["pm25","no2","so2","o3","co"]:
    if pol not in df.columns:
        df[pol] = np.nan  # keep column for unified UI

latest = df.iloc[-1]

# =========================
# HERO — CURRENT AQI
# =========================
aqi_now = float(pick_scalar(latest, ["predicted_aqi"], default=np.nan))
band_label, band_icon, band_color = get_aqi_band(aqi_now)

with left:
    st.markdown(
        f"""
        <div class='hero-aqi'>
          <div class='subtle'>Last file: <b>{src_name}</b> • Local time: {latest[time_col]} ({LOCAL_TZ_OFFSET})</div>
          <h1 style='margin: 6px 0 4px 0; font-size: 40px;'>
            Predicted AQI: <b>{int(aqi_now) if not np.isnan(aqi_now) else '—'}</b> {band_icon}
          </h1>
          <div style='font-size: 18px; margin-bottom: 6px;'>Status: <b>{band_label}</b></div>
          <div class='subtle'>Source: Visual Crossing (features) + Trained XGBoost model (CI/CD) • Baseline check: WAQI live AQI</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# SNAPSHOT CARDS — WEATHER
# =========================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.subheader("Current Weather Snapshot")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("<div class='card'>🌡️ **Temperature**<br/>"
                f"<h3>{round(pick_scalar(latest,['temperature'], np.nan),1) if not np.isnan(pick_scalar(latest,['temperature'], np.nan)) else '—'} °C</h3></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'>💧 **Humidity**<br/>"
                f"<h3>{round(pick_scalar(latest,['humidity'], np.nan),1) if not np.isnan(pick_scalar(latest,['humidity'], np.nan)) else '—'} %</h3></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'>🌬️ **Wind Speed**<br/>"
                f"<h3>{round(pick_scalar(latest,['wind_speed'], np.nan),1) if not np.isnan(pick_scalar(latest,['wind_speed'], np.nan)) else '—'} m/s</h3></div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='card'>⏲️ **Pressure**<br/>"
                f"<h3>{round(pick_scalar(latest,['pressure'], np.nan),1) if not np.isnan(pick_scalar(latest,['pressure'], np.nan)) else '—'} hPa</h3></div>", unsafe_allow_html=True)

# =========================
# POLLUTANT CARDS
# =========================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.subheader("Pollutant Snapshot (if available)")
pcols = st.columns(5)
pollutant_labels = {
    "pm25":"PM2.5 (µg/m³)", "no2":"NO₂ (µg/m³)", "so2":"SO₂ (µg/m³)",
    "o3":"O₃ (µg/m³)", "co":"CO (mg/m³)"
}
for i, pol in enumerate(["pm25","no2","so2","o3","co"]):
    with pcols[i]:
        val = pick_scalar(latest, [pol], np.nan)
        show = "—" if np.isnan(val) else round(float(val), 2)
        st.markdown(f"<div class='card'>**{pollutant_labels[pol]}**<br/><h3>{show}</h3></div>", unsafe_allow_html=True)

# =========================
# PREDICTED vs WAQI (100 newest)
# =========================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.subheader("Predicted vs WAQI (latest 100 points)")
tail = df.tail(100).copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=tail[time_col], y=tail["predicted_aqi"],
                         mode="lines+markers", name="Predicted AQI"))
if tail["waqi_aqi"].notna().any():
    fig.add_trace(go.Scatter(x=tail[time_col], y=tail["waqi_aqi"],
                             mode="lines+markers", name="WAQI AQI"))

# add light AQI bands
for low, high, label, icon, color in AQI_BANDS:
    fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=low, y1=high,
                  fillcolor=color, opacity=0.16, line_width=0)

fig.update_layout(
    height=360, margin=dict(l=10, r=10, t=40, b=10),
    yaxis_title="AQI", xaxis_title="Time"
)
st.plotly_chart(fig, use_container_width=True)

# =========================
# HOURLY (NEXT 24H) & DAILY (NEXT 7D)
# =========================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.subheader("Forecasts")

leftF, rightF = st.columns(2)

with leftF:
    st.markdown("**⏳ Hourly AQI (next 24h)**")
    hourly = df.copy()
    # assume your pipeline appends future hourly rows; choose last 24 rows
    hourly_last24 = hourly.tail(24)
    fig_h = px.line(hourly_last24, x=time_col, y="predicted_aqi", markers=True)
    fig_h.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10),
                        yaxis_title="AQI", xaxis_title="Time")
    st.plotly_chart(fig_h, use_container_width=True)

with rightF:
    st.markdown("**📅 Daily AQI (7-day mean)**")
    df["_date"] = pd.to_datetime(df[time_col]).dt.date
    daily = df.groupby("_date", as_index=False)["predicted_aqi"].mean().tail(7)
    fig_d = px.bar(daily, x="_date", y="predicted_aqi", text_auto=True)
    fig_d.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10),
                        yaxis_title="AQI", xaxis_title="Date")
    st.plotly_chart(fig_d, use_container_width=True)

# =========================
# WHO COMPARISON (bar)
# =========================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.subheader("WHO Guideline Comparison")

compare_rows = []
for pol, limit in WHO_LIMITS.items():
    val = pick_scalar(latest, [pol], np.nan)
    if not np.isnan(val):
        compare_rows.append({
            "Pollutant": pol.upper(),
            "Value": float(val),
            "WHO Limit": float(limit)
        })

if compare_rows:
    cmp = pd.DataFrame(compare_rows)
    fig_who = px.bar(cmp, x="Pollutant", y=["Value", "WHO Limit"], barmode="group", text_auto=True)
    fig_who.update_layout(height=340, margin=dict(l=10,r=10,t=30,b=10),
                          yaxis_title="Concentration", xaxis_title="Pollutant")
    st.plotly_chart(fig_who, use_container_width=True)
else:
    st.info("WHO comparison hidden — pollutant columns not present in the latest CSV.")

# =========================
# FOOTER / NOTES
# =========================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown(
    f"""
**Notes**  
- **City:** {CITY_NAME} • **Timezone offset (display):** {LOCAL_TZ_OFFSET}  
- **Data sources:** Visual Crossing (meteorology), WAQI (live AQI baseline).  
- **Model:** Saved XGBoost (deployed via CI/CD), uses engineered features (incl. one-step lag).  
- **Caveat:** WAQI free-tier sometimes returns **stale/glitchy** values when stations don't update.  
  For reliability, compare predictions to **trusted station snapshots** (screenshots) at the target time.  
- **File loaded:** `{src_name}` (auto-selected latest from GitHub `{GITHUB_OUTPUT_DIR}/`).  
    """
)

with st.expander("Column mapping used (for debugging)"):
    mapping = {
        "predicted_aqi": ["Predicted_AQI","predicted_aqi","aqi_pred","aqi_predicted"],
        "waqi_aqi": ["Live_AQI_used","waqi_aqi","aqi_live","live_aqi"],
        "temperature": ["temp","temperature"],
        "humidity": ["humidity","rh"],
        "wind_speed": ["windspeed","wind_speed"],
        "pressure": ["pressure"],
        "visibility": ["visibility"],
        "precip": ["precip","precipitation"],
        "pm25": ["pm25"], "no2": ["no2"], "so2": ["so2"], "o3": ["o3"], "co": ["co"]
    }
    st.json(mapping)
    st.dataframe(df.tail(5))

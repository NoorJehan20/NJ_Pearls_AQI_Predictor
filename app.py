# app_test.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# CONFIG
GITHUB_USER = "NoorJehan20"
GITHUB_REPO = "aqi-pm25-collector"
GITHUB_BRANCH = "main"
GITHUB_OUTPUT_DIR = "outputs"

CITY_NAME = "Karachi, Pakistan"
LOCAL_TZ_OFFSET = "+05:00"

st.set_page_config(page_title=f"Pearls AQI Predictor", layout="wide")

# STYLE
st.markdown(
    """
    <style>
      .hero-aqi {
        padding: 18px 22px; border-radius: 18px;
        background: linear-gradient(135deg, #111827, #1f2937); color: #fff;
        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
      }
      .subtle { color: #9CA3AF; font-size: 0.9rem; }
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

# HELPERS
AQI_BANDS = [
    (0, 50,  "Good", "🙂", "#d1fae5"),
    (51,100, "Moderate", "😐", "#fef9c3"),
    (101,150,"Unhealthy (Sensitive)", "🤧", "#ffedd5"),
    (151,200,"Unhealthy", "😷", "#fee2e2"),
    (201,300,"Very Unhealthy", "🤢", "#ede9fe"),
    (301,500,"Hazardous", "☠️", "#f5f5f5")
]

def get_aqi_band(aqi: float):
    if pd.isna(aqi): return ("Unknown", "❔", "#f3f4f6")
    aqi = float(aqi)
    for low, high, label, icon, color in AQI_BANDS:
        if low <= aqi <= high:
            return (label, icon, color)
    return ("Unknown", "❔", "#f3f4f6")

def pick_col(df, options, default=np.nan):
    for c in options:
        if c in df.columns:
            return df[c]
    return pd.Series([default]*len(df))

def pick_scalar(latest, options, default=np.nan):
    for c in options:
        if c in latest.index:
            return latest[c]
    return default

def parse_date_in_name(name: str):
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", name)
    if m:
        try: return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except: return None
    return None

@st.cache_data(ttl=600)
def list_repo_csvs(user, repo, branch, dir_path):
    url = f"https://api.github.com/repos/{user}/{repo}/contents/{dir_path}?ref={branch}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    items = r.json()
    return [
        {"name": it["name"], "download_url": it["download_url"], "date_in_name": parse_date_in_name(it["name"])}
        for it in items if it.get("type") == "file" and it["name"].endswith(".csv")
    ]

@st.cache_data(ttl=600)
def fetch_latest_csv_from_github():
    csvs = list_repo_csvs(GITHUB_USER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_OUTPUT_DIR)
    if not csvs: raise RuntimeError("No CSV files found")
    dated = [c for c in csvs if c["date_in_name"]]
    latest = max(dated, key=lambda x: x["date_in_name"]) if dated else max(csvs, key=lambda x: x["name"])
    df = pd.read_csv(latest["download_url"])
    return latest["name"], df

def coerce_datetime(df):
    for c in ["datetime", "time", "timestamp"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c])
                return c
            except: pass
    df["datetime"] = pd.to_datetime(range(len(df)), unit="h", origin="unix")
    return "datetime"

# LOAD DATA
st.title(f"🌍 Pearls AQI Dashboard -- {CITY_NAME}")
st.caption("Predicted AQI, WAQI baseline, and weather-driven insights.")

src_name, df = fetch_latest_csv_from_github()
time_col = coerce_datetime(df)
df.sort_values(by=time_col, inplace=True)
df.reset_index(drop=True, inplace=True)

# AQI + weather
df["predicted_aqi"] = pick_col(df, ["Predicted_AQI", "predicted_aqi", "aqi_pred"])
df["waqi_aqi"]      = pick_col(df, ["Live_AQI_used", "waqi_aqi", "live_aqi"])
df["temperature"]   = pick_col(df, ["temp", "temperature"])
df["humidity"]      = pick_col(df, ["humidity", "rh"])
df["wind_speed"]    = pick_col(df, ["windspeed", "wind_speed"])
df["pressure"]      = pick_col(df, ["pressure"])
df["visibility"]    = pick_col(df, ["visibility"])
df["precip"]        = pick_col(df, ["precip", "precipitation"])

latest = df.iloc[-1]

# HERO
aqi_now = float(pick_scalar(latest, ["predicted_aqi"], np.nan))
band_label, band_icon, band_color = get_aqi_band(aqi_now)

st.markdown(
    f"""
    <div class='hero-aqi'>
      <div class='subtle'>File: <b>{src_name}</b> • Local time: {latest[time_col]} ({LOCAL_TZ_OFFSET})</div>
      <h1>Predicted AQI: <b>{int(aqi_now) if not np.isnan(aqi_now) else '—'}</b> {band_icon}</h1>
      <div>Status: <b>{band_label}</b></div>
      <div class='subtle'>Baseline check: WAQI live AQI</div>
    </div>
    """, unsafe_allow_html=True
)

# WEATHER SNAPSHOT
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.subheader("Current Weather Snapshot")

c1, c2, c3, c4 = st.columns(4)

# Swapped Pressure with Dew Point
weather_metrics = [
    ("temperature", "Temperature", "°C", "🌡️"),
    ("humidity", "Humidity", "%", "💧"),
    ("wind_speed", "Wind Speed", "m/s", "🌬️"),
    ("dew", "Dew Point", "°C", "❄️"),  # ✅ replacing pressure
]

for i, (col, label, unit, emoji) in enumerate(weather_metrics):
    with [c1, c2, c3, c4][i]:
        val = pick_scalar(latest, [col], np.nan)
        show = "—" if np.isnan(val) else round(val, 1)
        st.markdown(
            f"""
            <div class='card' style='color:#111827;'>
                {emoji} <b>{label}</b><br/>
                <h3 style='color:#111827;'>{show} {unit}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

# PREDICTED vs WAQI
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.subheader("Predicted vs WAQI (latest 100 points)")

tail = df.tail(100)
fig = go.Figure()
fig.add_trace(go.Scatter(x=tail[time_col], y=tail["predicted_aqi"], mode="lines+markers", name="Predicted AQI"))
if tail["waqi_aqi"].notna().any():
    fig.add_trace(go.Scatter(x=tail[time_col], y=tail["waqi_aqi"], mode="lines+markers", name="WAQI AQI"))

# light AQI bands
for low, high, _, _, color in AQI_BANDS:
    fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=low, y1=high,
                  fillcolor=color, opacity=0.16, line_width=0)

fig.update_layout(height=360, yaxis_title="AQI", xaxis_title="Time")
st.plotly_chart(fig, use_container_width=True)

# HOURLY & DAILY FORECASTS
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.subheader("Forecasts")

leftF, rightF = st.columns(2)
with leftF:
    st.markdown("**⏳ Hourly AQI (next 24h)**")
    hourly_last24 = df.tail(24)
    fig_h = px.line(hourly_last24, x=time_col, y="predicted_aqi", markers=True)
    st.plotly_chart(fig_h, use_container_width=True)

with rightF:
    st.markdown("**📅 Daily AQI (7-day mean)**")
    df["_date"] = pd.to_datetime(df[time_col]).dt.date
    daily = df.groupby("_date", as_index=False)["predicted_aqi"].mean().tail(7)
    fig_d = px.bar(daily, x="_date", y="predicted_aqi", text_auto=True)
    st.plotly_chart(fig_d, use_container_width=True)

# FOOTER
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown(f"""
**Notes**  
- City: {CITY_NAME}  
- Sources: Visual Crossing (weather features) + WAQI (baseline AQI).  
- Model: Trained XGBoost (deployed via CI/CD).  
- File loaded: `{src_name}`  
""")

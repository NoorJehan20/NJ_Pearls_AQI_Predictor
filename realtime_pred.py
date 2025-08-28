import requests
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# USER FILLABLE CONFIG
VISUAL_CROSSING_KEY = os.getenv("VISUAL_CROSSING_KEY")
WAQI_TOKEN = os.getenv("WAQI_TOKEN")            
MODEL_PATH = "best_model_xgboost_new.pkl"   
CITY = "Karachi"
LAT, LON = 24.8607, 67.0011
FORECAST_DAYS = 3
TIMEZONE = "Asia/Karachi"
OUT_CSV = "realtime_aqi_forecast.csv"

# HELPERS
def categorize_aqi(aqi):
    aqi = float(aqi)
    if aqi <= 50: return "Good 😀"
    if aqi <= 100: return "Moderate 😐"
    if aqi <= 150: return "Unhealthy for Sensitive Groups 😷"
    if aqi <= 200: return "Unhealthy 😷"
    if aqi <= 300: return "Very Unhealthy ☠"
    return "Hazardous 💀"

def safe_get(d, *keys, default=np.nan):
    """Nested-get with default for json responses."""
    try:
        for k in keys:
            d = d[k]
        return d
    except Exception:
        return default

# Loading model
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
except Exception as e:
    raise SystemExit(f"❌ Failed to load model at {MODEL_PATH}: {e}")

# Fetch live AQI from WAQI (main)
def fetch_waqi(lat, lon, token):
    try:
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok":
            print("⚠️ WAQI returned non-ok status:", data.get("data"))
            return None, None
        aqi = safe_get(data, "data", "aqi", default=None)
        time_iso = safe_get(data, "data", "time", "iso", default=None)
        # parse time (WAQI uses ISO with tz)
        ts = None
        if time_iso:
            try:
                ts = datetime.fromisoformat(time_iso)
            except Exception:
                try:
                    ts = datetime.strptime(time_iso, "%Y-%m-%d %H:%M:%S %z")
                except Exception:
                    ts = None
        return aqi, ts
    except Exception as e:
        print("❌ Error fetching WAQI:", e)
        return None, None

live_aqi, live_ts = fetch_waqi(LAT, LON, WAQI_TOKEN)
if live_aqi is None:
    print("⚠️ WAQI failed — attempting to continue but results may be less reliable.")
else:
    # convert to local tz for display
    if live_ts is not None:
        if live_ts.tzinfo is None:
            # assume UTC if no tz
            live_ts = live_ts.replace(tzinfo=pytz.UTC)
        live_ts_local = live_ts.astimezone(pytz.timezone(TIMEZONE))
    else:
        live_ts_local = None

print(f"📌 Live AQI (WAQI): {live_aqi}  Time (local): {live_ts_local}")

# Cross-check: try AQI.in (via same WAQI endpoint or webpage)
def fetch_aqi_in_via_waqi(lat, lon, token):
    # attempt same endpoint; for more advanced scraping of aqi.in webpage you'd add requests + parsing
    try:
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}"
        r = requests.get(url, timeout=10)
        data = r.json()
        aqi2 = safe_get(data, "data", "aqi", default=None)
        return aqi2
    except Exception:
        return None

cross_aqi = fetch_aqi_in_via_waqi(LAT, LON, WAQI_TOKEN)
print(f"📌 Cross-check AQI (secondary source): {cross_aqi}")

# compute diff and trust
if (live_aqi is not None) and (cross_aqi is not None):
    try:
        diff = abs(float(live_aqi) - float(cross_aqi))
    except Exception:
        diff = None
else:
    diff = None

if diff is None:
    trust = "Unknown"
elif diff <= 10:
    trust = "High"
elif diff <= 20:
    trust = "Medium"
else:
    trust = "Low"
print(f"🔍 Cross-check diff = {diff}, Trust = {trust}")

# Fetch VisualCrossing hourly forecast (next FORECAST_DAYS) 
def fetch_visualcrossing(city, key, days=3):
    try:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/next{days}days?unitGroup=metric&include=hours&key={key}&contentType=json"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise SystemExit(f"❌ VisualCrossing fetch failed: {e}")

vc_json = fetch_visualcrossing(CITY, VISUAL_CROSSING_KEY, FORECAST_DAYS)
# flatten hours into a DataFrame
hours = []
for d in vc_json.get("days", []):
    for h in d.get("hours", []):
        rec = {
            "datetimeEpoch": h.get("datetimeEpoch", None),
            "datetime_str": h.get("datetime", None),  # time only (like "00:00:00")
            "temp": h.get("temp", np.nan),
            "dew": h.get("dew", np.nan),
            "humidity": h.get("humidity", np.nan),
            "precip": h.get("precip", np.nan),
            "windspeed": h.get("windspeed", np.nan),
            "winddir": h.get("winddir", np.nan),
            "visibility": h.get("visibility", np.nan),
            "so2": h.get("so2", np.nan),
            "no2": h.get("no2", np.nan),
            "o3": h.get("o3", np.nan),
            "co": h.get("co", np.nan),
            "pressure": h.get("pressure", np.nan)
        }
        hours.append(rec)
if len(hours) == 0:
    raise SystemExit("❌ No hourly forecast returned by VisualCrossing.")

df_hours = pd.DataFrame(hours)

# 5) Fix datetime: combine date (from epoch) + time into proper tz-aware datetime 
# epoch already contains full timestamp; prefer to use it:
df_hours["datetime"] = pd.to_datetime(df_hours["datetimeEpoch"], unit="s", utc=True).dt.tz_convert(TIMEZONE)
# keep only future rows (relative to now in local tz)
now_local = datetime.now(pytz.timezone(TIMEZONE))
df_hours = df_hours[df_hours["datetime"] >= now_local].reset_index(drop=True)
# limit to next 72 hours just in case
df_hours = df_hours.head(72).copy()

# Feature engineering to match training features 
feature_cols = ['temp','dew','humidity','precip','windspeed','winddir','visibility',
                'so2','no2','o3','co','hour','day_of_week','is_weekend','dow_sin','dow_cos','aqi_lag1']

# add time features
df_hours["hour"] = df_hours["datetime"].dt.hour
df_hours["day_of_week"] = df_hours["datetime"].dt.dayofweek
df_hours["is_weekend"] = df_hours["day_of_week"].isin([5,6]).astype(int)
df_hours["dow_sin"] = np.sin(2 * np.pi * df_hours["day_of_week"] / 7)
df_hours["dow_cos"] = np.cos(2 * np.pi * df_hours["day_of_week"] / 7)

# ensure pollutant columns exist (if missing, fill with NaN)
for p in ["so2","no2","o3","co"]:
    if p not in df_hours.columns:
        df_hours[p] = np.nan

# Prepare aqi_lag1: recursive approach 
# start with live_aqi for the first row
if live_aqi is None:
    print("⚠️ live_aqi is missing; filling aqi_lag1 with NaN")
    df_hours["aqi_lag1"] = np.nan
else:
    # fill aqi_lag1 iteratively: first row gets live_aqi, subsequent rows will be filled after each prediction loop
    df_hours = df_hours.reset_index(drop=True)
    df_hours["aqi_lag1"] = np.nan
    df_hours.loc[0, "aqi_lag1"] = float(live_aqi)

# Run recursive prediction using my model 
expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else feature_cols
missing = [c for c in expected_features if c not in df_hours.columns]
if missing:
    # add missing columns as NaN
    for c in missing:
        df_hours[c] = np.nan
    print(f"⚠️ Added missing feature columns with NaN: {missing}")

preds = []
for i in range(len(df_hours)):
    # select features in the exact order
    X_row = df_hours.loc[i, expected_features].values.reshape(1, -1)
    try:
        pred = model.predict(X_row)[0]
    except Exception as e:
        raise SystemExit(f"❌ Model prediction failed at row {i}: {e}")
    # clamp predictions to realistic AQI bounds
    pred = float(max(0, min(500, pred)))
    preds.append(pred)
    # set next row's aqi_lag1 (recursive)
    if i+1 < len(df_hours):
        df_hours.loc[i+1, "aqi_lag1"] = pred

df_hours["Predicted_AQI"] = preds
df_hours["AQI_Category"] = df_hours["Predicted_AQI"].apply(categorize_aqi)
df_hours["Live_AQI_used"] = float(live_aqi) if live_aqi is not None else np.nan

# Accuracy metrics comparing predicted series to live AQI 
if live_aqi is not None:
    y_true = np.array([float(live_aqi)] * len(df_hours))
    y_pred = df_hours["Predicted_AQI"].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-9))) * 100
    print(f"\n📊 Accuracy vs current live AQI={live_aqi} (coarse): MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
else:
    mae = rmse = mape = None
    print("⚠️ Skipped accuracy metrics because live AQI is missing.")

# Save CSV
save_cols = ["datetime","Predicted_AQI","AQI_Category","Live_AQI_used"] + expected_features
df_hours.to_csv(OUT_CSV, index=False, columns=[c for c in save_cols if c in df_hours.columns])
print(f"💾 Saved forecast & predictions to {OUT_CSV}")

# Visualization (PKT local time on x-axis)
plt.figure(figsize=(12,6))
# background AQI bands
bands = [(0,50,"#00e400"), (51,100,"#ffff66"), (101,150,"#ffcc66"),
         (151,200,"#ff6666"), (201,300,"#b266ff"), (301,500,"#7e0023")]
for low, high, color in bands:
    plt.fill_between(df_hours["datetime"], low, high, color=color, alpha=0.12)

plt.plot(df_hours["datetime"], df_hours["Predicted_AQI"], marker="o", label="Predicted AQI", color="royalblue")
if live_aqi is not None:
    plt.axhline(y=float(live_aqi), color="crimson", linestyle="--", label=f"Live AQI (WAQI={live_aqi})")

plt.scatter(df_hours["datetime"].iloc[0], df_hours["Live_AQI_used"].iloc[0], color="red", s=100, zorder=5, label="Current AQI (seed)")

plt.title(f"{CITY} — Predicted AQI (next {len(df_hours)} hrs)  | Trust: {trust} | Cross-diff: {diff}")
plt.xlabel("Local time (PKT)")
plt.ylabel("AQI")
plt.xticks(rotation=30)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
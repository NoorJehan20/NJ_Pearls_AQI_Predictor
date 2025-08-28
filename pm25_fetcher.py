import requests
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import pytz
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# CONFIG 
WAQI_TOKEN = os.getenv("WAQI_TOKEN")
VISUAL_CROSSING_KEY = os.getenv("VISUAL_CROSSING_KEY")

if not WAQI_TOKEN or not VISUAL_CROSSING_KEY:
    raise ValueError("Missing API keys from environment variables")
    
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
    try:
        for k in keys:
            d = d[k]
        return d
    except Exception:
        return default

# LOAD MODEL
try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    raise SystemExit(f"Failed to load model at {MODEL_PATH}: {e}")

# FETCH LIVE AQI 
def fetch_waqi(lat, lon, token):
    try:
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok":
            print("WAQI returned non-ok status:", data.get("data"))
            return None, None
        aqi = safe_get(data, "data", "aqi", default=None)
        time_iso = safe_get(data, "data", "time", "iso", default=None)
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
        print("Error fetching WAQI:", e)
        return None, None

live_aqi, live_ts = fetch_waqi(LAT, LON, WAQI_TOKEN)
if live_aqi is None:
    print("WAQI failed — predictions will use fallback/NaN lag values")
if live_ts is not None:
    if live_ts.tzinfo is None:
        live_ts = live_ts.replace(tzinfo=pytz.UTC)
    live_ts_local = live_ts.astimezone(pytz.timezone(TIMEZONE))
else:
    live_ts_local = None

print(f"Live AQI (WAQI): {live_aqi}  Time (local): {live_ts_local}")

# FETCH VISUAL CROSSING HOURLY FORECAST
def fetch_visualcrossing(city, key, days=3):
    try:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/next{days}days?unitGroup=metric&include=hours&key={key}&contentType=json"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise SystemExit(f"VisualCrossing fetch failed: {e}")

vc_json = fetch_visualcrossing(CITY, VISUAL_CROSSING_KEY, FORECAST_DAYS)

hours = []
for day in vc_json.get("days", []):
    for h in day.get("hours", []):
        rec = {
            "datetimeEpoch": h.get("datetimeEpoch", None),
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
            "co": h.get("co", np.nan)
        }
        hours.append(rec)

if not hours:
    raise SystemExit("❌ No hourly forecast returned by VisualCrossing.")

df_hours = pd.DataFrame(hours)
df_hours["datetime"] = pd.to_datetime(df_hours["datetimeEpoch"], unit="s", utc=True).dt.tz_convert(TIMEZONE)

now_local = datetime.now(pytz.timezone(TIMEZONE))
df_hours = df_hours[df_hours["datetime"] >= now_local].reset_index(drop=True)
df_hours = df_hours.head(72).copy()  

# FEATURE ENGINEERING
df_hours["hour"] = df_hours["datetime"].dt.hour
df_hours["day_of_week"] = df_hours["datetime"].dt.dayofweek
df_hours["is_weekend"] = df_hours["day_of_week"].isin([5,6]).astype(int)
df_hours["dow_sin"] = np.sin(2 * np.pi * df_hours["day_of_week"] / 7)
df_hours["dow_cos"] = np.cos(2 * np.pi * df_hours["day_of_week"] / 7)

# Ensure pollutant columns present
for pollutant in ["so2", "no2", "o3", "co"]:
    if pollutant not in df_hours.columns:
        df_hours[pollutant] = np.nan

# LAG FEATURE PREP 
if live_aqi is None:
    print("live_aqi missing; filling lag features with NaN")
    df_hours["aqi_lag1"] = np.nan
else:
    df_hours["aqi_lag1"] = np.nan
    df_hours.loc[0, "aqi_lag1"] = float(live_aqi)

# PREDICTION LOOP 
feature_cols = ['temp','dew','humidity','precip','windspeed','winddir','visibility',
                'so2','no2','o3','co','hour','day_of_week','is_weekend','dow_sin','dow_cos','aqi_lag1']

expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else feature_cols

# Add missing columns if any
for col in expected_features:
    if col not in df_hours.columns:
        df_hours[col] = np.nan
        print(f"Missing feature '{col}' added with NaN")

predictions = []
for i in range(len(df_hours)):
    row_features = df_hours.loc[i, expected_features].values.reshape(1, -1)
    try:
        pred = model.predict(row_features)[0]
    except Exception as e:
        raise SystemExit(f"❌ Model prediction failed at row {i}: {e}")
    pred = float(np.clip(pred, 0, 500))
    predictions.append(pred)
    if i + 1 < len(df_hours):
        df_hours.loc[i+1, "aqi_lag1"] = pred

df_hours["Predicted_AQI"] = predictions
df_hours["AQI_Category"] = df_hours["Predicted_AQI"].apply(categorize_aqi)
df_hours["Live_AQI_used"] = float(live_aqi) if live_aqi is not None else np.nan

# ACCURACY METRICS 
if live_aqi is not None:
    y_true = np.array([float(live_aqi)] * len(df_hours))
    y_pred = df_hours["Predicted_AQI"].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-9))) * 100
    print(f"\nAccuracy vs live AQI={live_aqi} (coarse): MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
else:
    mae = rmse = mape = None
    print("Skipped accuracy metrics due to missing live AQI")

# SAVE HOURLY CSV SAFELY 
os.makedirs("outputs/hourly", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

timestamp = datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y-%m-%d_%H")

hourly_file = f"outputs/hourly/aqi_forecast_{timestamp}.csv"
daily_realtime_file = "realtime_aqi_forecast.csv"

save_cols = ["datetime","Predicted_AQI","AQI_Category","Live_AQI_used"] + expected_features

# Save timestamped hourly file (for merging later)
df_hours.to_csv(hourly_file, index=False, columns=[c for c in save_cols if c in df_hours.columns])
print(f"💾 Saved hourly forecast to {hourly_file}")

# Also save/update a "latest realtime" CSV (like before, so you can always see most recent data)
df_hours.to_csv(daily_realtime_file, index=False, columns=[c for c in save_cols if c in df_hours.columns])
print(f"💾 Saved latest realtime forecast to {daily_realtime_file}")

# PLOT AND SAVE 
plt.figure(figsize=(12,6))
plt.plot(df_hours["datetime"], df_hours["Predicted_AQI"], marker="o", label="Predicted AQI", color="royalblue")

if live_aqi is not None:
    plt.axhline(y=float(live_aqi), color="crimson", linestyle="--", label=f"Live AQI (WAQI={live_aqi})")

plt.title(f"{CITY} — Predicted AQI (next {len(df_hours)} hrs)")
plt.xlabel("Local time (PKT)")
plt.ylabel("AQI")
plt.xticks(rotation=30)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plot_file = f"outputs/aqi_forecast_plot_{timestamp}.png"
plt.savefig(plot_file)
plt.close()
print(f"💾 Saved AQI forecast plot as {plot_file}")

plt.close()
print("💾 Saved AQI forecast plot as aqi_forecast_plot.png")
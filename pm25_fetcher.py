import os
import sys
import requests
import pandas as pd
from datetime import datetime
import pytz
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def categorize_aqi(aqi):
    aqi = float(aqi)
    if aqi <= 50: return "Good 😀"
    if aqi <= 100: return "Moderate 😐"
    if aqi <= 150: return "Unhealthy for Sensitive Groups 😷"
    if aqi <= 200: return "Unhealthy 😷"
    if aqi <= 300: return "Very Unhealthy ☠"
    return "Hazardous 💀"

def main():
    try:
        # Load API keys from environment variables
        waqi_key = os.getenv("WAQI_TOKEN")
        vc_key = os.getenv("VISUAL_CROSSING_KEY")
        print("DEBUG: WAQI_TOKEN =", waqi_key)
        print("DEBUG: VISUAL_CROSSING_KEY =", vc_key)

        if not waqi_key or not vc_key:
            raise ValueError("Missing WAQI_TOKEN or VISUAL_CROSSING_KEY in environment variables")

        # Config
        CITY = "Karachi"
        TIMEZONE = "Asia/Karachi"
        MODEL_PATH = "best_model_xgboost_new.pkl"
        OUTPUT_CSV = "realtime_aqi_forecast.csv"

        # Load trained model
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Loaded model from {MODEL_PATH}")

        # Fetch live AQI from WAQI
        waqi_url = f"https://api.waqi.info/feed/{CITY}/?token={waqi_key}"
        waqi_resp = requests.get(waqi_url)
        waqi_resp.raise_for_status()
        waqi_data = waqi_resp.json()
        if waqi_data.get("status") != "ok":
            raise ValueError(f"WAQI API returned error status: {waqi_data}")
        live_aqi = waqi_data.get("data", {}).get("aqi")
        if live_aqi is None:
            raise ValueError("Live AQI data missing from WAQI response")
        local_time = datetime.now(pytz.timezone(TIMEZONE))
        print(f"📌 Live AQI (WAQI): {live_aqi}  Time (local): {local_time}")

        # Fetch Visual Crossing hourly data for today (needed for features)
        vc_url = (
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
            f"{CITY}/today?unitGroup=metric&include=hours&key={vc_key}&contentType=json"
        )
        vc_resp = requests.get(vc_url)
        vc_resp.raise_for_status()
        vc_data = vc_resp.json()
        days = vc_data.get("days", [])
        if not days:
            raise ValueError("Visual Crossing response missing 'days' data")
        hourly_data = days[0].get("hours", [])
        if not hourly_data:
            raise ValueError("No hourly data from Visual Crossing")

        # Build DataFrame from hourly data
        df_hours = pd.DataFrame(hourly_data)

        # Fix datetime (using datetimeEpoch field)
        df_hours["datetime"] = pd.to_datetime(df_hours["datetimeEpoch"], unit="s", utc=True).dt.tz_convert(TIMEZONE)

        # Only keep current and future hours
        df_hours = df_hours[df_hours["datetime"] >= local_time].reset_index(drop=True)

        # Add time-based features
        df_hours["hour"] = df_hours["datetime"].dt.hour
        df_hours["day_of_week"] = df_hours["datetime"].dt.dayofweek
        df_hours["is_weekend"] = df_hours["day_of_week"].isin([5,6]).astype(int)
        df_hours["dow_sin"] = np.sin(2 * np.pi * df_hours["day_of_week"] / 7)
        df_hours["dow_cos"] = np.cos(2 * np.pi * df_hours["day_of_week"] / 7)

        # Ensure pollutant columns exist, fill missing with NaN
        for p in ["so2", "no2", "o3", "co"]:
            if p not in df_hours.columns:
                df_hours[p] = np.nan

        # Add lag feature aqi_lag1 (recursive prediction seed)
        df_hours["aqi_lag1"] = np.nan
        df_hours.loc[0, "aqi_lag1"] = float(live_aqi)

        # Feature columns in correct order expected by model
        feature_cols = [
            'temp', 'dew', 'humidity', 'precip', 'windspeed', 'winddir', 'visibility',
            'so2', 'no2', 'o3', 'co', 'hour', 'day_of_week', 'is_weekend', 'dow_sin', 'dow_cos', 'aqi_lag1'
        ]

        # Fill any missing feature columns with NaN
        for col in feature_cols:
            if col not in df_hours.columns:
                df_hours[col] = np.nan

        # Recursive prediction loop
        preds = []
        for i in range(len(df_hours)):
            X_row = df_hours.loc[i, feature_cols].values.reshape(1, -1)
            pred = model.predict(X_row)[0]
            pred = max(0, min(500, pred))  # clamp to valid AQI range
            preds.append(pred)
            if i + 1 < len(df_hours):
                df_hours.loc[i + 1, "aqi_lag1"] = pred

        df_hours["Predicted_AQI"] = preds

        # Accuracy metrics vs live AQI (coarse)
        y_true = np.array([float(live_aqi)] * len(df_hours))
        y_pred = df_hours["Predicted_AQI"].values
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mape = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-9))) * 100

        print(f"📊 Accuracy vs current live AQI={live_aqi} (coarse): MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

        # Save output
        out_cols = ["datetime", "Predicted_AQI"] + feature_cols
        df_hours.to_csv(OUTPUT_CSV, index=False, columns=out_cols)
        print(f"💾 Saved forecast & predictions to {OUTPUT_CSV}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

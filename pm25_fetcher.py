import os
import sys
import requests
import pandas as pd
from datetime import datetime
import pytz
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
        waqi_data = requests.get(waqi_url).json()

        if waqi_data["status"] != "ok":
            raise ValueError(f"WAQI API Error: {waqi_data}")

        live_aqi = waqi_data["data"]["aqi"]
        local_time = datetime.now(pytz.timezone(TIMEZONE))
        print(f"📌 Live AQI (WAQI): {live_aqi}  Time (local): {local_time}")

        # Cross-check AQI from Visual Crossing
        vc_url = (
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
            f"{CITY}/today?unitGroup=metric&include=hours&key={vc_key}&contentType=json"
        )
        vc_data = requests.get(vc_url).json()

        hourly_data = vc_data.get("days", [])[0].get("hours", [])
        current_hour = local_time.hour
        cross_aqi = None

        for h in hourly_data:
            if h["datetime"][:2] == str(current_hour).zfill(2):
                cross_aqi = h.get("pm2p5") or h.get("aqi") or None
                break

        if cross_aqi is None:
            cross_aqi = np.nan

        diff = abs(live_aqi - cross_aqi) if not np.isnan(cross_aqi) else np.nan
        trust_level = "High" if diff <= 10 else "Medium" if diff <= 25 else "Low"

        print(f"📌 Cross-check AQI (secondary source): {cross_aqi}")
        print(f"🔍 Cross-check diff = {diff}, Trust = {trust_level}")

        # Build features for prediction
        df = pd.DataFrame({
            "live_aqi": [live_aqi],
            "hour": [local_time.hour],
            "dayofweek": [local_time.weekday()],
        })

        X = df.values

        # Make prediction
        pred_aqi = model.predict(X)[0]

        # Evaluate vs live AQI (rough check)
        mae = mean_absolute_error([live_aqi], [pred_aqi])
        rmse = mean_squared_error([live_aqi], [pred_aqi], squared=False)
        mape = np.mean(np.abs((np.array([live_aqi]) - np.array([pred_aqi])) / np.array([live_aqi]))) * 100

        print(f"📊 Accuracy vs current live AQI={live_aqi} (coarse): MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

        # Save predictions
        out_df = pd.DataFrame([{
            "timestamp": local_time.isoformat(),
            "live_aqi": live_aqi,
            "cross_aqi": cross_aqi,
            "predicted_aqi": pred_aqi,
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        }])

        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"💾 Saved forecast & predictions to {OUTPUT_CSV}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

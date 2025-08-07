# ✅ FILE 1: pm25_fetcher.py
import os
import pandas as pd
import requests
from datetime import datetime

# Constants
CITY = "Karachi"
WAQI_TOKEN = os.environ.get("WAQI_TOKEN")  
OUTPUT_FILE = "waqi_karachi_pm25_hourly.csv"

# Function to fetch PM2.5 data from WAQI
def fetch_pm25(city, token):
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}")
    data = response.json()
    pm25 = data.get('data', {}).get('iaqi', {}).get('pm25', {}).get('v')
    timestamp = data.get('data', {}).get('time', {}).get('s')
    if pm25 is None or timestamp is None:
        raise Exception("Missing PM2.5 or timestamp in response")
    return {"timestamp": timestamp, "city": city, "pm25": pm25}

# Fetch data
try:
    record = fetch_pm25(CITY, WAQI_TOKEN)
    df_new = pd.DataFrame([record])
    print("✅ Fetched:", df_new)
except Exception as e:
    df_new = pd.DataFrame()
    print("❌ Error fetching data:", e)

# Append to CSV
if not df_new.empty:
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(OUTPUT_FILE, index=False)
    print("📦 Saved to:", OUTPUT_FILE)

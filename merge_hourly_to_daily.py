import pandas as pd
import glob
import os
from datetime import datetime

# Folder where hourly CSVs are stored
OUTPUT_DIR = "outputs"

# Today's date string
today_str = datetime.utcnow().strftime("%Y%m%d")

# Find all hourly CSV files from today
hourly_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"realtime_aqi_forecast_{today_str}_*.csv")))

if not hourly_files:
    print(f"No hourly files found for today ({today_str}). Nothing to merge.")
    exit(0)

# Read and combine
df_list = []
for file in hourly_files:
    try:
        df_list.append(pd.read_csv(file))
    except Exception as e:
        print(f"Error reading {file}: {e}")

daily_df = pd.concat(df_list, ignore_index=True)

# Save daily CSV
daily_file = os.path.join(OUTPUT_DIR, f"daily_aqi_{today_str}.csv")
daily_df.to_csv(daily_file, index=False)
print(f"Daily file saved: {daily_file}")

# Optional: remove hourly files after merging
for file in hourly_files:
    os.remove(file)
print("Hourly files removed after merging.")

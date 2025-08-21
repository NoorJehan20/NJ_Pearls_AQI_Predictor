import os
import glob
import pandas as pd
from datetime import datetime

# Folders
HOURLY_FOLDER = "outputs/hourly"
DAILY_FOLDER = "outputs"

os.makedirs(HOURLY_FOLDER, exist_ok=True)
os.makedirs(DAILY_FOLDER, exist_ok=True)

# Merge all hourly files for today
hourly_files = glob.glob(os.path.join(HOURLY_FOLDER, "*.csv"))
if not hourly_files:
    print("No hourly files found. Exiting.")
    exit()

df_list = []
for file in hourly_files:
    try:
        df = pd.read_csv(file)
        df_list.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not df_list:
    print("No valid hourly CSVs found.")
    exit()

merged_df = pd.concat(df_list, ignore_index=True)

# Ensure datetime column exists
if "datetime" not in merged_df.columns:
    raise ValueError("No 'datetime' column found in hourly CSVs.")

# Remove duplicates (by datetime)
merged_df.drop_duplicates(subset=["datetime"], inplace=True)

# Sort by datetime
merged_df["datetime"] = pd.to_datetime(merged_df["datetime"])
merged_df.sort_values(by="datetime", inplace=True)

# Create today's filename
today_str = datetime.utcnow().strftime("%Y-%m-%d")
daily_file = os.path.join(DAILY_FOLDER, f"daily_aqi_{today_str}.csv")

# If file already exists, append only new rows
if os.path.exists(daily_file):
    existing_df = pd.read_csv(daily_file)
    existing_df["datetime"] = pd.to_datetime(existing_df["datetime"])
    combined_df = pd.concat([existing_df, merged_df], ignore_index=True)
    combined_df.drop_duplicates(subset=["datetime"], inplace=True)
    combined_df.sort_values(by="datetime", inplace=True)
    combined_df.to_csv(daily_file, index=False)
    print(f"✅ Updated existing file: {daily_file}")
else:
    merged_df.to_csv(daily_file, index=False)
    print(f"✅ Created new file: {daily_file}")

print("Merge complete. No data was overwritten.")

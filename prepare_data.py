import pandas as pd
import geopandas as gpd
import requests
import zipfile
import os
import shutil
from io import BytesIO
import numpy as np
import time

# --- Configuration ---
YEARS = range(2020, 2025)  # Includes 2020, 2021, 2022, 2023, 2024
MONTHS = range(1, 13)      # All months from Jan to Dec
SAMPLES_PER_FILE = 2000     # Smaller sample per monthly file
TOTAL_SAMPLE_SIZE = 100000  # Target total sample size (~10-15MB)
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
OUTPUT_DATA = "data/trips_sample.parquet"
OUTPUT_ZONES = "data/taxi_zones.geojson"

# --- 1. Download & Sample Trip Data from Multiple Years ---
print("Downloading and sampling multi-year trip data...")
os.makedirs("data", exist_ok=True)

sampled_dfs = []
total_downloaded = 0

for year in YEARS:
    for month in MONTHS:
        # Format month to have two digits (e.g., '01', '02')
        month_str = f"{month:02d}"
        file_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month_str}.parquet"
        
        print(f"Processing {year}-{month_str}...")
        try:
            # Download the monthly file
            df_month = pd.read_parquet(file_url)
            
            # Basic cleaning
            df_month = df_month.dropna(subset=['PULocationID', 'DOLocationID', 'fare_amount', 'trip_distance'])
            df_month = df_month[(df_month['fare_amount'] > 0) & (df_month['trip_distance'] > 0)]
            
            # Take a sample from this month
            sample_size = min(SAMPLES_PER_FILE, len(df_month))
            if sample_size > 0:
                df_sample = df_month.sample(n=sample_size, random_state=42)
                sampled_dfs.append(df_sample)
                total_downloaded += len(df_sample)
                print(f"  -> Sampled {len(df_sample)} trips (Total so far: {total_downloaded})")
            
            # If we've reached our target total, stop downloading
            if total_downloaded >= TOTAL_SAMPLE_SIZE:
                print(f"Reached target sample size ({TOTAL_SAMPLE_SIZE}). Stopping download.")
                break
                
        except Exception as e:
            print(f"  -> Error: {e}. Skipping this file.")
            continue
        
        time.sleep(0.5)  # Be polite to the server
    if total_downloaded >= TOTAL_SAMPLE_SIZE:
        break

if not sampled_dfs:
    raise Exception("No data was successfully downloaded. Please check the URL patterns and your internet connection.")

# Combine all the monthly samples
df_combined = pd.concat(sampled_dfs, ignore_index=True)
print(f"\nTotal trips collected: {len(df_combined)}")

# --- 2. Feature Engineering (same as before) ---
df_combined['tpep_pickup_datetime'] = pd.to_datetime(df_combined['tpep_pickup_datetime'])
df_combined['pickup_hour'] = df_combined['tpep_pickup_datetime'].dt.hour
df_combined['pickup_dayofweek'] = df_combined['tpep_pickup_datetime'].dt.dayofweek
df_combined['pickup_weekday'] = (df_combined['pickup_dayofweek'] < 5).astype(int)
df_combined['pickup_date'] = df_combined['tpep_pickup_datetime'].dt.date
df_combined['tip_percentage'] = np.where(
    df_combined['fare_amount'] > 0,
    (df_combined['tip_amount'] / df_combined['fare_amount']) * 100,
    0
)

# Save the final sampled data
df_combined.to_parquet(OUTPUT_DATA)
print(f"Saved {len(df_combined)} trips to {OUTPUT_DATA}")

# --- 3. Download & Process Taxi Zones (unchanged) ---
print("Downloading taxi zone shapefile...")
response = requests.get(ZONES_URL)
with zipfile.ZipFile(BytesIO(response.content)) as z:
    z.extractall("data/temp_zones")

shp_file = None
for root, dirs, files in os.walk("data/temp_zones"):
    for file in files:
        if file.endswith(".shp"):
            shp_file = os.path.join(root, file)
            break
    if shp_file:
        break

if shp_file is None:
    raise FileNotFoundError("No shapefile found in the downloaded zip")

print(f"Found shapefile at: {shp_file}")
gdf = gpd.read_file(shp_file)
gdf = gdf.to_crs("EPSG:4326")
gdf.to_file(OUTPUT_ZONES, driver='GeoJSON')
print(f"Saved zone map to {OUTPUT_ZONES}")

shutil.rmtree("data/temp_zones")
print("Data preparation complete!")
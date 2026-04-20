import pandas as pd
import geopandas as gpd

# Check trips data
try:
    trips = pd.read_parquet('data/trips_sample.parquet')
    print(f"✅ Trips data loaded: {len(trips)} rows")
    print(f"Columns: {trips.columns.tolist()}")
    print(f"Date range: {trips['tpep_pickup_datetime'].min()} to {trips['tpep_pickup_datetime'].max()}")
    print(f"Sample trip:\n{trips.head(1)}")
except Exception as e:
    print(f"❌ Error loading trips: {e}")

# Check zones data
try:
    zones = gpd.read_file('data/taxi_zones.geojson')
    print(f"✅ Zones loaded: {len(zones)} zones")
    print(f"Columns: {zones.columns.tolist()}")
except Exception as e:
    print(f"❌ Error loading zones: {e}")
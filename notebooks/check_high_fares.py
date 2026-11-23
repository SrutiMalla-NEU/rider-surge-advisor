import pandas as pd
from src.clean import COLS

raw = pd.read_parquet("data/raw/nyc/tlc/hvfhv/fhvhv_tripdata_2025-10.parquet", columns=COLS)
raw = raw[raw["trip_miles"] >= 1]
raw["fare_per_mile"] = raw["base_passenger_fare"] / raw["trip_miles"]

high = raw[raw["fare_per_mile"] > 50]
print(f"Trips above $50 per mile: {len(high):,}")

print("\nBy day:")
print(high["request_datetime"].dt.date.value_counts().sort_index())

print("\nBy hour:")
print(high["request_datetime"].dt.hour.value_counts().sort_index())

print("\nSome examples:")
print(high[["request_datetime", "trip_miles", "base_passenger_fare", "fare_per_mile"]]
      .sort_values("fare_per_mile", ascending=False).head(10))
import pandas as pd
from src.clean import COLS, clean_trips

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

raw = pd.read_parquet("data/raw/nyc/tlc/hvfhv/fhvhv_tripdata_2025-10.parquet", columns=COLS)
print(f"Rows: {len(raw):,}")

print("\n1. Missing values per column:")
print(raw.isna().sum())

print(f"\n2. Exact duplicate rows: {raw.duplicated().sum():,}")

print("\n3. Apps and flags:")
for col in ["hvfhs_license_num", "shared_request_flag", "access_a_ride_flag"]:
    print(raw[col].value_counts(dropna=False), "\n")

print("4. Number ranges:")
wait = (raw["pickup_datetime"] - raw["request_datetime"]).dt.total_seconds() / 60
ranges = pd.DataFrame({
    "trip_miles": raw["trip_miles"],
    "fare": raw["base_passenger_fare"],
    "wait_min": wait,
})
print(ranges.describe(percentiles=[0.01, 0.5, 0.99]).round(2))

clean = clean_trips(raw)
print(f"\n5. Rows kept after cleaning: {len(clean):,} of {len(raw):,} ({len(clean) / len(raw):.1%})")
print("\nFare per mile after cleaning:")
print(clean["fare_per_mile"].describe(percentiles=[0.01, 0.5, 0.99, 0.999]).round(2))
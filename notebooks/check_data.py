import pandas as pd

# Read only a few columns to save memory
cols = ["request_datetime", "pickup_datetime", "PULocationID",
        "trip_miles", "base_passenger_fare"]

df = pd.read_parquet(
    "data/raw/nyc/tlc/hvfhv/fhvhv_tripdata_2025-10.parquet",
    columns=cols,
)

print("Rows, columns:", df.shape)
print(df.head())

zones = pd.read_csv("data/raw/nyc/tlc/zones/taxi_zone_lookup.csv")
print(zones.head())
print("Number of zones:", len(zones))
print(zones["service_zone"].value_counts())
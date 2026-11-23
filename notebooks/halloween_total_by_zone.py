import pandas as pd
from src.clean import COLS, load_windows, clean_trips

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

window = ("2025-10-31 21:00", "2025-11-01 03:00")   # the surge peak
months = ["2025-10", "2025-11"]
FEES = ["tolls", "bcf", "sales_tax", "congestion_surcharge", "airport_fee", "cbd_congestion_fee"]

# Uber and Lyft: total price before tip, for 2-4 mile trips
hv_files = [f"data/raw/nyc/tlc/hvfhv/fhvhv_tripdata_{m}.parquet" for m in months]
rides = clean_trips(load_windows(hv_files, [window], columns=COLS + FEES))
rides["total"] = rides["base_passenger_fare"] + rides[FEES].fillna(0).sum(axis=1)
rides = rides[rides["trip_miles"].between(2, 4)]
rides_zone = rides.pivot_table(index="PULocationID", columns="app", values="total", aggfunc="median")
rides_zone["rideshare_trips"] = rides.groupby("PULocationID").size()

# Yellow cabs: total price before tip, for 2-4 mile trips
parts = []
for m in months:
    t = pd.read_parquet(f"data/raw/nyc/tlc/yellow/yellow_tripdata_{m}.parquet",
                        columns=["tpep_pickup_datetime", "PULocationID", "trip_distance",
                                 "fare_amount", "total_amount", "tip_amount"])
    parts.append(t[(t["tpep_pickup_datetime"] >= window[0]) & (t["tpep_pickup_datetime"] < window[1])])
yellow = pd.concat(parts, ignore_index=True)
yellow = yellow[yellow["trip_distance"].between(2, 4) & (yellow["fare_amount"] > 0)]
yellow["total"] = yellow["total_amount"] - yellow["tip_amount"]
yellow_zone = yellow.groupby("PULocationID").agg(
    Yellow=("total", "median"), yellow_trips=("total", "size"))

# Combine, add zone names, show the 15 zones with the most cab pickups
zones = pd.read_csv("data/raw/nyc/tlc/zones/taxi_zone_lookup.csv").set_index("LocationID")
result = yellow_zone.join(rides_zone).join(zones[["Zone", "Borough"]])
result = result.sort_values("yellow_trips", ascending=False).head(15).round(2)
print(result[["Zone", "Borough", "yellow_trips", "rideshare_trips", "Yellow", "Lyft", "Uber"]]
      .to_string(index=False))
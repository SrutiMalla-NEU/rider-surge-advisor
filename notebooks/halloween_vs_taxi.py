import pandas as pd
from src.clean import load_windows, clean_trips
from src.display import hour_label

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

night = ("2025-10-31 18:00", "2025-11-01 06:00")
night_order = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
months = ["2025-10", "2025-11"]

# Uber and Lyft: median fare per mile by hour
hv_files = [f"data/raw/nyc/tlc/hvfhv/fhvhv_tripdata_{m}.parquet" for m in months]
rides = clean_trips(load_windows(hv_files, [night]))
rides_hourly = (rides.groupby([rides["request_datetime"].dt.hour, "app"])["fare_per_mile"]
                .median().unstack("app"))

# Taxis: load, filter to the night, clean, add fare per mile
def load_taxi(kind, time_col):
    parts = []
    for m in months:
        t = pd.read_parquet(f"data/raw/nyc/tlc/{kind}/{kind}_tripdata_{m}.parquet",
                            columns=[time_col, "trip_distance", "fare_amount"])
        parts.append(t[(t[time_col] >= night[0]) & (t[time_col] < night[1])])
    t = pd.concat(parts, ignore_index=True)
    t = t[(t["trip_distance"] >= 1) & (t["trip_distance"] <= 100) & (t["fare_amount"] > 0)]
    t["fare_per_mile"] = t["fare_amount"] / t["trip_distance"]
    return t.groupby(t[time_col].dt.hour).agg(
        **{kind: ("fare_per_mile", "median"), f"{kind}_trips": ("fare_per_mile", "size")})

yellow = load_taxi("yellow", "tpep_pickup_datetime")
green = load_taxi("green", "lpep_pickup_datetime")

result = rides_hourly.join(yellow).join(green).reindex(night_order).round(2)
result.index = [hour_label(h) for h in result.index]
print(result)
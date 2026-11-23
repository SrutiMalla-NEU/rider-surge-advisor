import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

cols = ["request_datetime", "pickup_datetime", "PULocationID",
        "trip_miles", "base_passenger_fare"]
files = [
    "data/raw/nyc/tlc/hvfhv/fhvhv_tripdata_2025-10.parquet",
    "data/raw/nyc/tlc/hvfhv/fhvhv_tripdata_2025-11.parquet",
]

# The two nights we want: 6 p.m. to 6 a.m. the next morning
nights = {
    "oct24": ("2025-10-24 18:00", "2025-10-25 06:00"),
    "hallow": ("2025-10-31 18:00", "2025-11-01 06:00"),
}

# 1. Load each file, keep only trips from those two nights, then combine
parts = []
for f in files:
    month = pd.read_parquet(f, columns=cols)
    keep = pd.Series(False, index=month.index)
    for start, end in nights.values():
        keep |= (month["request_datetime"] >= start) & (month["request_datetime"] < end)
    parts.append(month[keep])
    del month  # free the memory before loading the next file
df = pd.concat(parts, ignore_index=True)

# 2. Same features and cleaning as before
df["wait_min"] = (df["pickup_datetime"] - df["request_datetime"]).dt.total_seconds() / 60
df["fare_per_mile"] = df["base_passenger_fare"] / df["trip_miles"]
df = df[(df["trip_miles"] >= 1) & (df["base_passenger_fare"] > 0)
        & (df["wait_min"] >= 0) & (df["wait_min"] <= 60)]

# 3. Summarize one night by hour, in night order (6 p.m. ... 5 a.m.)
night_order = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]

def hourly(start, end):
    night = df[(df["request_datetime"] >= start) & (df["request_datetime"] < end)]
    summary = night.groupby(night["request_datetime"].dt.hour).agg(
        trips=("fare_per_mile", "size"),
        wait=("wait_min", "median"),
        fare_mile=("fare_per_mile", "median"),
    ).round(2)
    return summary.reindex(night_order)

comparison = (
    hourly(*nights["oct24"]).add_suffix("_oct24")
    .join(hourly(*nights["hallow"]).add_suffix("_hallow"))
)
print(comparison)
import pandas as pd

pd.set_option("display.width", 200)  # so the table prints on one line
pd.set_option("display.max_columns", None)  # show every column

cols = ["request_datetime", "pickup_datetime", "PULocationID",
        "trip_miles", "base_passenger_fare"]
df = pd.read_parquet(
    "data/raw/nyc/tlc/hvfhv/fhvhv_tripdata_2025-10.parquet",
    columns=cols,
)

# 1. Create two new columns
df["wait_min"] = (df["pickup_datetime"] - df["request_datetime"]).dt.total_seconds() / 60
df["fare_per_mile"] = df["base_passenger_fare"] / df["trip_miles"]

# 2. Basic cleaning: drop very short trips and impossible values
df = df[
    (df["trip_miles"] >= 1)
    & (df["base_passenger_fare"] > 0)
    & (df["wait_min"] >= 0)
    & (df["wait_min"] <= 60)
]

# 3. Keep only trips requested on each Friday
def one_day(date_str):
    start = pd.Timestamp(date_str)
    end = start + pd.Timedelta(days=1)
    return df[(df["request_datetime"] >= start) & (df["request_datetime"] < end)]

normal = one_day("2025-10-24")
halloween = one_day("2025-10-31")

# 4. Summarize each day by hour
def hourly(day):
    return day.groupby(day["request_datetime"].dt.hour).agg(
        trips=("fare_per_mile", "size"),
        wait=("wait_min", "median"),
        fare_mile=("fare_per_mile", "median"),
    ).round(2)

comparison = hourly(normal).add_suffix("_oct24").join(hourly(halloween).add_suffix("_hallow"))
print(comparison)
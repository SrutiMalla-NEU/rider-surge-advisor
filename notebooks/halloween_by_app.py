import pandas as pd
from src.clean import load_windows, clean_trips
from src.display import hour_label          # CHANGE 1: import the helper

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

files = ["data/raw/nyc/tlc/hvfhv/fhvhv_tripdata_2025-10.parquet",
         "data/raw/nyc/tlc/hvfhv/fhvhv_tripdata_2025-11.parquet"]
nights = {"oct24": ("2025-10-24 18:00", "2025-10-25 06:00"),
          "hallow": ("2025-10-31 18:00", "2025-11-01 06:00")}
night_order = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]

df = clean_trips(load_windows(files, nights.values()))

def by_app(start, end, label):
    night = df[(df["request_datetime"] >= start) & (df["request_datetime"] < end)]
    table = (night.groupby([night["request_datetime"].dt.hour, "app"])["fare_per_mile"]
             .median().unstack("app").round(2))
    return table.reindex(night_order).add_suffix(f"_{label}")

result = by_app(*nights["oct24"], "oct24").join(by_app(*nights["hallow"], "hallow"))
result.index = [hour_label(h) for h in result.index]   # CHANGE 2: relabel the hours
print(result)
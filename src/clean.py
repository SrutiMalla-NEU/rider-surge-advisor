import pandas as pd

# Columns we need from the HVFHV files
COLS = ["hvfhs_license_num", "request_datetime", "pickup_datetime",
        "PULocationID", "trip_miles", "base_passenger_fare",
        "shared_request_flag", "access_a_ride_flag"]

# License numbers from the data dictionary
APPS = {"HV0003": "Uber", "HV0005": "Lyft"}


def load_windows(paths, windows, columns=COLS):
    """Load trips from each file, keeping only requests inside the time windows."""
    parts = []
    for path in paths:
        month = pd.read_parquet(path, columns=columns)
        keep = pd.Series(False, index=month.index)
        for start, end in windows:
            keep |= (month["request_datetime"] >= start) & (month["request_datetime"] < end)
        parts.append(month[keep])
        del month
    return pd.concat(parts, ignore_index=True)


def clean_trips(df):
    """Apply the project's cleaning rules and add feature columns."""
    df = df.drop_duplicates()
    df = df[df["hvfhs_license_num"].isin(APPS)].copy()
    df = df[(df["shared_request_flag"] != "Y") & (df["access_a_ride_flag"] != "Y")]
    df = df[~df["PULocationID"].isin([264, 265])]   # unknown / outside NYC

    df["app"] = df["hvfhs_license_num"].map(APPS)
    df["wait_min"] = (df["pickup_datetime"] - df["request_datetime"]).dt.total_seconds() / 60
    df["fare_per_mile"] = df["base_passenger_fare"] / df["trip_miles"]
    df["fare_per_mile"] = df["fare_per_mile"].clip(upper=50)   # cap, don't remove

    return df[(df["trip_miles"] >= 1) & (df["trip_miles"] <= 100)
              & (df["base_passenger_fare"] > 0)
              & (df["wait_min"] >= 0) & (df["wait_min"] <= 60)
              & (df["fare_per_mile"] >= 1)]
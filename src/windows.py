"""Summarize clean trips into one row per pickup zone per time window."""
from pathlib import Path

import pandas as pd
from src.clean import COLS, clean_trips

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def make_windows(trips, freq="15min"):
    """Group trips by pickup zone and time window, and summarize each group."""
    trips = trips.assign(window_start=trips["request_datetime"].dt.floor(freq))
    return (trips.groupby(["PULocationID", "window_start"])
            .agg(trips=("fare_per_mile", "size"),
                 fare_per_mile=("fare_per_mile", "median"),
                 wait_min=("wait_min", "median"))
            .reset_index())


def build_month(month):
    """Load one month, clean it, summarize it into windows, and save the result."""
    raw = pd.read_parquet(f"data/raw/nyc/tlc/hvfhv/fhvhv_tripdata_{month}.parquet", columns=COLS)
    clean = clean_trips(raw)
    windows = make_windows(clean)

    out = Path("data/processed/windows")
    out.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out / f"windows_{month}.parquet", index=False)

    print(f"{month}: {len(raw):,} raw trips -> {len(clean):,} clean trips -> {len(windows):,} windows")
    print("\nTrips per window:")
    print(windows["trips"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(1))
    print(f"\nWindows with fewer than 5 trips: {(windows['trips'] < 5).mean():.1%}")
    print("\nFirst 10 windows:")
    print(windows.head(10))
    return windows


if __name__ == "__main__":
    months = pd.period_range("2025-01", "2026-09", freq="M").astype(str)
    for month in months:
        if (Path("data/processed/windows") / f"windows_{month}.parquet").exists():
            print(f"Already have windows for {month}")
            continue
        build_month(month)
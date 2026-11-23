"""Build the label: did the price drop within the horizon?"""
from pathlib import Path

import pandas as pd

MIN_TRIPS = 5
START = "2025-01-01"   # NEW: first day of our data range


def load_all_windows(folder="data/processed/windows"):
    """Combine every month's snapshots into one table."""
    files = sorted(Path(folder).glob("windows_*.parquet"))
    w = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    w = w[w["window_start"] >= START]   # NEW: drop stray snapshots from before our range
    # Snapshots at month edges can appear in two files; keep the bigger piece
    w = (w.sort_values("trips", ascending=False)
          .drop_duplicates(["PULocationID", "window_start"])
          .sort_values(["PULocationID", "window_start"])
          .reset_index(drop=True))
    return w


def add_future_price(w, horizon="30min"):
    """Attach the price from the same zone, `horizon` later, to each snapshot."""
    future = w[["PULocationID", "window_start", "fare_per_mile", "trips"]].copy()
    future["window_start"] = future["window_start"] - pd.Timedelta(horizon)
    future = future.rename(columns={"fare_per_mile": "future_fare_per_mile",
                                    "trips": "future_trips"})
    return w.merge(future, on=["PULocationID", "window_start"], how="inner")


def add_label(w, threshold=0.15, min_trips=MIN_TRIPS):
    """Keep reliable snapshots and mark whether the price dropped by `threshold`."""
    w = w[(w["trips"] >= min_trips) & (w["future_trips"] >= min_trips)].copy()
    w["change"] = w["future_fare_per_mile"] / w["fare_per_mile"] - 1
    w["price_dropped"] = (w["change"] <= -threshold).astype(int)
    return w
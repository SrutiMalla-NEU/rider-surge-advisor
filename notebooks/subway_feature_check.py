"""Check the citywide subway disruption clue: how often it fires, and whether it relates to price drops."""
import pandas as pd
from src.labels import load_all_windows, add_future_price, add_label
from src.features import add_time_features, add_subway_features

pd.set_option("display.width", 200)

w = add_subway_features(add_label(add_future_price(add_time_features(load_all_windows()))))

print(f"Snapshots: {len(w):,}")
print(f"Unknown (after the alert data ends): {w['major_subway_disruptions'].isna().sum():,}")

known = w.dropna(subset=["major_subway_disruptions"])
groups = known["major_subway_disruptions"].clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"})

print("\nShare of snapshots by number of major disruptions active:")
print(groups.value_counts(normalize=True).sort_index().round(3))

print("\nFor each group: median price per mile, and how often the price really dropped 15%+")
print(known.groupby(groups).agg(
    snapshots=("fare_per_mile", "size"),
    median_price=("fare_per_mile", "median"),
    share_dropped=("price_dropped", "mean"),
).round(3))
import pandas as pd

pd.set_option("display.width", 200)

w = pd.read_parquet("data/processed/windows/windows_2025-10.parquet")
zones = pd.read_csv("data/raw/nyc/tlc/zones/taxi_zone_lookup.csv").set_index("LocationID")

print("If we required a minimum number of trips per window:")
for n in [3, 5, 10]:
    small = w["trips"] < n
    share_of_trips = w.loc[small, "trips"].sum() / w["trips"].sum()
    print(f"  Under {n:>2} trips: {small.mean():5.1%} of windows, {share_of_trips:5.1%} of trips")

print("\nThe 10 quietest zones in October:")
by_zone = w.groupby("PULocationID")["trips"].sum().sort_values()
print(by_zone.head(10).to_frame().join(zones[["Zone", "Borough"]]))
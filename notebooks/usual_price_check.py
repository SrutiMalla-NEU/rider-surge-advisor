import pandas as pd
from src.labels import load_all_windows, add_future_price, add_label
from src.features import add_time_features, add_usual_price, SPLIT_DATE

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

w = add_label(add_future_price(add_time_features(load_all_windows())))
w = add_usual_price(w)

print(f"Snapshots: {len(w):,}, missing usual price: {w['usual_fare_per_mile'].isna().sum():,}")
print(w[["PULocationID", "window_start", "fare_per_mile", "usual_fare_per_mile",
         "usual_change_30min", "price_dropped"]].head())

# Does the clue help? Group snapshots by the usual change, and see how often the price really dropped
test = w[w["window_start"] >= SPLIT_DATE].copy()
test["usual_change_group"] = pd.qcut(test["usual_change_30min"], 5,
                                     labels=["biggest usual drop", "2", "3", "4", "biggest usual rise"])
print("\nTest months: share of snapshots where the price really dropped 15%+")
print(test.groupby("usual_change_group", observed=True)["price_dropped"].mean().round(3))
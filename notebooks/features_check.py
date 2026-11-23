import pandas as pd
from src.labels import load_all_windows, MIN_TRIPS
from src.features import add_time_features, add_weather, add_holiday_features

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

w = add_holiday_features(add_weather(add_time_features(load_all_windows())))
w = w[w["trips"] >= MIN_TRIPS]

print(f"Snapshots: {len(w):,}")
print(f"Snapshots missing weather: {w['temperature_2m'].isna().sum():,}")
print(w.head())

print(f"\nShare of snapshots while raining: {w['is_raining'].mean():.1%}")
print("\nMedian price per mile, dry vs. raining:")
print(w.groupby("is_raining")["fare_per_mile"].median().round(2))
print("\nMedian price per mile by day of week (0 = Monday):")
print(w.groupby("day_of_week")["fare_per_mile"].median().round(2))

print("\nMedian price per mile: normal vs. special night, late-night hours only (10 p.m.-3 a.m.)")
late = w[w["hour"].isin([22, 23, 0, 1, 2])]
print(late.groupby("is_special_night")["fare_per_mile"].median().round(2))
print(f"\nHoliday snapshots: {w['is_holiday'].mean():.1%}, special-night snapshots: {w['is_special_night'].mean():.1%}")
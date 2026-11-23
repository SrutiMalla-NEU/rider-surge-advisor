"""Build the training table: one row per snapshot, with every clue and the answer."""
from pathlib import Path

import pandas as pd
from src.labels import load_all_windows, add_future_price, add_label
from src.features import (add_time_features, add_weather, add_holiday_features,
                          add_usual_price, add_subway_features, SPLIT_DATE)

OUT = Path("data/processed/training_table_30min.parquet")

# The clues the model may use. Each one is something the live app will actually know.
FEATURES = [
    "PULocationID",                                  # zone
    "hour", "minute", "day_of_week", "is_weekend",   # time
    "is_holiday", "is_special_night",                # holidays
    "temperature_2m", "precipitation", "snowfall",   # weather
    "wind_speed_10m", "is_raining",
    "usual_fare_per_mile", "usual_change_30min",     # usual price
    "major_subway_disruptions", "subway_known",      # subway (FIX: added subway_known)
]
LABEL = "price_dropped"
KEEP_FOR_CHECKING = ["window_start", "fare_per_mile", "trips"]   # never given to the model


def build_table():
    """Run every step, in order, and return the finished table."""
    w = load_all_windows()
    w = add_label(add_future_price(w, horizon="30min"))
    w = add_time_features(w)
    w = add_weather(w)
    w = add_holiday_features(w)
    w = add_usual_price(w)
    w = add_subway_features(w)
    return w[KEEP_FOR_CHECKING + FEATURES + [LABEL]]


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    table = build_table()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT, index=False)

    is_train = table["window_start"] < SPLIT_DATE
    print(f"Saved {len(table):,} rows to {OUT}")
    print(f"Training rows: {is_train.sum():,}   Test rows: {(~is_train).sum():,}")
    print(f"Price dropped 15%+:  training {table.loc[is_train, LABEL].mean():.1%},"
          f"  test {table.loc[~is_train, LABEL].mean():.1%}")
    print("\nMissing values per clue (should all be 0 now):")
    print(table[FEATURES].isna().sum())
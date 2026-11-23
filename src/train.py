"""Train the main model (gradient boosting) once and save it, so checks can reuse it."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features import SPLIT_DATE

TABLE = "data/processed/training_table_30min.parquet"
MODEL_FILE = Path("models/boosting_30min.joblib")

CLUES = ["zone_code", "hour", "minute", "day_of_week", "is_weekend",
         "is_holiday", "is_special_night",
         "temperature_2m", "precipitation", "snowfall", "wind_speed_10m", "is_raining",
         "usual_fare_per_mile", "usual_change_30min",
         "major_subway_disruptions", "subway_known"]


def add_zone_code(df, zone_code):
    """Turn zone IDs into category codes; zones outside the busiest 254 share code 254."""
    return df.assign(zone_code=df["PULocationID"].map(zone_code).fillna(254).astype(int))


def train():
    """Train on a 3M-snapshot sample of the training months and return everything needed to predict."""
    table = pd.read_parquet(TABLE)
    train_all = table[table["window_start"] < SPLIT_DATE]

    busiest = train_all["PULocationID"].value_counts().index[:254]
    zone_code = {zone: i for i, zone in enumerate(busiest)}
    sample = add_zone_code(train_all.sample(3_000_000, random_state=42), zone_code)

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                           categorical_features=["zone_code"], random_state=42)
    model.fit(sample[CLUES], sample["price_dropped"])
    return {"model": model, "zone_code": zone_code, "clues": CLUES}


if __name__ == "__main__":
    print("Training the gradient boosting model (several minutes)...")
    bundle = train()
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    print(f"Trees built: {bundle['model'].n_iter_}")
    print(f"Saved model to {MODEL_FILE}")
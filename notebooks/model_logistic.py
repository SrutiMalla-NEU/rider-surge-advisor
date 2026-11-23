"""First model: logistic regression, compared with the baselines at the same share of "wait" advice."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.evaluate import score
from src.features import SPLIT_DATE

pd.set_option("display.width", 200)

table = pd.read_parquet("data/processed/training_table_30min.parquet")
train = table[table["window_start"] < SPLIT_DATE].sample(1_000_000, random_state=42)
test = table[table["window_start"] >= SPLIT_DATE]

# Clues that are categories (each value becomes its own yes/no column)
CATEGORICAL = ["hour", "day_of_week"]
# Clues that are numbers (rescaled to similar ranges)
NUMERIC = ["minute", "is_weekend", "is_holiday", "is_special_night",
           "temperature_2m", "precipitation", "snowfall", "wind_speed_10m", "is_raining",
           "usual_fare_per_mile", "usual_change_30min",
           "major_subway_disruptions", "subway_known"]
CLUES = CATEGORICAL + NUMERIC

model = make_pipeline(
    ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numbers", StandardScaler(), NUMERIC),
    ]),
    LogisticRegression(max_iter=1000),
)

print("Training on 1,000,000 snapshots...")
model.fit(train[CLUES], train["price_dropped"])

print("Predicting on the test months...")
probs = pd.Series(model.predict_proba(test[CLUES])[:, 1], index=test.index)
y = test["price_dropped"]

# Same share of "wait" advice as each baseline rule, so precision can be compared fairly
results = []
for share, baseline in [(0.312, "usually drops 2%+"), (0.098, "usually drops 5%+"),
                        (0.018, "usually drops 10%+")]:
    cutoff = probs.quantile(1 - share)
    says_wait = (probs >= cutoff).astype(int)
    results.append(score(f"Logistic (same wait share as '{baseline}')", y, says_wait))
print("\n" + pd.DataFrame(results).round(3).to_string(index=False))

# Which clues push toward "the price will drop"? (numbers only; bigger = stronger push)
names = model[0].get_feature_names_out()
weights = pd.Series(model[-1].coef_[0], index=names)
print("\nWeights for the number clues (positive = more likely to drop):")
print(weights[weights.index.str.startswith("numbers__")].sort_values().round(3))
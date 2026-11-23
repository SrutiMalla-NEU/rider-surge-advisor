"""Second model: gradient boosting, compared with the baselines at the same share of "wait" advice."""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from src.evaluate import score
from src.features import SPLIT_DATE

pd.set_option("display.width", 200)

table = pd.read_parquet("data/processed/training_table_30min.parquet")
train_all = table[table["window_start"] < SPLIT_DATE].copy()
test = table[table["window_start"] >= SPLIT_DATE].copy()

# Zone as a category. This model allows at most 255 categories, so the 254 busiest
# training zones get their own code, and the few quietest ones share code 254 ("other").
busiest = train_all["PULocationID"].value_counts().index[:254]
zone_code = {zone: i for i, zone in enumerate(busiest)}
for df in (train_all, test):
    df["zone_code"] = df["PULocationID"].map(zone_code).fillna(254).astype(int)

train = train_all.sample(3_000_000, random_state=42)

CLUES = ["zone_code", "hour", "minute", "day_of_week", "is_weekend",
         "is_holiday", "is_special_night",
         "temperature_2m", "precipitation", "snowfall", "wind_speed_10m", "is_raining",
         "usual_fare_per_mile", "usual_change_30min",
         "major_subway_disruptions", "subway_known"]

model = HistGradientBoostingClassifier(
    max_iter=300,                       # at most 300 trees
    learning_rate=0.1,                  # how big a correction each new tree makes
    categorical_features=["zone_code"],
    random_state=42,                    # early stopping is on by default for big data
)

print("Training on 3,000,000 snapshots (this can take several minutes)...")
model.fit(train[CLUES], train["price_dropped"])
print(f"Trees built: {model.n_iter_}")

probs = pd.Series(model.predict_proba(test[CLUES])[:, 1], index=test.index)
y = test["price_dropped"]

results = []
for share, baseline in [(0.312, "usually drops 2%+"), (0.098, "usually drops 5%+"),
                        (0.018, "usually drops 10%+")]:
    cutoff = probs.quantile(1 - share)
    says_wait = (probs >= cutoff).astype(int)
    results.append(score(f"Boosting (same wait share as '{baseline}')", y, says_wait))
print("\n" + pd.DataFrame(results).round(3).to_string(index=False))

# Which clues matter? Shuffle one clue at a time and measure how much worse the model gets.
print("\nMeasuring clue importance on 200,000 test snapshots...")
sample = test.sample(200_000, random_state=42)
importance = permutation_importance(model, sample[CLUES], sample["price_dropped"],
                                    scoring="average_precision", n_repeats=3,
                                    random_state=42, n_jobs=-1)
print("Importance (how much worse the model gets when the clue is shuffled):")
print(pd.Series(importance.importances_mean, index=CLUES)
      .sort_values(ascending=False).round(4))
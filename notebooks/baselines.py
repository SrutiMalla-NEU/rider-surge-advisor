"""Score simple rules on the test months. Any model must beat these."""
import pandas as pd
from src.evaluate import score
from src.features import SPLIT_DATE

pd.set_option("display.width", 200)

table = pd.read_parquet("data/processed/training_table_30min.parquet")
test = table[table["window_start"] >= SPLIT_DATE]
y = test["price_dropped"]
print(f"Test snapshots: {len(test):,}   Real drops: {y.mean():.1%}\n")

results = [
    score("Always book now", y, pd.Series(0, index=y.index)),
    score("Always wait", y, pd.Series(1, index=y.index)),
]
for cut in [0.02, 0.05, 0.10]:
    says_wait = (test["usual_change_30min"] <= -cut).astype(int)
    results.append(score(f"Wait if price usually drops {cut:.0%}+", y, says_wait))

print(pd.DataFrame(results).round(3).to_string(index=False))
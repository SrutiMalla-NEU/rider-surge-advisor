"""Check the saved model in the situations that matter most for stranded riders."""
import joblib
import pandas as pd
from src.features import SPLIT_DATE
from src.train import MODEL_FILE, TABLE, add_zone_code

pd.set_option("display.width", 200)

bundle = joblib.load(MODEL_FILE)
table = pd.read_parquet(TABLE)
test = add_zone_code(table[table["window_start"] >= SPLIT_DATE], bundle["zone_code"])
test["prob"] = bundle["model"].predict_proba(test[bundle["clues"]])[:, 1]

zone_names = pd.read_csv("data/raw/nyc/tlc/zones/taxi_zone_lookup.csv").set_index("LocationID")["Zone"]
nightlife = zone_names[zone_names.isin(["East Village", "Lower East Side",
                                        "West Village", "Union Sq"])].index
late = test["hour"].isin([22, 23, 0, 1, 2, 3])

situations = {
    "All test snapshots": pd.Series(True, index=test.index),
    "Late night (10 p.m.-4 a.m.)": late,
    "Late night, weekend": late & (test["is_weekend"] == 1),
    "Late night, nightlife zones": late & test["PULocationID"].isin(nightlife),
    "Holidays": test["is_holiday"] == 1,
    "Heavy rain (0.1 in+ in past hour)": test["precipitation"] >= 0.1,
    "2+ major subway problems": test["major_subway_disruptions"] >= 2,
    "Special nights": test["is_special_night"] == 1,
}

rows = []
for name, mask in situations.items():
    group = test[mask]
    row = {"situation": name, "snapshots": len(group)}
    if len(group) > 0:
        top_n = max(1, len(group) // 10)   # each method says "wait" for its top 10%
        row["real_drops"] = group["price_dropped"].mean()
        row["rule_right_when_wait"] = group.nsmallest(top_n, "usual_change_30min")["price_dropped"].mean()
        row["model_right_when_wait"] = group.nlargest(top_n, "prob")["price_dropped"].mean()
    rows.append(row)

print(pd.DataFrame(rows).round(3).to_string(index=False))
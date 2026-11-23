"""Explore subway-only MTA service alerts: labels, affected lines, and how long problems last."""
import pandas as pd

pd.set_option("display.width", 200)

alerts = pd.read_parquet("data/raw/nyc/mta/service_alerts.parquet")
subway = alerts[alerts["agency"] == "NYCT Subway"].copy()
print(f"Subway alerts: {len(subway):,}")

print("\nStatus labels (combined labels split apart):")
labels = subway["status_label"].str.split("|").explode().str.strip()
print(labels.value_counts().head(25))

print("\nMost-affected lines:")
lines = subway["affected"].str.split("|").explode().str.strip()
print(lines.value_counts().head(30))

print("\nHow long problems last (first to last alert with the same event ID):")
events = subway.groupby("event_id").agg(start=("date", "min"), end=("date", "max"),
                                         updates=("alert_id", "count"))
events["minutes"] = (events["end"] - events["start"]).dt.total_seconds() / 60
print(f"Incidents: {len(events):,}")
print(events["minutes"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(1))

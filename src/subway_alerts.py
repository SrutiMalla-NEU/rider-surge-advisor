"""Download MTA service alerts for our date range from NY Open Data."""
from pathlib import Path

import pandas as pd
import requests

URL = "https://data.ny.gov/resource/7kct-peq7.json"
OUT = Path("data/raw/nyc/mta/service_alerts.parquet")
PAGE = 50000


def fetch_all():
    """Download all alerts from Jan 2025 to Sep 2026, one page of rows at a time."""
    rows, offset = [], 0
    while True:
        params = {
            "$where": "date >= '2025-01-01T00:00:00' AND date < '2026-10-01T00:00:00'",
            "$order": "alert_id",
            "$limit": PAGE,
            "$offset": offset,
        }
        response = requests.get(URL, params=params, timeout=120)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        offset += PAGE
        print(f"Downloaded {len(rows):,} alerts so far")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    alerts = fetch_all()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_parquet(OUT, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 90)
    print(f"\nTotal alerts: {len(alerts):,}")
    print(f"Date range: {alerts['date'].min()} to {alerts['date'].max()}")
    print("\nAlerts by agency:")
    print(alerts["agency"].value_counts())
    print("\nMost common status labels (all agencies):")
    print(alerts["status_label"].value_counts().head(20))
    print("\nA few example headers:")
    print(alerts["header"].sample(8, random_state=1).to_string())
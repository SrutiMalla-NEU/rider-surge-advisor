"""Download TLC trip record files for a range of months."""
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"   
DATASETS = {"fhvhv": "hvfhv", "yellow": "yellow", "green": "green"}   


def download(kind, month):
    """Download one file, skipping it if it's already there."""
    folder = Path("data/raw/nyc/tlc") / DATASETS[kind]
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{kind}_tripdata_{month}.parquet"
    path = folder / name

    if path.exists():
        print(f"Already have {name}")
        return

    response = requests.get(f"{BASE_URL}/{name}", stream=True, timeout=60)
    if response.status_code != 200:
        print(f"Not available: {name} (status {response.status_code})")
        return

    temp = path.with_suffix(".part")
    with open(temp, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    temp.rename(path)
    print(f"Downloaded {name}")


if __name__ == "__main__":
    months = pd.period_range("2025-01", "2026-09", freq="M").astype(str)
    for month in months:
        for kind in DATASETS:
            download(kind, month)
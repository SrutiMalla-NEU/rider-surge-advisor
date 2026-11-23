"""Download historical hourly weather for New York City from Open-Meteo."""
from datetime import date
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
CENTRAL_PARK = {"latitude": 40.7829, "longitude": -73.9654}
VARIABLES = ["temperature_2m", "precipitation", "rain", "snowfall",
             "wind_speed_10m", "weather_code"]
OUT = Path("data/raw/nyc/weather/weather_nyc_hourly.parquet")


def fetch(start, end):
    """Ask Open-Meteo for hourly weather between two dates."""
    params = {
        **CENTRAL_PARK,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARIABLES),
        "timezone": "America/New_York",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "wind_speed_unit": "mph",
    }
    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


if __name__ == "__main__":
    weather = pd.concat([fetch("2025-01-01", "2025-12-31"),
                         fetch("2026-01-01", "2026-09-30")], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(OUT, index=False)

    print(f"Hours of weather: {len(weather):,}")
    print(weather.head())
    print("\nCheck: October 30, 2025, 1-7 p.m. (the record storm)")
    storm = weather[(weather["time"].dt.date == date(2025, 10, 30))
                    & weather["time"].dt.hour.between(13, 19)]
    print(storm[["time", "precipitation", "temperature_2m", "weather_code"]])
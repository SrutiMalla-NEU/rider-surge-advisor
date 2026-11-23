"""Add clues (features) to each snapshot."""
from datetime import timedelta

import holidays
import pandas as pd

WEATHER_FILE = "data/raw/nyc/weather/weather_nyc_hourly.parquet"
ALERTS_FILE = "data/raw/nyc/mta/service_alerts.parquet"
YEARS = [2024, 2025, 2026]
US_HOLIDAYS = holidays.US(state="NY", years=YEARS)
SPLIT_DATE = "2026-04-01"   # training before this date, testing from this date on

MAJOR_LABELS = {"suspended", "part-suspended", "severe-delays"}
MAJOR_MINUTES = 60
ALERTS_END = "2026-08-12"   # the MTA alert data stops here; later snapshots get "unknown"


def add_time_features(w):
    """Add hour, minute, day of week and weekend clues from the snapshot's start time."""
    t = w["window_start"]
    return w.assign(
        hour=t.dt.hour,
        minute=t.dt.minute,
        day_of_week=t.dt.dayofweek,               # Monday = 0 ... Sunday = 6
        is_weekend=(t.dt.dayofweek >= 5).astype(int),
    )


def add_weather(w, weather_file=WEATHER_FILE):
    """Attach the weather from the start of each snapshot's hour."""
    weather = pd.read_parquet(weather_file).drop_duplicates("time")
    weather = weather.rename(columns={"time": "weather_hour"})
    w = w.assign(weather_hour=w["window_start"].dt.floor("h"))
    w = w.merge(weather, on="weather_hour", how="left")
    w["is_raining"] = (w["precipitation"] >= 0.01).astype(int)
    return w.drop(columns="weather_hour")


def special_nights():
    """Big party nights that aren't official holidays."""
    days = []
    for y in YEARS:
        days += [f"{y}-03-17", f"{y}-10-31", f"{y}-12-24", f"{y}-12-31"]
    for day, name in US_HOLIDAYS.items():
        if "Thanksgiving" in name:
            days.append(str(day - timedelta(days=1)))   # the night before Thanksgiving
    return pd.to_datetime(days)


def add_holiday_features(w):
    """Mark official holidays and big party nights. Times before 6 a.m. count as the night before."""
    night_of = (w["window_start"] - pd.Timedelta(hours=6)).dt.normalize()
    holiday_dates = pd.to_datetime(list(US_HOLIDAYS.keys()))
    return w.assign(
        is_holiday=night_of.isin(holiday_dates).astype(int),
        is_special_night=night_of.isin(special_nights()).astype(int),
    )


def add_usual_price(w, split_date=SPLIT_DATE):
    """Add the usual price for each zone and time slot, learned from training months only,
    and how much the price usually changes over the next 30 minutes."""
    w = w.assign(slot=w["hour"] * 4 + w["minute"] // 15)   # 0 to 95: which 15-minute slot of the day
    train = w[w["window_start"] < split_date]
    keys = ["PULocationID", "is_weekend", "slot"]

    usual = (train.groupby(keys)["fare_per_mile"].median()
             .rename("usual_fare_per_mile").reset_index())

    # The usual price two slots (30 minutes) later, lined up with the current slot
    later = usual.rename(columns={"usual_fare_per_mile": "usual_fare_30min_later"})
    later["slot"] = (later["slot"] - 2) % 96

    w = w.merge(usual, on=keys, how="left").merge(later, on=keys, how="left")
    w["usual_change_30min"] = w["usual_fare_30min_later"] / w["usual_fare_per_mile"] - 1

    # FIX: fill gaps. Missing usual price -> the zone's overall usual price (training months).
    # Missing usual change -> 0, meaning "no usual change expected".
    zone_usual = train.groupby("PULocationID")["fare_per_mile"].median()
    w["usual_fare_per_mile"] = w["usual_fare_per_mile"].fillna(train["fare_per_mile"].median())
    w["usual_change_30min"] = w["usual_change_30min"].fillna(0)
    return w


def major_subway_counts(alerts_file=ALERTS_FILE):
    """For every 15-minute slot, count the major subway disruptions active at that moment.

    A problem counts as major from the first alert with a serious label, or once it has
    actually lasted 60 minutes, so no snapshot uses information from the future.
    """
    a = pd.read_parquet(alerts_file)
    a = a[a["agency"] == "NYCT Subway"].copy()
    labels = a["status_label"].fillna("").str.split("|")
    a["is_major_label"] = labels.apply(lambda ls: any(l.strip() in MAJOR_LABELS for l in ls))

    incidents = a.groupby("event_id").agg(start=("date", "min"), end=("date", "max"))
    first_serious = a[a["is_major_label"]].groupby("event_id")["date"].min()
    long_enough = (incidents["end"] - incidents["start"]) >= pd.Timedelta(minutes=MAJOR_MINUTES)
    reached_60 = (incidents["start"] + pd.Timedelta(minutes=MAJOR_MINUTES)).where(long_enough)

    # The moment each problem becomes "major": whichever comes first
    incidents["major_from"] = pd.concat(
        [first_serious.reindex(incidents.index), reached_60], axis=1).min(axis=1)
    incidents = incidents.dropna(subset=["major_from"])

    # When it stops counting: its last alert, or 60 minutes after it became major if
    # there were no later alerts
    one_hour_later = incidents["major_from"] + pd.Timedelta(minutes=MAJOR_MINUTES)
    incidents["major_until"] = incidents["end"].where(incidents["end"] > incidents["major_from"],
                                                      one_hour_later)

    slots = [pd.date_range(f.ceil("15min"), u.floor("15min"), freq="15min")
             for f, u in zip(incidents["major_from"], incidents["major_until"])]
    return pd.Series(slots).explode().value_counts()


def add_subway_features(w):
    """Add how many major subway disruptions are active at each snapshot's start (citywide).

    FIX: after the alert data ends, the count is unknown. Instead of leaving it empty,
    `subway_known` says whether we have data (1) or not (0), and the count is filled with 0.
    """
    counts = major_subway_counts()
    w = w.copy()
    w["subway_known"] = (w["window_start"] < ALERTS_END).astype(int)
    w["major_subway_disruptions"] = w["window_start"].map(counts).fillna(0)
    w.loc[w["subway_known"] == 0, "major_subway_disruptions"] = 0
    return w
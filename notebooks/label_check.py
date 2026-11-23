from src.labels import load_all_windows, add_future_price, MIN_TRIPS

w = add_future_price(load_all_windows())
w = w[(w["trips"] >= MIN_TRIPS) & (w["future_trips"] >= MIN_TRIPS)]
change = w["future_fare_per_mile"] / w["fare_per_mile"] - 1

print(f"Snapshots with a reliable 30-minute-later price: {len(w):,}")
print("\nPrice change 30 minutes later:")
print(change.describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).round(3))

print("\nHow often does the price drop by at least...")
for t in [0.05, 0.10, 0.15, 0.20, 0.25]:
    print(f"  {t:.0%}: {(change <= -t).mean():.1%} of snapshots")
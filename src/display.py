def hour_label(h):
    """Turn a 24-hour number into a readable label, like 23 -> '11 PM'."""
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12} {suffix}"
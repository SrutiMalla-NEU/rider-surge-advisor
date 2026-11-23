from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

pd.set_option("display.width", 200)

rows = []
for path in sorted(Path("data/raw/nyc/tlc").rglob("*.parquet")):
    meta = pq.ParquetFile(path).metadata
    rows.append({
        "dataset": path.parent.name,
        "file": path.name,
        "rows": meta.num_rows,
        "columns": tuple(pq.read_schema(path).names),
        "size_mb": round(path.stat().st_size / 1e6),
    })
files = pd.DataFrame(rows)

print(files[["file", "rows", "size_mb"]].to_string(index=False))
print(f"\nTotal size: {files['size_mb'].sum():,} MB")

print("\nColumn sets per dataset (1 = consistent across all months):")
print(files.groupby("dataset")["columns"].nunique())

print("\nMonths whose columns differ from the first month:")
for dataset in ["yellow", "green"]:
    sub = files[files["dataset"] == dataset]
    first = set(sub["columns"].iloc[0])
    for _, row in sub.iterrows():
        cols = set(row["columns"])
        if cols != first:
            print(f"{row['file']}: added {cols - first or 'nothing'}, missing {first - cols or 'nothing'}")
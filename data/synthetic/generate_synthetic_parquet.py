# /data/synthetic/generate_synthetic_parquet.py
import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate(n_rows: int = 100_000, seed: int = 0, out_path: str = "./data/synthetic/synthetic.parquet"):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n_rows, freq="T"),
        "feature1": rng.normal(0, 1, n_rows),
        "feature2": rng.integers(0, 100, n_rows),
        "target": rng.binomial(1, 0.5, n_rows)
    })
    out_dir = os.path.dirname(out_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {n_rows} rows to {out_path}")

if __name__ == "__main__":
    generate(n_rows=100000, seed=42)

# mlflow_run_template.py
"""
Minimal MLflow-based, end-to-end run example.

- Uses a local file-based MLflow store by default
- Logs: parameters, metrics, artifacts
- Demonstrates end-to-end logging for Phase 0

Environment:
- MLFLOW_TRACKING_URI: path to mlruns (default: ./mlruns)
- MLFLOW_EXPERIMENT_NAME: name of the experiment
"""

import os
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "Phase0_Phase0")

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def select_target_column(df: pd.DataFrame) -> str:
    # Preferable explicit name
    if "y" in df.columns:
        return "y"
    # Common alternative
    if "target" in df.columns:
        return "target"
    # Fallback: use the last numeric column
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        return numeric_cols[-1]
    raise ValueError("No numeric target column found in data.")

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    # Load your external Parquet data
    parquet_path = "/Users/jeells/Documents/GitHub/infera-group/data/synthetic/synthetic.parquet"
    if Path(parquet_path).exists():
        df = pd.read_parquet(parquet_path)
        print(f"Loaded parquet with shape {df.shape}. Columns: {list(df.columns)}")
    else:
        # Fallback to synthetic data if parquet isn't found
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=100, freq="T"),
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.randint(0, 100, 100),
            "target": np.random.binomial(1, 0.5, 100)
        })
        print("Parquet not found; using synthetic data.")

    target_col = select_target_column(df)
    mean_target = float(df[target_col].mean())

    with mlflow.start_run(run_name=f"hello_world_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}") as run:
        mlflow.log_param("param_scale", 1.0)
        mlflow.log_param("param_seed", 42)

        # Log a metric derived from the data
        mlflow.log_metric("mean_target", mean_target)

        # Optionally log the Parquet as an artifact for provenance
        artifact_dir = "artifacts"
        ensure_dir(artifact_dir)
        if Path(parquet_path).exists():
            mlflow.log_artifact(parquet_path, artifact_path="data_parquet")

        # Small summary artifact
        artifact_path = os.path.join(artifact_dir, "summary.txt")
        with open(artifact_path, "w") as f:
            f.write(f"mean_{target_col}={mean_target}\nrows={len(df)}\n")
        mlflow.log_artifact(artifact_path)

    print("MLflow run completed. Check mlruns/ for results.")

if __name__ == "__main__":
    main()

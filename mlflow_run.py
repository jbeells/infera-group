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

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    # Dummy synthetic data to log
    data = {
        "x": np.arange(100),
        "y": np.random.normal(loc=0.0, scale=1.0, size=100)
    }
    df = pd.DataFrame(data)

    with mlflow.start_run(run_name=f"hello_world_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}") as run:
        mlflow.log_param("param_scale", 1.0)
        mlflow.log_param("param_seed", 42)

        # Simple metric derived from data
        mean_y = float(df["y"].mean())
        mlflow.log_metric("mean_y", mean_y)

        # Log a small artifact
        artifact_dir = "artifacts"
        ensure_dir(artifact_dir)
        artifact_path = os.path.join(artifact_dir, "summary.txt")
        with open(artifact_path, "w") as f:
            f.write(f"mean_y={mean_y}\nrows={len(df)}\n")
        mlflow.log_artifact(artifact_path)

    print("MLflow run completed. Check mlruns/ for results.")

if __name__ == "__main__":
    main()

### 1.3 monitoring.py

# /shared/monitoring.py
"""
Lightweight drift and calibration monitoring.

Public API:
- psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float
- ks_statistic(expected: pd.Series, actual: pd.Series) -> float
- drift_report(df_expected, df_actual, feature_cols: List[str]) -> Dict[str, dict]
"""

from typing import List, Dict
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e_hist, bin_edges = np.histogram(expected, bins=bins, density=True)
    a_hist, _ = np.histogram(actual, bins=bin_edges, density=True)
    # avoid divide by zero
    e_hist = np.clip(e_hist, 1e-6, None)
    a_hist = np.clip(a_hist, 1e-6, None)
    psi_vals = (a_hist - e_hist) * np.log(a_hist / e_hist)
    return float(np.sum(psi_vals))

def ks_statistic(expected: pd.Series, actual: pd.Series) -> float:
    stat, p = ks_2samp(expected, actual)
    return float(stat)

def drift_report(df_expected: pd.DataFrame, df_actual: pd.DataFrame, feature_cols: List[str]) -> Dict[str, dict]:
    report = {}
    for col in feature_cols:
        if col in df_expected.columns and col in df_actual.columns:
            report[col] = {
                "psi": psi(df_expected[col], df_actual[col]),
                "ks": ks_statistic(df_expected[col], df_actual[col])
            }
    return report

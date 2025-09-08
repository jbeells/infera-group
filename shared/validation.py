# /shared/validation.py
"""
Data quality and leakage checks.

Public API:
- time_split(df, date_col, train_end) -> (train_df, test_df)
- leakage_check(train_df, test_df, feature_cols) -> dict
- basic_sanity_checks(df, required_types: dict) -> dict
"""

from typing import Tuple, List, Dict
import pandas as pd

def time_split(df: pd.DataFrame, date_col: str, train_end) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df[date_col] <= train_end]
    test = df[df[date_col] > train_end]
    return train, test

def leakage_check(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    leaks = {}
    for col in feature_cols:
        if col in train_df.columns and col in test_df.columns:
            train_mean = train_df[col].mean()
            test_mean = test_df[col].mean()
            leaks[col] = float(abs(train_mean - test_mean))
    return leaks

def basic_sanity_checks(df: pd.DataFrame, required_types: Dict[str, type]) -> Dict[str, str]:
    issues = {}
    for col, typ in required_types.items():
        # Use pandas' dtype comparison helpers for reliability
        import pandas as pd
        if not pd.api.types.is_dtype_equal(df[col].dtype, typ):
            issues[col] = f"wrong_type({df[col].dtype}, expected={typ})"
    return issues

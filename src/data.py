"""
Data loading and preprocessing.

Provides two main functions:
  - load_classification_data() : returns train/test split for PCG classification
  - load_regression_data()     : returns train/test split for grain size regression

Both functions centralise the logic of:
  1. Reading the Excel file
  2. Encoding the categorical 'Condition' column
  3. Converting the PCG target from yes/no to 1/0
  4. Filtering rows (dropping NaN in target)
  5. Train/test splitting with a fixed random state
  6. Standardising the features (zero mean, unit variance)

This avoids duplicating the same preprocessing logic across every
experiment script.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    DATASET_PATH,
    GRAIN_SIZE_COLUMN,
    INPUT_COLUMNS,
    PCG_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
)


def _load_raw_dataframe() -> pd.DataFrame:
    """Read the Excel file and encode the categorical 'Condition' column.

    Returns a DataFrame with an extra 'Condition_enc' column (numeric).
    Original 'Condition' column is kept for reference but not used downstream.
    """
    df = pd.read_excel(DATASET_PATH)

    # LabelEncoder turns strings like 'LBT-2mm/s' into integers 0, 1, 2, ...
    # We only do this for Condition; other inputs are already numeric.
    le = LabelEncoder()
    df["Condition_enc"] = le.fit_transform(df["Condition"].astype(str))

    return df


def _encoded_input_columns() -> list[str]:
    """Return the list of input column names, with 'Condition' replaced
    by its encoded numeric version 'Condition_enc'."""
    return ["Condition_enc"] + INPUT_COLUMNS[1:]


def load_classification_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for PCG classification (binary: yes=1, no=0).

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
        Features are standardised (zero mean, unit variance on train set).
        Targets are binary integers (0 or 1).
    """
    df = _load_raw_dataframe()
    input_cols = _encoded_input_columns()

    # Convert PCG string ('yes'/'no') to binary integer
    df["PCG_bin"] = (df[PCG_COLUMN].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Drop rows where the target is missing
    mask = df["PCG_bin"].notna()
    X = df.loc[mask, input_cols].values
    y = df.loc[mask, "PCG_bin"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Fit scaler on TRAIN only (to avoid data leakage), then apply to both
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_regression_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for grain size regression.

    IMPORTANT: only samples WITHOUT PCG are used. The presence of PCG
    fundamentally alters the microstructural evolution mechanism, so
    mixing PCG-positive and PCG-negative samples would contaminate
    the regression model (this restriction was suggested by the professor
    and is physically motivated).

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
        Features are standardised. Target is continuous grain size.
    """
    df = _load_raw_dataframe()
    input_cols = _encoded_input_columns()

    # Keep only rows where PCG == 'no'
    no_pcg_mask = df[PCG_COLUMN].astype(str).str.strip().str.lower() == "no"
    df = df[no_pcg_mask].copy()

    X = df[input_cols].values
    y = df[GRAIN_SIZE_COLUMN].values

    # Drop rows where the regression target is NaN
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

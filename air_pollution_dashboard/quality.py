"""Input validation and particulate-only US EPA AQI (May 2024 breakpoints).

This is not India's CPCB AQI, a full six-pollutant AQI, or AirNow NowCast.
Concentrations are truncated before interpolation; indices are rounded half up
and capped at 500 for display. See the AirNow Technical Assistance Document.
"""
import math

import numpy as np
import pandas as pd

PM25_BANDS = (
    (0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
)
PM10_BANDS = (
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
)


def _particulate_index(value, bands, precision):
    if value is None or not np.isfinite(value) or value < 0:
        return np.nan
    concentration = math.floor(float(value) * precision) / precision
    if concentration > bands[-1][1]:
        return 500.0
    for lower, upper, index_lower, index_upper in bands:
        if lower <= concentration <= upper:
            index = (index_upper - index_lower) * (concentration - lower) / (upper - lower) + index_lower
            return float(math.floor(index + 0.5))
    return np.nan


def compute_aqi(pm25, pm10=np.nan):
    indices = [
        _particulate_index(pm25, PM25_BANDS, 10),
        _particulate_index(pm10, PM10_BANDS, 1),
    ]
    available = [i for i in indices if np.isfinite(i)]
    return max(available) if available else np.nan


def validate_measurements(frame, features):
    missing = [f for f in features if f not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(numeric.to_numpy(dtype=float)) | (numeric.to_numpy(dtype=float) < 0)
    if invalid.any():
        rows = np.flatnonzero(invalid.any(axis=1)) + 1
        shown = ", ".join(str(r) for r in rows[:8])
        raise ValueError(
            f"Pollutants must be finite, non-negative numbers. Invalid data row(s): {shown}"
            + ("…" if len(rows) > 8 else "")
        )
    if numeric.empty:
        raise ValueError("The file contains no data rows.")
    return numeric


def csv_export(frame):
    """Neutralize spreadsheet formulas in untrusted text while retaining numbers."""
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].map(
            lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
            else value
        )
    return safe.to_csv(index=False).encode("utf-8")

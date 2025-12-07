"""
Utility Functions Module

This module provides utility functions for data gap analysis.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def calculate_completeness(series: pd.Series) -> float:
    """
    Calculate the completeness percentage of a series.

    Args:
        series: Pandas Series to analyze.

    Returns:
        Completeness percentage (0-100).
    """
    if len(series) == 0:
        return 100.0
    return (series.notna().sum() / len(series)) * 100


def calculate_data_quality_score(
    completeness: float,
    validity: float,
    consistency: float,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> float:
    """
    Calculate an overall data quality score.

    Args:
        completeness: Completeness percentage.
        validity: Validity percentage.
        consistency: Consistency percentage.
        weights: Weights for each dimension.

    Returns:
        Weighted quality score (0-100).
    """
    return (
        completeness * weights[0]
        + validity * weights[1]
        + consistency * weights[2]
    )


def detect_date_format(value: str) -> str | None:
    """
    Detect the date format of a string value.

    Args:
        value: String value to analyze.

    Returns:
        Detected date format string or None.
    """
    date_patterns = {
        "%Y-%m-%d": r"^\d{4}-\d{2}-\d{2}$",
        "%m/%d/%Y": r"^\d{1,2}/\d{1,2}/\d{4}$",
        "%d-%m-%Y": r"^\d{2}-\d{2}-\d{4}$",
        "%d/%m/%Y": r"^\d{2}/\d{2}/\d{4}$",
        "%Y/%m/%d": r"^\d{4}/\d{2}/\d{2}$",
        "%Y-%m-%dT%H:%M:%S": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
    }

    for fmt, pattern in date_patterns.items():
        if re.match(pattern, value.strip()):
            return fmt

    return None


def normalize_string(value: str) -> str:
    """
    Normalize a string for comparison.

    Args:
        value: String to normalize.

    Returns:
        Normalized string.
    """
    return value.strip().lower()


def hash_record(record: pd.Series, columns: list[str] | None = None) -> str:
    """
    Generate a hash for a record for duplicate detection.

    Args:
        record: Pandas Series representing a record.
        columns: Columns to include in hash. Uses all if None.

    Returns:
        MD5 hash string.
    """
    if columns:
        values = record[columns].astype(str).tolist()
    else:
        values = record.astype(str).tolist()

    combined = "|".join(values)
    return hashlib.md5(combined.encode()).hexdigest()


def calculate_entropy(series: pd.Series) -> float:
    """
    Calculate the Shannon entropy of a series.

    Args:
        series: Pandas Series to analyze.

    Returns:
        Entropy value.
    """
    value_counts = series.dropna().value_counts(normalize=True)
    if len(value_counts) == 0:
        return 0.0

    entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
    return float(entropy)


def detect_data_type(series: pd.Series) -> str:
    """
    Infer the semantic data type of a series.

    Args:
        series: Pandas Series to analyze.

    Returns:
        Inferred data type string.
    """
    sample = series.dropna().astype(str).head(100)

    if len(sample) == 0:
        return "empty"

    # Check for email
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    if sample.str.match(email_pattern).mean() > 0.9:
        return "email"

    # Check for URL
    url_pattern = r"^https?://"
    if sample.str.match(url_pattern).mean() > 0.9:
        return "url"

    # Check for phone
    phone_pattern = r"^[\+]?[\d\s\-\(\)]{10,}$"
    if sample.str.match(phone_pattern).mean() > 0.9:
        return "phone"

    # Check for UUID
    uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    if sample.str.lower().str.match(uuid_pattern).mean() > 0.9:
        return "uuid"

    # Check for date
    try:
        pd.to_datetime(sample, format="%Y-%m-%d", errors="raise")
        return "date"
    except (ValueError, TypeError):
        pass

    # Check numeric
    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series):
            return "integer"
        return "float"

    # Check boolean
    unique_lower = set(sample.str.lower().unique())
    if unique_lower.issubset({"true", "false", "yes", "no", "0", "1", "t", "f", "y", "n"}):
        return "boolean"

    return "string"


def calculate_psi(
    expected: pd.Series, actual: pd.Series, buckets: int = 10
) -> float:
    """
    Calculate Population Stability Index between two distributions.

    Args:
        expected: Expected/reference distribution.
        actual: Actual/current distribution.
        buckets: Number of buckets for binning.

    Returns:
        PSI value.
    """
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Create bins based on expected distribution
    try:
        _, bin_edges = pd.qcut(expected, q=buckets, retbins=True, duplicates="drop")
    except ValueError:
        return 0.0

    # Calculate proportions
    expected_counts = pd.cut(expected, bins=bin_edges, include_lowest=True).value_counts(normalize=True)
    actual_counts = pd.cut(actual, bins=bin_edges, include_lowest=True).value_counts(normalize=True)

    # Align indices
    expected_counts = expected_counts.reindex(actual_counts.index, fill_value=0.0001)
    actual_counts = actual_counts.replace(0, 0.0001)

    # Calculate PSI
    psi = ((actual_counts - expected_counts) * np.log(actual_counts / expected_counts)).sum()

    return float(psi)


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a percentage value for display.

    Args:
        value: Percentage value (0-100).
        decimals: Decimal places.

    Returns:
        Formatted percentage string.
    """
    return f"{value:.{decimals}f}%"


def format_number(value: int | float, abbreviate: bool = True) -> str:
    """
    Format a number for display.

    Args:
        value: Number to format.
        abbreviate: Whether to abbreviate large numbers.

    Returns:
        Formatted number string.
    """
    if not abbreviate:
        return f"{value:,}"

    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return str(value)


def get_date_range_gaps(
    dates: pd.Series,
    start: datetime | None = None,
    end: datetime | None = None,
    frequency: str = "D",
) -> list[tuple[datetime, datetime]]:
    """
    Find gaps in a date range.

    Args:
        dates: Series of dates.
        start: Expected start date.
        end: Expected end date.
        frequency: Expected frequency.

    Returns:
        List of (gap_start, gap_end) tuples.
    """
    dates = pd.to_datetime(dates).dropna().sort_values()

    if len(dates) == 0:
        return []

    start = start or dates.min()
    end = end or dates.max()

    expected = pd.date_range(start=start, end=end, freq=frequency)
    actual = set(dates.dt.normalize())
    missing = sorted(set(expected.normalize()) - actual)

    if not missing:
        return []

    gaps = []
    gap_start = missing[0]
    gap_end = missing[0]

    for i in range(1, len(missing)):
        if (missing[i] - gap_end).days <= 1:
            gap_end = missing[i]
        else:
            gaps.append((gap_start.to_pydatetime(), gap_end.to_pydatetime()))
            gap_start = missing[i]
            gap_end = missing[i]

    gaps.append((gap_start.to_pydatetime(), gap_end.to_pydatetime()))

    return gaps


def validate_schema(
    df: pd.DataFrame, schema: dict[str, type | str]
) -> dict[str, list[str]]:
    """
    Validate a DataFrame against an expected schema.

    Args:
        df: DataFrame to validate.
        schema: Dictionary mapping column names to expected types.

    Returns:
        Dictionary with 'missing', 'extra', and 'type_mismatch' lists.
    """
    result = {
        "missing": [],
        "extra": [],
        "type_mismatch": [],
    }

    expected_columns = set(schema.keys())
    actual_columns = set(df.columns)

    result["missing"] = list(expected_columns - actual_columns)
    result["extra"] = list(actual_columns - expected_columns)

    for col, expected_type in schema.items():
        if col not in df.columns:
            continue

        actual_type = df[col].dtype

        if isinstance(expected_type, str):
            type_match = expected_type.lower() in str(actual_type).lower()
        else:
            type_match = pd.api.types.is_dtype_equal(actual_type, expected_type)

        if not type_match:
            result["type_mismatch"].append({
                "column": col,
                "expected": str(expected_type),
                "actual": str(actual_type),
            })

    return result


def calculate_coverage_metrics(
    actual: set[Any] | Sequence[Any],
    expected: set[Any] | Sequence[Any],
) -> dict[str, Any]:
    """
    Calculate coverage metrics between actual and expected sets.

    Args:
        actual: Actual values.
        expected: Expected values.

    Returns:
        Dictionary with coverage metrics.
    """
    actual_set = set(actual)
    expected_set = set(expected)

    covered = actual_set & expected_set
    missing = expected_set - actual_set
    extra = actual_set - expected_set

    coverage_pct = (
        len(covered) / len(expected_set) * 100
        if expected_set
        else 100.0
    )

    return {
        "covered_count": len(covered),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "expected_count": len(expected_set),
        "actual_count": len(actual_set),
        "coverage_percentage": coverage_pct,
        "covered": list(covered),
        "missing": list(missing),
        "extra": list(extra),
    }

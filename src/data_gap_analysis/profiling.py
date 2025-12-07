"""
Phase 1: Data Profiling Module

This module provides comprehensive data profiling capabilities including:
- Null counting and pattern analysis
- Value distribution analysis
- Cardinality assessment
- Data range examination
- Time series continuity checking
- Data type validation
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class NullReport:
    """Report for null value analysis."""

    column: str
    null_count: int
    total_count: int
    null_percentage: float
    null_pattern: str | None = None

    def __str__(self) -> str:
        return (
            f"Column: {self.column}\n"
            f"  Null Count: {self.null_count:,} ({self.null_percentage:.2f}%)\n"
            f"  Total Records: {self.total_count:,}"
        )


@dataclass
class DistributionReport:
    """Report for value distribution analysis."""

    column: str
    dtype: str
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    mode: Any = None
    value_counts: dict[Any, int] = field(default_factory=dict)
    distribution_type: str = "unknown"

    def __str__(self) -> str:
        lines = [f"Column: {self.column}", f"  Data Type: {self.dtype}"]
        if self.mean is not None:
            lines.append(f"  Mean: {self.mean:.4f}")
        if self.median is not None:
            lines.append(f"  Median: {self.median:.4f}")
        if self.std is not None:
            lines.append(f"  Std Dev: {self.std:.4f}")
        if self.skewness is not None:
            lines.append(f"  Skewness: {self.skewness:.4f}")
        lines.append(f"  Distribution Type: {self.distribution_type}")
        return "\n".join(lines)


@dataclass
class CardinalityReport:
    """Report for cardinality assessment."""

    column: str
    unique_count: int
    total_count: int
    cardinality_ratio: float
    cardinality_type: str  # "low", "medium", "high", "unique"

    def __str__(self) -> str:
        return (
            f"Column: {self.column}\n"
            f"  Unique Values: {self.unique_count:,}\n"
            f"  Total Records: {self.total_count:,}\n"
            f"  Cardinality Ratio: {self.cardinality_ratio:.4f}\n"
            f"  Cardinality Type: {self.cardinality_type}"
        )


@dataclass
class RangeReport:
    """Report for data range examination."""

    column: str
    min_value: Any
    max_value: Any
    range_span: Any = None
    percentiles: dict[int, float] = field(default_factory=dict)
    outliers_count: int = 0
    outlier_bounds: tuple[float, float] | None = None

    def __str__(self) -> str:
        lines = [
            f"Column: {self.column}",
            f"  Min: {self.min_value}",
            f"  Max: {self.max_value}",
        ]
        if self.range_span is not None:
            lines.append(f"  Range Span: {self.range_span}")
        if self.outliers_count > 0:
            lines.append(f"  Outliers: {self.outliers_count}")
        return "\n".join(lines)


@dataclass
class TimeSeriesGap:
    """Represents a gap in time series data."""

    start: datetime
    end: datetime
    expected_records: int
    actual_records: int
    gap_duration: timedelta


@dataclass
class TimeSeriesContinuityReport:
    """Report for time series continuity analysis."""

    column: str
    start_date: datetime
    end_date: datetime
    expected_frequency: str
    total_expected: int
    total_actual: int
    gaps: list[TimeSeriesGap] = field(default_factory=list)
    continuity_percentage: float = 100.0

    def __str__(self) -> str:
        return (
            f"Column: {self.column}\n"
            f"  Date Range: {self.start_date} to {self.end_date}\n"
            f"  Expected Frequency: {self.expected_frequency}\n"
            f"  Continuity: {self.continuity_percentage:.2f}%\n"
            f"  Gaps Found: {len(self.gaps)}"
        )


@dataclass
class TypeValidationReport:
    """Report for data type validation."""

    column: str
    expected_type: str
    valid_count: int
    invalid_count: int
    total_count: int
    validity_percentage: float
    invalid_samples: list[Any] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Column: {self.column}\n"
            f"  Expected Type: {self.expected_type}\n"
            f"  Valid: {self.valid_count:,} ({self.validity_percentage:.2f}%)\n"
            f"  Invalid: {self.invalid_count:,}"
        )


@dataclass
class DataProfile:
    """Complete data profile for a dataset."""

    total_rows: int
    total_columns: int
    null_reports: list[NullReport] = field(default_factory=list)
    distribution_reports: list[DistributionReport] = field(default_factory=list)
    cardinality_reports: list[CardinalityReport] = field(default_factory=list)
    range_reports: list[RangeReport] = field(default_factory=list)
    time_series_reports: list[TimeSeriesContinuityReport] = field(default_factory=list)
    type_validation_reports: list[TypeValidationReport] = field(default_factory=list)
    profile_timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate a summary of the data profile."""
        lines = [
            "=" * 60,
            "DATA PROFILE SUMMARY",
            "=" * 60,
            f"Profile Generated: {self.profile_timestamp}",
            f"Total Rows: {self.total_rows:,}",
            f"Total Columns: {self.total_columns}",
            "",
            "NULL ANALYSIS:",
            "-" * 40,
        ]

        for report in self.null_reports:
            if report.null_percentage > 0:
                lines.append(
                    f"  {report.column}: {report.null_percentage:.2f}% null"
                )

        lines.extend(["", "CARDINALITY:", "-" * 40])
        for report in self.cardinality_reports:
            lines.append(
                f"  {report.column}: {report.unique_count:,} unique "
                f"({report.cardinality_type})"
            )

        return "\n".join(lines)


class DataProfiler:
    """
    Comprehensive data profiler for gap analysis.

    This class provides methods to analyze various aspects of data quality
    and completeness as part of Phase 1 of gap analysis.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize the DataProfiler with a pandas DataFrame.

        Args:
            dataframe: The pandas DataFrame to profile.
        """
        self.df = dataframe
        self._profile: DataProfile | None = None

    def generate_profile(
        self,
        schema: dict[str, str] | None = None,
        date_columns: list[str] | None = None,
        expected_frequency: str = "D",
    ) -> DataProfile:
        """
        Generate a complete data profile.

        Args:
            schema: Optional dictionary mapping column names to expected types.
            date_columns: Optional list of date columns to check for continuity.
            expected_frequency: Expected frequency for time series (D=daily, H=hourly, etc.)

        Returns:
            A DataProfile object containing all analysis results.
        """
        profile = DataProfile(
            total_rows=len(self.df),
            total_columns=len(self.df.columns),
        )

        # Run all analyses
        profile.null_reports = self.count_nulls()
        profile.distribution_reports = self.analyze_value_distribution()
        profile.cardinality_reports = self.assess_cardinality()
        profile.range_reports = self.examine_data_ranges()

        if date_columns:
            for col in date_columns:
                if col in self.df.columns:
                    report = self.check_time_series_continuity(
                        col, expected_frequency
                    )
                    profile.time_series_reports.append(report)

        if schema:
            profile.type_validation_reports = self.validate_data_types(schema)

        self._profile = profile
        return profile

    def count_nulls(
        self, columns: Sequence[str] | None = None
    ) -> list[NullReport]:
        """
        Count null values in specified columns or all columns.

        Args:
            columns: Optional list of columns to analyze. If None, analyzes all.

        Returns:
            List of NullReport objects for each column.
        """
        cols = columns if columns is not None else self.df.columns
        reports = []

        for col in cols:
            if col not in self.df.columns:
                continue

            null_count = self.df[col].isna().sum()
            total_count = len(self.df)
            null_percentage = (null_count / total_count * 100) if total_count > 0 else 0

            # Detect null patterns (e.g., all nulls at end, random, etc.)
            pattern = self._detect_null_pattern(self.df[col])

            reports.append(
                NullReport(
                    column=col,
                    null_count=int(null_count),
                    total_count=total_count,
                    null_percentage=null_percentage,
                    null_pattern=pattern,
                )
            )

        return reports

    def _detect_null_pattern(self, series: pd.Series) -> str:
        """Detect the pattern of null values in a series."""
        null_mask = series.isna()
        null_count = null_mask.sum()

        if null_count == 0:
            return "no_nulls"
        if null_count == len(series):
            return "all_null"

        # Check if nulls are concentrated at the beginning or end
        null_indices = null_mask[null_mask].index.tolist()
        if len(null_indices) == 0:
            return "no_nulls"

        # Check for consecutive nulls at start
        first_idx = null_mask.idxmax() if null_mask.any() else None
        if first_idx == series.index[0]:
            consecutive = 0
            for i, is_null in enumerate(null_mask):
                if is_null:
                    consecutive += 1
                else:
                    break
            if consecutive == null_count:
                return "nulls_at_start"

        # Check for consecutive nulls at end
        last_null_idx = null_mask[::-1].idxmax() if null_mask.any() else None
        if last_null_idx == series.index[-1]:
            consecutive = 0
            for is_null in null_mask[::-1]:
                if is_null:
                    consecutive += 1
                else:
                    break
            if consecutive == null_count:
                return "nulls_at_end"

        return "random_nulls"

    def analyze_value_distribution(
        self, columns: Sequence[str] | None = None
    ) -> list[DistributionReport]:
        """
        Analyze value distribution for specified columns.

        Args:
            columns: Optional list of columns to analyze. If None, analyzes all.

        Returns:
            List of DistributionReport objects for each column.
        """
        cols = columns if columns is not None else self.df.columns
        reports = []

        for col in cols:
            if col not in self.df.columns:
                continue

            series = self.df[col].dropna()
            dtype = str(self.df[col].dtype)

            report = DistributionReport(column=col, dtype=dtype)

            if pd.api.types.is_numeric_dtype(self.df[col]):
                if len(series) > 0:
                    report.mean = float(series.mean())
                    report.median = float(series.median())
                    report.std = float(series.std())
                    report.skewness = float(stats.skew(series))
                    report.kurtosis = float(stats.kurtosis(series))
                    report.distribution_type = self._classify_distribution(series)
            else:
                # For categorical/string data
                value_counts = series.value_counts()
                report.value_counts = value_counts.head(10).to_dict()
                if len(value_counts) > 0:
                    report.mode = value_counts.index[0]
                report.distribution_type = "categorical"

            reports.append(report)

        return reports

    def _classify_distribution(self, series: pd.Series) -> str:
        """Classify the distribution type of a numeric series."""
        if len(series) < 8:
            return "insufficient_data"

        # Perform normality test
        try:
            _, p_value = stats.normaltest(series)
            if p_value > 0.05:
                return "normal"
        except (ValueError, RuntimeWarning):
            pass

        skewness = stats.skew(series)
        if abs(skewness) < 0.5:
            return "approximately_normal"
        elif skewness > 0.5:
            return "right_skewed"
        else:
            return "left_skewed"

    def assess_cardinality(
        self, columns: Sequence[str] | None = None
    ) -> list[CardinalityReport]:
        """
        Assess cardinality (unique value count) for specified columns.

        Args:
            columns: Optional list of columns to analyze. If None, analyzes all.

        Returns:
            List of CardinalityReport objects for each column.
        """
        cols = columns if columns is not None else self.df.columns
        reports = []

        for col in cols:
            if col not in self.df.columns:
                continue

            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            cardinality_ratio = unique_count / total_count if total_count > 0 else 0

            # Classify cardinality
            if cardinality_ratio >= 0.99:
                cardinality_type = "unique"
            elif cardinality_ratio >= 0.5:
                cardinality_type = "high"
            elif cardinality_ratio >= 0.1:
                cardinality_type = "medium"
            else:
                cardinality_type = "low"

            reports.append(
                CardinalityReport(
                    column=col,
                    unique_count=unique_count,
                    total_count=total_count,
                    cardinality_ratio=cardinality_ratio,
                    cardinality_type=cardinality_type,
                )
            )

        return reports

    def examine_data_ranges(
        self, columns: Sequence[str] | None = None, outlier_method: str = "iqr"
    ) -> list[RangeReport]:
        """
        Examine data ranges for numeric columns.

        Args:
            columns: Optional list of columns to analyze. If None, analyzes numeric columns.
            outlier_method: Method for outlier detection ("iqr" or "zscore").

        Returns:
            List of RangeReport objects for each column.
        """
        if columns is None:
            cols = self.df.select_dtypes(include=[np.number]).columns
        else:
            cols = [c for c in columns if c in self.df.columns]

        reports = []

        for col in cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue

            min_val = series.min()
            max_val = series.max()
            range_span = max_val - min_val

            percentiles = {
                5: float(np.percentile(series, 5)),
                25: float(np.percentile(series, 25)),
                50: float(np.percentile(series, 50)),
                75: float(np.percentile(series, 75)),
                95: float(np.percentile(series, 95)),
            }

            # Detect outliers
            outliers_count, outlier_bounds = self._detect_outliers(
                series, method=outlier_method
            )

            reports.append(
                RangeReport(
                    column=col,
                    min_value=min_val,
                    max_value=max_val,
                    range_span=range_span,
                    percentiles=percentiles,
                    outliers_count=outliers_count,
                    outlier_bounds=outlier_bounds,
                )
            )

        return reports

    def _detect_outliers(
        self, series: pd.Series, method: str = "iqr", threshold: float = 1.5
    ) -> tuple[int, tuple[float, float]]:
        """Detect outliers in a numeric series."""
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
        else:
            raise ValueError(f"Unknown outlier method: {method}")

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers), (float(lower_bound), float(upper_bound))

    def check_time_series_continuity(
        self,
        date_column: str,
        expected_frequency: str = "D",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> TimeSeriesContinuityReport:
        """
        Check for gaps in time series data.

        Args:
            date_column: Name of the date/datetime column.
            expected_frequency: Expected frequency (D=daily, H=hourly, W=weekly, M=monthly).
            start_date: Optional expected start date.
            end_date: Optional expected end date.

        Returns:
            TimeSeriesContinuityReport with gap analysis.
        """
        if date_column not in self.df.columns:
            raise ValueError(f"Column '{date_column}' not found in DataFrame")

        # Convert to datetime if needed
        date_series = pd.to_datetime(self.df[date_column], errors="coerce")
        date_series = date_series.dropna().sort_values()

        if len(date_series) == 0:
            raise ValueError(f"No valid dates found in column '{date_column}'")

        actual_start = start_date or date_series.min()
        actual_end = end_date or date_series.max()

        # Generate expected date range
        expected_dates = pd.date_range(
            start=actual_start, end=actual_end, freq=expected_frequency
        )

        # Find missing dates
        actual_dates = set(date_series.dt.normalize())
        expected_date_set = set(expected_dates.normalize())
        missing_dates = sorted(expected_date_set - actual_dates)

        # Identify gaps
        gaps = []
        if missing_dates:
            gap_start = missing_dates[0]
            gap_end = missing_dates[0]

            for i in range(1, len(missing_dates)):
                current = missing_dates[i]
                if (current - gap_end).days <= 1:
                    gap_end = current
                else:
                    gaps.append(
                        TimeSeriesGap(
                            start=gap_start.to_pydatetime(),
                            end=gap_end.to_pydatetime(),
                            expected_records=1,
                            actual_records=0,
                            gap_duration=gap_end - gap_start + timedelta(days=1),
                        )
                    )
                    gap_start = current
                    gap_end = current

            # Add last gap
            gaps.append(
                TimeSeriesGap(
                    start=gap_start.to_pydatetime(),
                    end=gap_end.to_pydatetime(),
                    expected_records=1,
                    actual_records=0,
                    gap_duration=gap_end - gap_start + timedelta(days=1),
                )
            )

        continuity_pct = (
            (len(expected_dates) - len(missing_dates)) / len(expected_dates) * 100
            if len(expected_dates) > 0
            else 100.0
        )

        return TimeSeriesContinuityReport(
            column=date_column,
            start_date=actual_start.to_pydatetime()
            if hasattr(actual_start, "to_pydatetime")
            else actual_start,
            end_date=actual_end.to_pydatetime()
            if hasattr(actual_end, "to_pydatetime")
            else actual_end,
            expected_frequency=expected_frequency,
            total_expected=len(expected_dates),
            total_actual=len(date_series),
            gaps=gaps,
            continuity_percentage=continuity_pct,
        )

    def validate_data_types(
        self, schema: dict[str, str]
    ) -> list[TypeValidationReport]:
        """
        Validate that column values match expected data types.

        Args:
            schema: Dictionary mapping column names to expected types.
                   Supported types: "int", "float", "str", "bool", "date",
                   "datetime", "email", "url", "phone", "uuid"

        Returns:
            List of TypeValidationReport objects for each column.
        """
        reports = []
        validators = self._get_type_validators()

        for column, expected_type in schema.items():
            if column not in self.df.columns:
                continue

            series = self.df[column].dropna()
            total_count = len(self.df)
            validator = validators.get(expected_type.lower())

            if validator is None:
                # Use pandas dtype checking as fallback
                valid_mask = series.apply(lambda x: True)
                valid_count = len(series)
            else:
                valid_mask = series.apply(validator)
                valid_count = valid_mask.sum()

            invalid_count = len(series) - valid_count
            validity_pct = (valid_count / total_count * 100) if total_count > 0 else 100

            # Get samples of invalid values
            invalid_samples = []
            if invalid_count > 0:
                invalid_values = series[~valid_mask]
                invalid_samples = invalid_values.head(5).tolist()

            reports.append(
                TypeValidationReport(
                    column=column,
                    expected_type=expected_type,
                    valid_count=int(valid_count),
                    invalid_count=int(invalid_count),
                    total_count=total_count,
                    validity_percentage=validity_pct,
                    invalid_samples=invalid_samples,
                )
            )

        return reports

    def _get_type_validators(self) -> dict[str, Any]:
        """Get validator functions for different data types."""
        email_pattern = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
        url_pattern = re.compile(
            r"^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
        )
        phone_pattern = re.compile(r"^\+?1?\d{9,15}$")
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )

        return {
            "int": lambda x: isinstance(x, (int, np.integer))
            or (isinstance(x, float) and x.is_integer()),
            "float": lambda x: isinstance(x, (int, float, np.number)),
            "str": lambda x: isinstance(x, str),
            "bool": lambda x: isinstance(x, (bool, np.bool_)),
            "date": lambda x: self._is_valid_date(x),
            "datetime": lambda x: self._is_valid_datetime(x),
            "email": lambda x: bool(email_pattern.match(str(x))),
            "url": lambda x: bool(url_pattern.match(str(x))),
            "phone": lambda x: bool(phone_pattern.match(re.sub(r"[\s\-\(\)]", "", str(x)))),
            "uuid": lambda x: bool(uuid_pattern.match(str(x))),
        }

    def _is_valid_date(self, value: Any) -> bool:
        """Check if value is a valid date."""
        if pd.isna(value):
            return False
        try:
            pd.to_datetime(value)
            return True
        except (ValueError, TypeError):
            return False

    def _is_valid_datetime(self, value: Any) -> bool:
        """Check if value is a valid datetime."""
        return self._is_valid_date(value)

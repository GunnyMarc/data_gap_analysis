"""
Phase 3: Quality Gap Identification Module

This module identifies data quality issues including:
- Inconsistent formats or units
- Duplicate records
- Outliers that indicate errors
- Referential integrity issues
- Stale data that hasn't been updated
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FormatIssue:
    """Represents a format inconsistency in a column."""

    column: str
    detected_formats: dict[str, int]
    dominant_format: str
    inconsistency_percentage: float
    sample_values: dict[str, list[Any]] = field(default_factory=dict)

    def __str__(self) -> str:
        formats_str = ", ".join(
            f"{fmt}: {count}" for fmt, count in self.detected_formats.items()
        )
        return (
            f"Column: {self.column}\n"
            f"  Detected Formats: {formats_str}\n"
            f"  Dominant Format: {self.dominant_format}\n"
            f"  Inconsistency: {self.inconsistency_percentage:.2f}%"
        )


@dataclass
class DuplicateReport:
    """Report for duplicate detection."""

    key_columns: list[str]
    exact_duplicate_count: int
    exact_duplicate_groups: int
    fuzzy_duplicate_count: int
    fuzzy_duplicate_groups: int
    duplicate_records: pd.DataFrame | None = None
    duplicate_percentage: float = 0.0

    def __str__(self) -> str:
        return (
            f"Duplicate Report:\n"
            f"  Key Columns: {self.key_columns}\n"
            f"  Exact Duplicates: {self.exact_duplicate_count} records "
            f"in {self.exact_duplicate_groups} groups\n"
            f"  Fuzzy Duplicates: {self.fuzzy_duplicate_count} records "
            f"in {self.fuzzy_duplicate_groups} groups\n"
            f"  Duplicate Percentage: {self.duplicate_percentage:.2f}%"
        )


@dataclass
class OutlierReport:
    """Report for outlier detection."""

    column: str
    method: str
    outlier_count: int
    total_count: int
    outlier_percentage: float
    lower_bound: float
    upper_bound: float
    outlier_indices: list[int] = field(default_factory=list)
    outlier_values: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Column: {self.column}\n"
            f"  Method: {self.method}\n"
            f"  Outliers: {self.outlier_count} ({self.outlier_percentage:.2f}%)\n"
            f"  Bounds: [{self.lower_bound:.4f}, {self.upper_bound:.4f}]"
        )


@dataclass
class ReferentialIntegrityIssue:
    """Report for referential integrity issues."""

    foreign_key_column: str
    reference_table: str
    reference_key_column: str
    orphaned_count: int
    total_count: int
    orphaned_percentage: float
    orphaned_values: list[Any] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Referential Integrity Issue:\n"
            f"  Foreign Key: {self.foreign_key_column}\n"
            f"  References: {self.reference_table}.{self.reference_key_column}\n"
            f"  Orphaned Records: {self.orphaned_count} ({self.orphaned_percentage:.2f}%)"
        )


@dataclass
class StaleDataReport:
    """Report for stale data detection."""

    timestamp_column: str
    staleness_threshold_days: int
    stale_count: int
    total_count: int
    stale_percentage: float
    oldest_record_date: datetime | None = None
    newest_record_date: datetime | None = None
    stale_record_indices: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Stale Data Report:\n"
            f"  Timestamp Column: {self.timestamp_column}\n"
            f"  Threshold: {self.staleness_threshold_days} days\n"
            f"  Stale Records: {self.stale_count} ({self.stale_percentage:.2f}%)\n"
            f"  Date Range: {self.oldest_record_date} to {self.newest_record_date}"
        )


@dataclass
class QualityReport:
    """Complete quality gap analysis report."""

    format_issues: list[FormatIssue] = field(default_factory=list)
    duplicate_reports: list[DuplicateReport] = field(default_factory=list)
    outlier_reports: list[OutlierReport] = field(default_factory=list)
    referential_integrity_issues: list[ReferentialIntegrityIssue] = field(
        default_factory=list
    )
    stale_data_reports: list[StaleDataReport] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate a summary of quality issues."""
        lines = [
            "=" * 60,
            "QUALITY GAP ANALYSIS SUMMARY",
            "=" * 60,
            f"Analysis Timestamp: {self.analysis_timestamp}",
            "",
        ]

        if self.format_issues:
            lines.append("FORMAT INCONSISTENCIES:")
            lines.append("-" * 40)
            for issue in self.format_issues:
                lines.append(f"  {issue.column}: {len(issue.detected_formats)} formats")
            lines.append("")

        if self.duplicate_reports:
            lines.append("DUPLICATE RECORDS:")
            lines.append("-" * 40)
            for report in self.duplicate_reports:
                lines.append(
                    f"  Exact: {report.exact_duplicate_count}, "
                    f"Fuzzy: {report.fuzzy_duplicate_count}"
                )
            lines.append("")

        if self.outlier_reports:
            lines.append("OUTLIERS:")
            lines.append("-" * 40)
            for report in self.outlier_reports:
                lines.append(
                    f"  {report.column}: {report.outlier_count} outliers "
                    f"({report.outlier_percentage:.2f}%)"
                )
            lines.append("")

        if self.referential_integrity_issues:
            lines.append("REFERENTIAL INTEGRITY ISSUES:")
            lines.append("-" * 40)
            for issue in self.referential_integrity_issues:
                lines.append(
                    f"  {issue.foreign_key_column}: "
                    f"{issue.orphaned_count} orphaned records"
                )
            lines.append("")

        if self.stale_data_reports:
            lines.append("STALE DATA:")
            lines.append("-" * 40)
            for report in self.stale_data_reports:
                lines.append(
                    f"  {report.timestamp_column}: {report.stale_count} stale records "
                    f"({report.stale_percentage:.2f}%)"
                )

        return "\n".join(lines)


class QualityAnalyzer:
    """
    Analyzer for data quality gap identification.

    This class provides methods to detect various data quality issues
    as part of Phase 3 of gap analysis.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize the QualityAnalyzer with a pandas DataFrame.

        Args:
            dataframe: The pandas DataFrame to analyze.
        """
        self.df = dataframe

    def run_full_analysis(
        self,
        format_columns: list[str] | None = None,
        duplicate_key_columns: list[str] | None = None,
        numeric_columns: list[str] | None = None,
        foreign_key_configs: list[dict[str, Any]] | None = None,
        timestamp_column: str | None = None,
        staleness_threshold_days: int = 90,
    ) -> QualityReport:
        """
        Run complete quality gap analysis.

        Args:
            format_columns: Columns to check for format inconsistencies.
            duplicate_key_columns: Key columns for duplicate detection.
            numeric_columns: Columns for outlier detection.
            foreign_key_configs: List of dicts with foreign key configurations.
            timestamp_column: Column for stale data detection.
            staleness_threshold_days: Days threshold for stale data.

        Returns:
            QualityReport with all quality issues.
        """
        report = QualityReport()

        if format_columns:
            report.format_issues = self.detect_inconsistent_formats(format_columns)

        if duplicate_key_columns:
            dup_report = self.find_duplicates(duplicate_key_columns)
            report.duplicate_reports.append(dup_report)

        if numeric_columns:
            report.outlier_reports = self.detect_outliers(numeric_columns)

        if foreign_key_configs:
            for config in foreign_key_configs:
                issue = self.check_referential_integrity(
                    config["foreign_key"],
                    config["reference_df"],
                    config["reference_key"],
                )
                report.referential_integrity_issues.append(issue)

        if timestamp_column:
            stale_report = self.detect_stale_data(
                timestamp_column, staleness_threshold_days
            )
            report.stale_data_reports.append(stale_report)

        return report

    def detect_inconsistent_formats(
        self, columns: list[str]
    ) -> list[FormatIssue]:
        """
        Detect format inconsistencies in specified columns.

        Analyzes columns for mixed formats in dates, phone numbers,
        currencies, and other common patterns.

        Args:
            columns: List of columns to analyze.

        Returns:
            List of FormatIssue objects for columns with inconsistencies.
        """
        issues = []

        for col in columns:
            if col not in self.df.columns:
                continue

            series = self.df[col].dropna().astype(str)
            if len(series) == 0:
                continue

            # Detect format patterns
            format_counts = self._detect_format_patterns(series, col)

            if len(format_counts) > 1:
                total = sum(format_counts.values())
                dominant = max(format_counts, key=format_counts.get)
                dominant_count = format_counts[dominant]
                inconsistency = ((total - dominant_count) / total) * 100

                # Get sample values for each format
                samples = {}
                for fmt in format_counts:
                    matching = series[
                        series.apply(lambda x: self._classify_format(x, col) == fmt)
                    ]
                    samples[fmt] = matching.head(3).tolist()

                issues.append(
                    FormatIssue(
                        column=col,
                        detected_formats=format_counts,
                        dominant_format=dominant,
                        inconsistency_percentage=inconsistency,
                        sample_values=samples,
                    )
                )

        return issues

    def _detect_format_patterns(
        self, series: pd.Series, column_name: str
    ) -> dict[str, int]:
        """Detect format patterns in a series."""
        format_counts: Counter[str] = Counter()

        for value in series:
            fmt = self._classify_format(str(value), column_name)
            format_counts[fmt] += 1

        return dict(format_counts)

    def _classify_format(self, value: str, column_name: str) -> str:
        """Classify the format of a value based on column context."""
        value = value.strip()

        # Date format detection
        if "date" in column_name.lower() or "time" in column_name.lower():
            date_patterns = {
                "YYYY-MM-DD": r"^\d{4}-\d{2}-\d{2}$",
                "MM/DD/YYYY": r"^\d{1,2}/\d{1,2}/\d{4}$",
                "DD-MM-YYYY": r"^\d{2}-\d{2}-\d{4}$",
                "DD/MM/YYYY": r"^\d{2}/\d{2}/\d{4}$",
                "YYYY/MM/DD": r"^\d{4}/\d{2}/\d{2}$",
                "ISO8601": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
                "TIMESTAMP": r"^\d{10,13}$",
            }
            for fmt_name, pattern in date_patterns.items():
                if re.match(pattern, value):
                    return fmt_name
            return "OTHER_DATE"

        # Phone format detection
        if "phone" in column_name.lower() or "tel" in column_name.lower():
            phone_patterns = {
                "E164": r"^\+\d{10,15}$",
                "US_DASHES": r"^\d{3}-\d{3}-\d{4}$",
                "US_PARENS": r"^\(\d{3}\)\s*\d{3}-\d{4}$",
                "US_DOTS": r"^\d{3}\.\d{3}\.\d{4}$",
                "PLAIN": r"^\d{10,11}$",
            }
            for fmt_name, pattern in phone_patterns.items():
                if re.match(pattern, value):
                    return fmt_name
            return "OTHER_PHONE"

        # Currency format detection
        if any(c in column_name.lower() for c in ["price", "amount", "cost", "currency"]):
            currency_patterns = {
                "USD_PREFIX": r"^\$[\d,]+\.?\d*$",
                "EUR_PREFIX": r"^€[\d,]+\.?\d*$",
                "GBP_PREFIX": r"^£[\d,]+\.?\d*$",
                "PLAIN_DECIMAL": r"^[\d,]+\.\d{2}$",
                "PLAIN_INT": r"^\d+$",
            }
            for fmt_name, pattern in currency_patterns.items():
                if re.match(pattern, value):
                    return fmt_name
            return "OTHER_CURRENCY"

        # Generic format classification
        if re.match(r"^[\d,]+\.?\d*$", value):
            return "NUMERIC"
        elif re.match(r"^[A-Z]+$", value):
            return "UPPERCASE"
        elif re.match(r"^[a-z]+$", value):
            return "LOWERCASE"
        elif re.match(r"^[A-Za-z]+$", value):
            return "MIXED_CASE"
        else:
            return "MIXED"

    def find_duplicates(
        self,
        key_columns: list[str],
        fuzzy_columns: list[str] | None = None,
        similarity_threshold: float = 0.85,
    ) -> DuplicateReport:
        """
        Find exact and fuzzy duplicate records.

        Args:
            key_columns: Columns to use for exact duplicate detection.
            fuzzy_columns: Columns to use for fuzzy matching.
            similarity_threshold: Threshold for fuzzy matching (0-1).

        Returns:
            DuplicateReport with duplicate information.
        """
        # Find exact duplicates
        exact_mask = self.df.duplicated(subset=key_columns, keep=False)
        exact_duplicates = self.df[exact_mask]
        exact_groups = self.df[exact_mask].groupby(key_columns).ngroups

        # Fuzzy duplicate detection
        fuzzy_count = 0
        fuzzy_groups = 0

        if fuzzy_columns:
            # Use simple string similarity for fuzzy matching
            fuzzy_count, fuzzy_groups = self._find_fuzzy_duplicates(
                fuzzy_columns, similarity_threshold
            )

        total = len(self.df)
        duplicate_pct = (
            (len(exact_duplicates) + fuzzy_count) / total * 100 if total > 0 else 0
        )

        return DuplicateReport(
            key_columns=key_columns,
            exact_duplicate_count=len(exact_duplicates),
            exact_duplicate_groups=exact_groups,
            fuzzy_duplicate_count=fuzzy_count,
            fuzzy_duplicate_groups=fuzzy_groups,
            duplicate_records=exact_duplicates if len(exact_duplicates) < 1000 else None,
            duplicate_percentage=duplicate_pct,
        )

    def _find_fuzzy_duplicates(
        self, columns: list[str], threshold: float
    ) -> tuple[int, int]:
        """Find fuzzy duplicates using string similarity."""
        # Simple implementation using exact lowercase matching
        # For production, consider using libraries like fuzzywuzzy or rapidfuzz
        concat_col = self.df[columns].fillna("").astype(str).agg(" ".join, axis=1)
        normalized = concat_col.str.lower().str.strip()

        duplicates = normalized.duplicated(keep=False)
        groups = normalized[duplicates].nunique()

        return int(duplicates.sum()), int(groups)

    def detect_outliers(
        self,
        numeric_columns: list[str] | None = None,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> list[OutlierReport]:
        """
        Detect outliers in numeric columns.

        Args:
            numeric_columns: Columns to analyze (all numeric if None).
            method: Detection method ("iqr", "zscore", "isolation_forest").
            threshold: Threshold for outlier detection.

        Returns:
            List of OutlierReport objects.
        """
        if numeric_columns is None:
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        reports = []

        for col in numeric_columns:
            if col not in self.df.columns:
                continue

            series = self.df[col].dropna()
            if len(series) < 3:
                continue

            if method == "iqr":
                outlier_mask, bounds = self._detect_iqr_outliers(series, threshold)
            elif method == "zscore":
                outlier_mask, bounds = self._detect_zscore_outliers(series, threshold)
            elif method == "isolation_forest":
                outlier_mask, bounds = self._detect_isolation_forest_outliers(series)
            else:
                raise ValueError(f"Unknown outlier method: {method}")

            outlier_indices = series[outlier_mask].index.tolist()
            outlier_values = series[outlier_mask].tolist()

            reports.append(
                OutlierReport(
                    column=col,
                    method=method,
                    outlier_count=len(outlier_indices),
                    total_count=len(series),
                    outlier_percentage=len(outlier_indices) / len(series) * 100,
                    lower_bound=bounds[0],
                    upper_bound=bounds[1],
                    outlier_indices=outlier_indices[:100],
                    outlier_values=outlier_values[:100],
                )
            )

        return reports

    def _detect_iqr_outliers(
        self, series: pd.Series, threshold: float
    ) -> tuple[pd.Series, tuple[float, float]]:
        """Detect outliers using IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr

        mask = (series < lower) | (series > upper)
        return mask, (float(lower), float(upper))

    def _detect_zscore_outliers(
        self, series: pd.Series, threshold: float
    ) -> tuple[pd.Series, tuple[float, float]]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series))
        mask = pd.Series(z_scores > threshold, index=series.index)

        mean = series.mean()
        std = series.std()
        lower = mean - threshold * std
        upper = mean + threshold * std

        return mask, (float(lower), float(upper))

    def _detect_isolation_forest_outliers(
        self, series: pd.Series
    ) -> tuple[pd.Series, tuple[float, float]]:
        """Detect outliers using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(series.values.reshape(-1, 1))
            mask = pd.Series(predictions == -1, index=series.index)

            # Use IQR bounds for reporting
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            return mask, (float(lower), float(upper))
        except ImportError:
            # Fall back to IQR if sklearn not available
            return self._detect_iqr_outliers(series, 1.5)

    def check_referential_integrity(
        self,
        foreign_key: str,
        reference_df: pd.DataFrame,
        reference_key: str,
    ) -> ReferentialIntegrityIssue:
        """
        Check for referential integrity issues.

        Args:
            foreign_key: Column in current DataFrame that references another table.
            reference_df: Reference DataFrame containing the primary key.
            reference_key: Column in reference DataFrame that is referenced.

        Returns:
            ReferentialIntegrityIssue with orphaned record information.
        """
        if foreign_key not in self.df.columns:
            raise ValueError(f"Column '{foreign_key}' not found in DataFrame")
        if reference_key not in reference_df.columns:
            raise ValueError(f"Column '{reference_key}' not found in reference DataFrame")

        foreign_values = set(self.df[foreign_key].dropna().unique())
        reference_values = set(reference_df[reference_key].dropna().unique())

        orphaned_values = foreign_values - reference_values
        orphaned_mask = self.df[foreign_key].isin(orphaned_values)
        orphaned_count = orphaned_mask.sum()

        total = len(self.df[foreign_key].dropna())
        orphaned_pct = (orphaned_count / total * 100) if total > 0 else 0

        return ReferentialIntegrityIssue(
            foreign_key_column=foreign_key,
            reference_table="reference_df",
            reference_key_column=reference_key,
            orphaned_count=int(orphaned_count),
            total_count=total,
            orphaned_percentage=orphaned_pct,
            orphaned_values=list(orphaned_values)[:100],
        )

    def detect_stale_data(
        self,
        timestamp_column: str,
        staleness_threshold_days: int = 90,
        reference_date: datetime | None = None,
    ) -> StaleDataReport:
        """
        Detect records that haven't been updated recently.

        Args:
            timestamp_column: Column containing last update timestamps.
            staleness_threshold_days: Days threshold for considering data stale.
            reference_date: Reference date for comparison (now if None).

        Returns:
            StaleDataReport with stale record information.
        """
        if timestamp_column not in self.df.columns:
            raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")

        ref_date = reference_date or datetime.now()
        threshold_date = ref_date - timedelta(days=staleness_threshold_days)

        # Convert to datetime
        date_series = pd.to_datetime(self.df[timestamp_column], errors="coerce")
        valid_dates = date_series.dropna()

        if len(valid_dates) == 0:
            return StaleDataReport(
                timestamp_column=timestamp_column,
                staleness_threshold_days=staleness_threshold_days,
                stale_count=0,
                total_count=len(self.df),
                stale_percentage=0.0,
            )

        stale_mask = valid_dates < threshold_date
        stale_count = stale_mask.sum()
        stale_indices = valid_dates[stale_mask].index.tolist()

        total = len(valid_dates)
        stale_pct = (stale_count / total * 100) if total > 0 else 0

        return StaleDataReport(
            timestamp_column=timestamp_column,
            staleness_threshold_days=staleness_threshold_days,
            stale_count=int(stale_count),
            total_count=total,
            stale_percentage=stale_pct,
            oldest_record_date=valid_dates.min().to_pydatetime(),
            newest_record_date=valid_dates.max().to_pydatetime(),
            stale_record_indices=stale_indices[:1000],
        )

    def detect_unit_inconsistencies(
        self, column: str, expected_unit: str | None = None
    ) -> dict[str, Any]:
        """
        Detect unit inconsistencies in a column.

        Args:
            column: Column to analyze.
            expected_unit: Expected unit (for validation).

        Returns:
            Dictionary with unit analysis results.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        series = self.df[column].dropna().astype(str)

        # Extract potential units
        unit_pattern = r"([a-zA-Z%°]+)$"
        units = series.str.extract(unit_pattern, expand=False).dropna()

        unit_counts = units.value_counts().to_dict()

        return {
            "column": column,
            "detected_units": unit_counts,
            "expected_unit": expected_unit,
            "unit_count": len(unit_counts),
            "is_consistent": len(unit_counts) <= 1,
            "has_expected_unit": expected_unit in unit_counts if expected_unit else None,
        }

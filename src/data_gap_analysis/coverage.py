"""
Phase 2: Data Coverage Comparison Module

This module analyzes data coverage across multiple dimensions:
- Geographic coverage
- Temporal coverage
- Entity coverage
- Attribute coverage
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GeographicCoverageReport:
    """Report for geographic coverage analysis."""

    location_column: str
    expected_regions: list[str]
    covered_regions: list[str]
    missing_regions: list[str]
    extra_regions: list[str]
    coverage_percentage: float
    region_record_counts: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Geographic Coverage Report:\n"
            f"  Location Column: {self.location_column}\n"
            f"  Expected Regions: {len(self.expected_regions)}\n"
            f"  Covered Regions: {len(self.covered_regions)}\n"
            f"  Missing Regions: {self.missing_regions}\n"
            f"  Coverage: {self.coverage_percentage:.2f}%"
        )


@dataclass
class TemporalCoverageReport:
    """Report for temporal coverage analysis."""

    date_column: str
    expected_start: datetime
    expected_end: datetime
    actual_start: datetime
    actual_end: datetime
    expected_frequency: str
    total_expected_periods: int
    total_actual_periods: int
    missing_periods: list[datetime] = field(default_factory=list)
    coverage_percentage: float = 0.0
    gaps_at_start_days: int = 0
    gaps_at_end_days: int = 0
    gaps_within_range: int = 0

    def __str__(self) -> str:
        return (
            f"Temporal Coverage Report:\n"
            f"  Date Column: {self.date_column}\n"
            f"  Expected Range: {self.expected_start.date()} to {self.expected_end.date()}\n"
            f"  Actual Range: {self.actual_start.date()} to {self.actual_end.date()}\n"
            f"  Coverage: {self.coverage_percentage:.2f}%\n"
            f"  Gaps at Start: {self.gaps_at_start_days} days\n"
            f"  Gaps at End: {self.gaps_at_end_days} days\n"
            f"  Gaps Within Range: {self.gaps_within_range} periods"
        )


@dataclass
class EntityCoverageReport:
    """Report for entity coverage analysis."""

    entity_column: str
    expected_entities: int
    covered_entities: int
    missing_entities: list[Any]
    extra_entities: list[Any]
    coverage_percentage: float
    entity_record_counts: dict[Any, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Entity Coverage Report:\n"
            f"  Entity Column: {self.entity_column}\n"
            f"  Expected Entities: {self.expected_entities}\n"
            f"  Covered Entities: {self.covered_entities}\n"
            f"  Missing Entities: {len(self.missing_entities)}\n"
            f"  Coverage: {self.coverage_percentage:.2f}%"
        )


@dataclass
class AttributeStatus:
    """Status of a single attribute."""

    name: str
    requirement_level: str  # "required", "recommended", "optional"
    present: bool
    non_null_count: int
    total_count: int
    completeness_percentage: float


@dataclass
class AttributeCoverageReport:
    """Report for attribute coverage analysis."""

    required_columns: list[str]
    recommended_columns: list[str]
    optional_columns: list[str]
    present_required: list[str]
    missing_required: list[str]
    present_recommended: list[str]
    missing_recommended: list[str]
    attribute_details: list[AttributeStatus] = field(default_factory=list)
    required_coverage_percentage: float = 0.0
    recommended_coverage_percentage: float = 0.0
    overall_coverage_percentage: float = 0.0

    def __str__(self) -> str:
        return (
            f"Attribute Coverage Report:\n"
            f"  Required Columns:\n"
            f"    Present: {self.present_required}\n"
            f"    Missing: {self.missing_required}\n"
            f"    Coverage: {self.required_coverage_percentage:.2f}%\n"
            f"  Recommended Columns:\n"
            f"    Present: {self.present_recommended}\n"
            f"    Missing: {self.missing_recommended}\n"
            f"    Coverage: {self.recommended_coverage_percentage:.2f}%\n"
            f"  Overall Schema Coverage: {self.overall_coverage_percentage:.2f}%"
        )


@dataclass
class CoverageReport:
    """Complete coverage analysis report."""

    geographic_coverage: GeographicCoverageReport | None = None
    temporal_coverage: TemporalCoverageReport | None = None
    entity_coverage: EntityCoverageReport | None = None
    attribute_coverage: AttributeCoverageReport | None = None
    analysis_timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate a summary of coverage analysis."""
        lines = [
            "=" * 60,
            "COVERAGE ANALYSIS SUMMARY",
            "=" * 60,
            f"Analysis Timestamp: {self.analysis_timestamp}",
            "",
        ]

        if self.geographic_coverage:
            lines.append(str(self.geographic_coverage))
            lines.append("")

        if self.temporal_coverage:
            lines.append(str(self.temporal_coverage))
            lines.append("")

        if self.entity_coverage:
            lines.append(str(self.entity_coverage))
            lines.append("")

        if self.attribute_coverage:
            lines.append(str(self.attribute_coverage))

        return "\n".join(lines)


class CoverageAnalyzer:
    """
    Analyzer for data coverage across multiple dimensions.

    This class provides methods to analyze how well your data represents
    the full scope of what should be covered as part of Phase 2 of gap analysis.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize the CoverageAnalyzer with a pandas DataFrame.

        Args:
            dataframe: The pandas DataFrame to analyze.
        """
        self.df = dataframe

    def analyze_all_coverage(
        self,
        location_column: str | None = None,
        expected_regions: list[str] | None = None,
        date_column: str | None = None,
        expected_start: str | datetime | None = None,
        expected_end: str | datetime | None = None,
        expected_frequency: str = "D",
        entity_column: str | None = None,
        reference_entities: Sequence[Any] | None = None,
        required_columns: list[str] | None = None,
        recommended_columns: list[str] | None = None,
        optional_columns: list[str] | None = None,
    ) -> CoverageReport:
        """
        Run all coverage analyses.

        Args:
            location_column: Column containing geographic data.
            expected_regions: List of expected geographic regions.
            date_column: Column containing date/datetime data.
            expected_start: Expected start date for temporal coverage.
            expected_end: Expected end date for temporal coverage.
            expected_frequency: Expected frequency (D=daily, H=hourly, etc.)
            entity_column: Column containing entity identifiers.
            reference_entities: List of expected entities.
            required_columns: List of required columns for schema.
            recommended_columns: List of recommended columns for schema.
            optional_columns: List of optional columns for schema.

        Returns:
            CoverageReport with all analysis results.
        """
        report = CoverageReport()

        if location_column and expected_regions:
            report.geographic_coverage = self.analyze_geographic_coverage(
                location_column, expected_regions
            )

        if date_column:
            report.temporal_coverage = self.analyze_temporal_coverage(
                date_column, expected_start, expected_end, expected_frequency
            )

        if entity_column and reference_entities is not None:
            report.entity_coverage = self.analyze_entity_coverage(
                entity_column, reference_entities
            )

        if required_columns or recommended_columns:
            report.attribute_coverage = self.analyze_attribute_coverage(
                required_columns or [],
                recommended_columns or [],
                optional_columns or [],
            )

        return report

    def analyze_geographic_coverage(
        self,
        location_column: str,
        expected_regions: list[str],
        normalize: bool = True,
    ) -> GeographicCoverageReport:
        """
        Analyze geographic coverage against expected regions.

        Args:
            location_column: Column containing geographic data (country, state, etc.)
            expected_regions: List of expected geographic regions.
            normalize: Whether to normalize region names for comparison.

        Returns:
            GeographicCoverageReport with analysis results.
        """
        if location_column not in self.df.columns:
            raise ValueError(f"Column '{location_column}' not found in DataFrame")

        # Get unique regions in data
        actual_regions = self.df[location_column].dropna().unique()

        if normalize:
            actual_set = {str(r).strip().upper() for r in actual_regions}
            expected_set = {str(r).strip().upper() for r in expected_regions}
        else:
            actual_set = set(actual_regions)
            expected_set = set(expected_regions)

        covered = actual_set & expected_set
        missing = expected_set - actual_set
        extra = actual_set - expected_set

        coverage_pct = (
            len(covered) / len(expected_set) * 100
            if len(expected_set) > 0
            else 100.0
        )

        # Get record counts per region
        region_counts = self.df[location_column].value_counts().to_dict()

        return GeographicCoverageReport(
            location_column=location_column,
            expected_regions=list(expected_set),
            covered_regions=list(covered),
            missing_regions=list(missing),
            extra_regions=list(extra),
            coverage_percentage=coverage_pct,
            region_record_counts=region_counts,
        )

    def analyze_temporal_coverage(
        self,
        date_column: str,
        expected_start: str | datetime | None = None,
        expected_end: str | datetime | None = None,
        expected_frequency: str = "D",
    ) -> TemporalCoverageReport:
        """
        Analyze temporal coverage for date range completeness.

        Args:
            date_column: Column containing date/datetime data.
            expected_start: Expected start date (uses min date if None).
            expected_end: Expected end date (uses max date if None).
            expected_frequency: Expected frequency (D=daily, H=hourly, W=weekly, M=monthly).

        Returns:
            TemporalCoverageReport with analysis results.
        """
        if date_column not in self.df.columns:
            raise ValueError(f"Column '{date_column}' not found in DataFrame")

        # Convert to datetime
        date_series = pd.to_datetime(self.df[date_column], errors="coerce")
        date_series = date_series.dropna()

        if len(date_series) == 0:
            raise ValueError(f"No valid dates found in column '{date_column}'")

        actual_start = date_series.min()
        actual_end = date_series.max()

        # Handle expected dates
        if expected_start is None:
            exp_start = actual_start
        else:
            exp_start = pd.to_datetime(expected_start)

        if expected_end is None:
            exp_end = actual_end
        else:
            exp_end = pd.to_datetime(expected_end)

        # Generate expected date range
        expected_dates = pd.date_range(
            start=exp_start, end=exp_end, freq=expected_frequency
        )

        # Normalize dates for comparison based on frequency
        if expected_frequency in ("D", "B"):
            actual_dates = set(date_series.dt.normalize())
            expected_set = set(expected_dates.normalize())
        elif expected_frequency == "H":
            actual_dates = set(date_series.dt.floor("h"))
            expected_set = set(expected_dates)
        elif expected_frequency in ("W", "W-SUN", "W-MON"):
            actual_dates = set(date_series.dt.to_period("W").dt.start_time)
            expected_set = set(expected_dates)
        elif expected_frequency in ("M", "MS", "ME"):
            actual_dates = set(date_series.dt.to_period("M").dt.start_time)
            expected_set = set(expected_dates)
        else:
            actual_dates = set(date_series.dt.normalize())
            expected_set = set(expected_dates.normalize())

        # Find missing periods
        missing_periods = sorted(expected_set - actual_dates)

        # Calculate gaps at start and end
        gaps_at_start = 0
        gaps_at_end = 0
        gaps_within = 0

        if actual_start > exp_start:
            gaps_at_start = (actual_start - exp_start).days

        if actual_end < exp_end:
            gaps_at_end = (exp_end - actual_end).days

        # Count gaps within the actual data range
        within_range_missing = [
            d for d in missing_periods if actual_start <= d <= actual_end
        ]
        gaps_within = len(within_range_missing)

        coverage_pct = (
            (len(expected_set) - len(missing_periods)) / len(expected_set) * 100
            if len(expected_set) > 0
            else 100.0
        )

        return TemporalCoverageReport(
            date_column=date_column,
            expected_start=exp_start.to_pydatetime(),
            expected_end=exp_end.to_pydatetime(),
            actual_start=actual_start.to_pydatetime(),
            actual_end=actual_end.to_pydatetime(),
            expected_frequency=expected_frequency,
            total_expected_periods=len(expected_set),
            total_actual_periods=len(actual_dates),
            missing_periods=[
                d.to_pydatetime() if hasattr(d, "to_pydatetime") else d
                for d in missing_periods[:100]  # Limit to first 100
            ],
            coverage_percentage=coverage_pct,
            gaps_at_start_days=gaps_at_start,
            gaps_at_end_days=gaps_at_end,
            gaps_within_range=gaps_within,
        )

    def analyze_entity_coverage(
        self,
        entity_column: str,
        reference_entities: Sequence[Any],
        sample_missing: int = 100,
        sample_extra: int = 100,
    ) -> EntityCoverageReport:
        """
        Analyze entity coverage against a reference list.

        Args:
            entity_column: Column containing entity identifiers.
            reference_entities: List or set of expected entities.
            sample_missing: Maximum number of missing entities to return.
            sample_extra: Maximum number of extra entities to return.

        Returns:
            EntityCoverageReport with analysis results.
        """
        if entity_column not in self.df.columns:
            raise ValueError(f"Column '{entity_column}' not found in DataFrame")

        actual_entities = set(self.df[entity_column].dropna().unique())
        expected_entities = set(reference_entities)

        covered = actual_entities & expected_entities
        missing = expected_entities - actual_entities
        extra = actual_entities - expected_entities

        coverage_pct = (
            len(covered) / len(expected_entities) * 100
            if len(expected_entities) > 0
            else 100.0
        )

        # Get record counts per entity
        entity_counts = self.df[entity_column].value_counts().to_dict()

        return EntityCoverageReport(
            entity_column=entity_column,
            expected_entities=len(expected_entities),
            covered_entities=len(covered),
            missing_entities=list(missing)[:sample_missing],
            extra_entities=list(extra)[:sample_extra],
            coverage_percentage=coverage_pct,
            entity_record_counts=entity_counts,
        )

    def analyze_attribute_coverage(
        self,
        required_columns: list[str],
        recommended_columns: list[str] | None = None,
        optional_columns: list[str] | None = None,
    ) -> AttributeCoverageReport:
        """
        Analyze attribute (column) coverage against schema requirements.

        Args:
            required_columns: List of required columns that must be present.
            recommended_columns: List of recommended columns.
            optional_columns: List of optional columns.

        Returns:
            AttributeCoverageReport with analysis results.
        """
        recommended_columns = recommended_columns or []
        optional_columns = optional_columns or []

        present_columns = set(self.df.columns)

        # Analyze required columns
        present_required = [c for c in required_columns if c in present_columns]
        missing_required = [c for c in required_columns if c not in present_columns]

        # Analyze recommended columns
        present_recommended = [c for c in recommended_columns if c in present_columns]
        missing_recommended = [
            c for c in recommended_columns if c not in present_columns
        ]

        # Analyze optional columns
        present_optional = [c for c in optional_columns if c in present_columns]

        # Calculate coverage percentages
        required_coverage = (
            len(present_required) / len(required_columns) * 100
            if required_columns
            else 100.0
        )

        recommended_coverage = (
            len(present_recommended) / len(recommended_columns) * 100
            if recommended_columns
            else 100.0
        )

        all_expected = required_columns + recommended_columns + optional_columns
        all_present = present_required + present_recommended + present_optional
        overall_coverage = (
            len(all_present) / len(all_expected) * 100 if all_expected else 100.0
        )

        # Generate detailed attribute status
        attribute_details = []

        for col in required_columns:
            present = col in present_columns
            non_null = self.df[col].notna().sum() if present else 0
            total = len(self.df) if present else 0
            completeness = (non_null / total * 100) if total > 0 else 0

            attribute_details.append(
                AttributeStatus(
                    name=col,
                    requirement_level="required",
                    present=present,
                    non_null_count=int(non_null),
                    total_count=total,
                    completeness_percentage=completeness,
                )
            )

        for col in recommended_columns:
            present = col in present_columns
            non_null = self.df[col].notna().sum() if present else 0
            total = len(self.df) if present else 0
            completeness = (non_null / total * 100) if total > 0 else 0

            attribute_details.append(
                AttributeStatus(
                    name=col,
                    requirement_level="recommended",
                    present=present,
                    non_null_count=int(non_null),
                    total_count=total,
                    completeness_percentage=completeness,
                )
            )

        for col in optional_columns:
            present = col in present_columns
            non_null = self.df[col].notna().sum() if present else 0
            total = len(self.df) if present else 0
            completeness = (non_null / total * 100) if total > 0 else 0

            attribute_details.append(
                AttributeStatus(
                    name=col,
                    requirement_level="optional",
                    present=present,
                    non_null_count=int(non_null),
                    total_count=total,
                    completeness_percentage=completeness,
                )
            )

        return AttributeCoverageReport(
            required_columns=required_columns,
            recommended_columns=recommended_columns,
            optional_columns=optional_columns,
            present_required=present_required,
            missing_required=missing_required,
            present_recommended=present_recommended,
            missing_recommended=missing_recommended,
            attribute_details=attribute_details,
            required_coverage_percentage=required_coverage,
            recommended_coverage_percentage=recommended_coverage,
            overall_coverage_percentage=overall_coverage,
        )

    def compare_datasets(
        self,
        reference_df: pd.DataFrame,
        key_columns: list[str],
        compare_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compare coverage between current dataset and a reference dataset.

        Args:
            reference_df: Reference DataFrame to compare against.
            key_columns: Columns to use as keys for matching records.
            compare_columns: Specific columns to compare (all if None).

        Returns:
            Dictionary with comparison results.
        """
        if compare_columns is None:
            compare_columns = [
                c for c in self.df.columns if c not in key_columns
            ]

        # Create composite keys
        current_keys = set(
            self.df[key_columns].apply(lambda x: tuple(x), axis=1)
        )
        reference_keys = set(
            reference_df[key_columns].apply(lambda x: tuple(x), axis=1)
        )

        matched_keys = current_keys & reference_keys
        missing_keys = reference_keys - current_keys
        extra_keys = current_keys - reference_keys

        # Calculate coverage metrics
        key_coverage = (
            len(matched_keys) / len(reference_keys) * 100
            if len(reference_keys) > 0
            else 100.0
        )

        # Analyze column-level coverage for matched records
        column_coverage = {}
        for col in compare_columns:
            if col in self.df.columns and col in reference_df.columns:
                current_non_null = self.df[col].notna().sum()
                ref_non_null = reference_df[col].notna().sum()
                column_coverage[col] = {
                    "current_non_null": int(current_non_null),
                    "reference_non_null": int(ref_non_null),
                    "coverage_ratio": (
                        current_non_null / ref_non_null if ref_non_null > 0 else 1.0
                    ),
                }

        return {
            "key_columns": key_columns,
            "total_reference_records": len(reference_keys),
            "total_current_records": len(current_keys),
            "matched_records": len(matched_keys),
            "missing_records": len(missing_keys),
            "extra_records": len(extra_keys),
            "key_coverage_percentage": key_coverage,
            "missing_keys_sample": list(missing_keys)[:50],
            "extra_keys_sample": list(extra_keys)[:50],
            "column_coverage": column_coverage,
        }

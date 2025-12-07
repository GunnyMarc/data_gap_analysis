"""
Main Gap Analysis Orchestrator

This module provides the main GapAnalyzer class that orchestrates
all phases of data gap analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from .coverage import CoverageAnalyzer, CoverageReport
from .drift import ConceptDriftReport, DataDriftReport, DriftAnalyzer
from .profiling import DataProfile, DataProfiler
from .quality import QualityAnalyzer, QualityReport
from .report import GapAnalysisReport


@dataclass
class GapAnalysisConfig:
    """Configuration for gap analysis."""

    # Phase 1: Profiling
    date_columns: list[str] = field(default_factory=list)
    expected_frequency: str = "D"
    schema: dict[str, str] = field(default_factory=dict)

    # Phase 2: Coverage
    location_column: str | None = None
    expected_regions: list[str] = field(default_factory=list)
    date_column: str | None = None
    expected_start: str | datetime | None = None
    expected_end: str | datetime | None = None
    entity_column: str | None = None
    reference_entities: list[Any] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    recommended_columns: list[str] = field(default_factory=list)
    optional_columns: list[str] = field(default_factory=list)

    # Phase 3: Quality
    format_columns: list[str] = field(default_factory=list)
    duplicate_key_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    foreign_key_configs: list[dict[str, Any]] = field(default_factory=list)
    timestamp_column: str | None = None
    staleness_threshold_days: int = 90

    # Phase 4: Drift
    reference_data: pd.DataFrame | None = None
    feature_columns: list[str] = field(default_factory=list)
    target_column: str | None = None
    categorical_columns: list[str] = field(default_factory=list)


class GapAnalyzer:
    """
    Main orchestrator for comprehensive gap analysis.

    This class coordinates all four phases of gap analysis:
    1. Data Profiling
    2. Coverage Comparison
    3. Quality Gap Identification
    4. Model Drift Detection

    Example:
        >>> analyzer = GapAnalyzer(df)
        >>> report = analyzer.run_full_analysis(schema=expected_schema)
        >>> print(report.summary())
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize the GapAnalyzer with a pandas DataFrame.

        Args:
            dataframe: The pandas DataFrame to analyze.
        """
        self.df = dataframe
        self._profiler = DataProfiler(dataframe)
        self._coverage_analyzer = CoverageAnalyzer(dataframe)
        self._quality_analyzer = QualityAnalyzer(dataframe)
        self._drift_analyzer = DriftAnalyzer()

    def run_full_analysis(
        self,
        config: GapAnalysisConfig | None = None,
        schema: dict[str, str] | None = None,
        reference_data: pd.DataFrame | None = None,
        date_column: str | None = None,
        run_drift: bool = True,
    ) -> GapAnalysisReport:
        """
        Run complete gap analysis across all phases.

        Args:
            config: Configuration object for all phases.
            schema: Expected schema (column -> type mapping).
            reference_data: Reference data for drift detection.
            date_column: Date column for temporal analysis.
            run_drift: Whether to run drift detection.

        Returns:
            GapAnalysisReport containing all analysis results.
        """
        if config is None:
            config = GapAnalysisConfig()

        # Override config with direct parameters if provided
        if schema:
            config.schema = schema
        if reference_data is not None:
            config.reference_data = reference_data
        if date_column:
            config.date_column = date_column
            config.date_columns = [date_column]

        # Phase 1: Profiling
        profile = self.run_profiling(
            schema=config.schema,
            date_columns=config.date_columns,
            expected_frequency=config.expected_frequency,
        )

        # Phase 2: Coverage
        coverage = self.run_coverage_analysis(
            location_column=config.location_column,
            expected_regions=config.expected_regions,
            date_column=config.date_column,
            expected_start=config.expected_start,
            expected_end=config.expected_end,
            entity_column=config.entity_column,
            reference_entities=config.reference_entities,
            required_columns=config.required_columns,
            recommended_columns=config.recommended_columns,
            optional_columns=config.optional_columns,
        )

        # Phase 3: Quality
        quality = self.run_quality_analysis(
            format_columns=config.format_columns,
            duplicate_key_columns=config.duplicate_key_columns,
            numeric_columns=config.numeric_columns,
            foreign_key_configs=config.foreign_key_configs,
            timestamp_column=config.timestamp_column,
            staleness_threshold_days=config.staleness_threshold_days,
        )

        # Phase 4: Drift
        data_drift = None
        concept_drift = None

        if run_drift and config.reference_data is not None:
            drift_results = self.run_drift_analysis(
                reference_data=config.reference_data,
                feature_columns=config.feature_columns,
                target_column=config.target_column,
                categorical_columns=config.categorical_columns,
            )
            data_drift = drift_results.get("data_drift")
            concept_drift = drift_results.get("concept_drift")

        return GapAnalysisReport(
            data_profile=profile,
            coverage_report=coverage,
            quality_report=quality,
            data_drift_report=data_drift,
            concept_drift_report=concept_drift,
        )

    def run_profiling(
        self,
        schema: dict[str, str] | None = None,
        date_columns: list[str] | None = None,
        expected_frequency: str = "D",
    ) -> DataProfile:
        """
        Run Phase 1: Data Profiling.

        Args:
            schema: Expected schema for type validation.
            date_columns: Date columns for time series analysis.
            expected_frequency: Expected data frequency.

        Returns:
            DataProfile with profiling results.
        """
        return self._profiler.generate_profile(
            schema=schema,
            date_columns=date_columns,
            expected_frequency=expected_frequency,
        )

    def run_coverage_analysis(
        self,
        location_column: str | None = None,
        expected_regions: list[str] | None = None,
        date_column: str | None = None,
        expected_start: str | datetime | None = None,
        expected_end: str | datetime | None = None,
        expected_frequency: str = "D",
        entity_column: str | None = None,
        reference_entities: list[Any] | None = None,
        required_columns: list[str] | None = None,
        recommended_columns: list[str] | None = None,
        optional_columns: list[str] | None = None,
    ) -> CoverageReport:
        """
        Run Phase 2: Coverage Analysis.

        Args:
            location_column: Column for geographic coverage.
            expected_regions: Expected geographic regions.
            date_column: Column for temporal coverage.
            expected_start: Expected start date.
            expected_end: Expected end date.
            expected_frequency: Expected data frequency.
            entity_column: Column for entity coverage.
            reference_entities: Expected entities.
            required_columns: Required schema columns.
            recommended_columns: Recommended schema columns.
            optional_columns: Optional schema columns.

        Returns:
            CoverageReport with coverage analysis results.
        """
        return self._coverage_analyzer.analyze_all_coverage(
            location_column=location_column,
            expected_regions=expected_regions or [],
            date_column=date_column,
            expected_start=expected_start,
            expected_end=expected_end,
            expected_frequency=expected_frequency,
            entity_column=entity_column,
            reference_entities=reference_entities,
            required_columns=required_columns,
            recommended_columns=recommended_columns,
            optional_columns=optional_columns,
        )

    def run_quality_analysis(
        self,
        format_columns: list[str] | None = None,
        duplicate_key_columns: list[str] | None = None,
        numeric_columns: list[str] | None = None,
        foreign_key_configs: list[dict[str, Any]] | None = None,
        timestamp_column: str | None = None,
        staleness_threshold_days: int = 90,
    ) -> QualityReport:
        """
        Run Phase 3: Quality Gap Analysis.

        Args:
            format_columns: Columns to check for format inconsistencies.
            duplicate_key_columns: Key columns for duplicate detection.
            numeric_columns: Columns for outlier detection.
            foreign_key_configs: Foreign key configurations.
            timestamp_column: Column for stale data detection.
            staleness_threshold_days: Staleness threshold in days.

        Returns:
            QualityReport with quality analysis results.
        """
        return self._quality_analyzer.run_full_analysis(
            format_columns=format_columns,
            duplicate_key_columns=duplicate_key_columns,
            numeric_columns=numeric_columns,
            foreign_key_configs=foreign_key_configs,
            timestamp_column=timestamp_column,
            staleness_threshold_days=staleness_threshold_days,
        )

    def run_drift_analysis(
        self,
        reference_data: pd.DataFrame,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        categorical_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run Phase 4: Drift Detection.

        Args:
            reference_data: Reference/baseline DataFrame.
            feature_columns: Feature columns to analyze.
            target_column: Target column for concept drift.
            categorical_columns: Categorical feature columns.

        Returns:
            Dictionary with data_drift and concept_drift reports.
        """
        results = {}

        # Auto-detect feature columns if not provided
        if feature_columns is None:
            feature_columns = [
                c for c in self.df.columns
                if c in reference_data.columns and c != target_column
            ]

        # Data drift detection
        data_drift = self._drift_analyzer.detect_data_drift(
            reference_data=reference_data,
            current_data=self.df,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
        )
        results["data_drift"] = data_drift

        # Concept drift detection (if target column provided)
        if target_column and target_column in self.df.columns:
            concept_drift = self._drift_analyzer.detect_concept_drift(
                reference_data=reference_data,
                current_data=self.df,
                feature_columns=feature_columns,
                target_column=target_column,
            )
            results["concept_drift"] = concept_drift

        return results

    def quick_profile(self) -> dict[str, Any]:
        """
        Generate a quick summary profile of the data.

        Returns:
            Dictionary with basic data statistics.
        """
        profile = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "columns": {},
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        for col in self.df.columns:
            col_info = {
                "dtype": str(self.df[col].dtype),
                "null_count": int(self.df[col].isna().sum()),
                "null_percentage": round(
                    self.df[col].isna().sum() / len(self.df) * 100, 2
                ),
                "unique_count": int(self.df[col].nunique()),
            }

            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_info.update({
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                    "mean": float(self.df[col].mean()),
                })

            profile["columns"][col] = col_info

        return profile

    def get_gap_summary(self) -> dict[str, Any]:
        """
        Get a high-level summary of data gaps.

        Returns:
            Dictionary with gap summary statistics.
        """
        null_counts = self.df.isnull().sum()
        total_cells = len(self.df) * len(self.df.columns)
        total_nulls = null_counts.sum()

        columns_with_nulls = (null_counts > 0).sum()
        columns_complete = len(self.df.columns) - columns_with_nulls

        return {
            "total_records": len(self.df),
            "total_columns": len(self.df.columns),
            "total_cells": total_cells,
            "total_null_cells": int(total_nulls),
            "overall_completeness": round((1 - total_nulls / total_cells) * 100, 2),
            "columns_with_gaps": int(columns_with_nulls),
            "complete_columns": int(columns_complete),
            "most_incomplete_columns": null_counts.nlargest(5).to_dict(),
        }

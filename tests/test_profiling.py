"""Tests for Phase 1: Data Profiling module."""

import numpy as np
import pandas as pd
import pytest

from data_gap_analysis.profiling import DataProfiler


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(1, 101),
        "name": [f"User_{i}" for i in range(1, 101)],
        "email": [f"user{i}@example.com" if i % 10 != 0 else None for i in range(1, 101)],
        "age": np.random.randint(18, 80, 100),
        "salary": np.random.normal(50000, 15000, 100),
        "created_at": pd.date_range("2024-01-01", periods=100, freq="D"),
        "category": np.random.choice(["A", "B", "C"], 100),
    })


class TestDataProfiler:
    """Tests for DataProfiler class."""

    def test_count_nulls(self, sample_dataframe):
        """Test null counting functionality."""
        profiler = DataProfiler(sample_dataframe)
        null_reports = profiler.count_nulls()

        assert len(null_reports) == len(sample_dataframe.columns)

        # Find email column report
        email_report = next(r for r in null_reports if r.column == "email")
        assert email_report.null_count == 10
        assert email_report.null_percentage == 10.0

    def test_assess_cardinality(self, sample_dataframe):
        """Test cardinality assessment."""
        profiler = DataProfiler(sample_dataframe)
        cardinality_reports = profiler.assess_cardinality()

        # ID should be unique
        id_report = next(r for r in cardinality_reports if r.column == "id")
        assert id_report.cardinality_type == "unique"
        assert id_report.unique_count == 100

        # Category should be low cardinality
        cat_report = next(r for r in cardinality_reports if r.column == "category")
        assert cat_report.cardinality_type == "low"
        assert cat_report.unique_count == 3

    def test_examine_data_ranges(self, sample_dataframe):
        """Test data range examination."""
        profiler = DataProfiler(sample_dataframe)
        range_reports = profiler.examine_data_ranges()

        # Find age column report
        age_report = next(r for r in range_reports if r.column == "age")
        assert age_report.min_value >= 18
        assert age_report.max_value < 80
        assert 50 in age_report.percentiles

    def test_analyze_value_distribution(self, sample_dataframe):
        """Test value distribution analysis."""
        profiler = DataProfiler(sample_dataframe)
        distribution_reports = profiler.analyze_value_distribution()

        # Find salary column report
        salary_report = next(r for r in distribution_reports if r.column == "salary")
        assert salary_report.mean is not None
        assert salary_report.std is not None
        assert salary_report.median is not None

    def test_check_time_series_continuity(self, sample_dataframe):
        """Test time series continuity checking."""
        profiler = DataProfiler(sample_dataframe)
        ts_report = profiler.check_time_series_continuity(
            date_column="created_at",
            expected_frequency="D",
        )

        assert ts_report.continuity_percentage == 100.0
        assert len(ts_report.gaps) == 0

    def test_validate_data_types(self, sample_dataframe):
        """Test data type validation."""
        profiler = DataProfiler(sample_dataframe)

        schema = {
            "id": "int",
            "email": "email",
            "age": "int",
        }

        validation_reports = profiler.validate_data_types(schema)

        # ID should be valid integer
        id_report = next(r for r in validation_reports if r.column == "id")
        assert id_report.validity_percentage == 100.0

        # Email should have ~90% validity (10% null)
        email_report = next(r for r in validation_reports if r.column == "email")
        assert email_report.validity_percentage == 90.0

    def test_generate_profile(self, sample_dataframe):
        """Test complete profile generation."""
        profiler = DataProfiler(sample_dataframe)
        profile = profiler.generate_profile(
            schema={"id": "int", "email": "email"},
            date_columns=["created_at"],
        )

        assert profile.total_rows == 100
        assert profile.total_columns == 7
        assert len(profile.null_reports) == 7
        assert len(profile.cardinality_reports) == 7
        assert len(profile.time_series_reports) == 1
        assert len(profile.type_validation_reports) == 2

"""Tests for Phase 3: Quality Gap Identification module."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from data_gap_analysis.quality import QualityAnalyzer


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)

    # Create data with intentional quality issues
    dates = ["2024-01-01"] * 30 + ["01/15/2024"] * 20 + ["2024-02-01"] * 50
    phones = ["+1234567890"] * 40 + ["123-456-7890"] * 30 + ["(123) 456-7890"] * 30

    return pd.DataFrame({
        "id": range(1, 101),
        "name": [f"User_{i}" for i in range(1, 101)],
        "email": [f"user{i}@example.com" for i in range(1, 101)],
        "date": dates,
        "phone": phones,
        "amount": list(np.random.normal(100, 20, 95)) + [1000, 2000, 3000, -500, 5000],  # Outliers
        "customer_id": [f"CUST_{i % 50:04d}" for i in range(1, 101)],  # Duplicates
        "last_updated": pd.date_range("2024-01-01", periods=100, freq="D") - timedelta(days=100),
    })


@pytest.fixture
def reference_dataframe():
    """Create a reference DataFrame for integrity testing."""
    return pd.DataFrame({
        "id": [f"CUST_{i:04d}" for i in range(0, 40)],
        "name": [f"Customer_{i}" for i in range(0, 40)],
    })


class TestQualityAnalyzer:
    """Tests for QualityAnalyzer class."""

    def test_detect_inconsistent_formats(self, sample_dataframe):
        """Test format inconsistency detection."""
        analyzer = QualityAnalyzer(sample_dataframe)

        format_issues = analyzer.detect_inconsistent_formats(["date", "phone"])

        assert len(format_issues) == 2

        date_issue = next(i for i in format_issues if i.column == "date")
        assert len(date_issue.detected_formats) > 1
        assert date_issue.inconsistency_percentage > 0

    def test_find_duplicates(self, sample_dataframe):
        """Test duplicate detection."""
        analyzer = QualityAnalyzer(sample_dataframe)

        duplicate_report = analyzer.find_duplicates(key_columns=["customer_id"])

        assert duplicate_report.exact_duplicate_count > 0
        assert duplicate_report.duplicate_percentage > 0

    def test_detect_outliers_iqr(self, sample_dataframe):
        """Test outlier detection using IQR method."""
        analyzer = QualityAnalyzer(sample_dataframe)

        outlier_reports = analyzer.detect_outliers(
            numeric_columns=["amount"],
            method="iqr",
            threshold=1.5,
        )

        assert len(outlier_reports) == 1
        assert outlier_reports[0].outlier_count > 0
        assert outlier_reports[0].outlier_percentage > 0

    def test_detect_outliers_zscore(self, sample_dataframe):
        """Test outlier detection using Z-score method."""
        analyzer = QualityAnalyzer(sample_dataframe)

        outlier_reports = analyzer.detect_outliers(
            numeric_columns=["amount"],
            method="zscore",
            threshold=2.0,
        )

        assert len(outlier_reports) == 1
        assert outlier_reports[0].method == "zscore"

    def test_check_referential_integrity(self, sample_dataframe, reference_dataframe):
        """Test referential integrity checking."""
        analyzer = QualityAnalyzer(sample_dataframe)

        integrity_issue = analyzer.check_referential_integrity(
            foreign_key="customer_id",
            reference_df=reference_dataframe,
            reference_key="id",
        )

        assert integrity_issue.orphaned_count > 0
        assert integrity_issue.orphaned_percentage > 0

    def test_detect_stale_data(self, sample_dataframe):
        """Test stale data detection."""
        analyzer = QualityAnalyzer(sample_dataframe)

        stale_report = analyzer.detect_stale_data(
            timestamp_column="last_updated",
            staleness_threshold_days=30,
        )

        assert stale_report.stale_count > 0
        assert stale_report.stale_percentage > 0

    def test_run_full_analysis(self, sample_dataframe, reference_dataframe):
        """Test complete quality analysis."""
        analyzer = QualityAnalyzer(sample_dataframe)

        report = analyzer.run_full_analysis(
            format_columns=["date", "phone"],
            duplicate_key_columns=["customer_id"],
            numeric_columns=["amount"],
            foreign_key_configs=[{
                "foreign_key": "customer_id",
                "reference_df": reference_dataframe,
                "reference_key": "id",
            }],
            timestamp_column="last_updated",
            staleness_threshold_days=30,
        )

        assert len(report.format_issues) > 0
        assert len(report.duplicate_reports) > 0
        assert len(report.outlier_reports) > 0
        assert len(report.referential_integrity_issues) > 0
        assert len(report.stale_data_reports) > 0

    def test_detect_unit_inconsistencies(self):
        """Test unit inconsistency detection."""
        df = pd.DataFrame({
            "price": ["$100", "€50", "$75", "100USD", "$200"],
        })

        analyzer = QualityAnalyzer(df)
        result = analyzer.detect_unit_inconsistencies("price")

        assert not result["is_consistent"]
        assert result["unit_count"] > 1

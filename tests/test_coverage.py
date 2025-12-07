"""Tests for Phase 2: Data Coverage Comparison module."""

import numpy as np
import pandas as pd
import pytest

from data_gap_analysis.coverage import CoverageAnalyzer


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(1, 101),
        "customer_id": [f"CUST_{i:04d}" for i in range(1, 101)],
        "country": np.random.choice(["US", "UK", "DE", "FR"], 100),
        "transaction_date": pd.date_range("2024-01-01", periods=100, freq="D"),
        "amount": np.random.uniform(10, 1000, 100),
        "category": np.random.choice(["Electronics", "Clothing", "Food"], 100),
    })


class TestCoverageAnalyzer:
    """Tests for CoverageAnalyzer class."""

    def test_analyze_geographic_coverage(self, sample_dataframe):
        """Test geographic coverage analysis."""
        analyzer = CoverageAnalyzer(sample_dataframe)

        geo_coverage = analyzer.analyze_geographic_coverage(
            location_column="country",
            expected_regions=["US", "UK", "DE", "FR", "JP"],
        )

        assert geo_coverage.coverage_percentage == 80.0
        assert "JP" in geo_coverage.missing_regions
        assert len(geo_coverage.covered_regions) == 4

    def test_analyze_temporal_coverage(self, sample_dataframe):
        """Test temporal coverage analysis."""
        analyzer = CoverageAnalyzer(sample_dataframe)

        temporal_coverage = analyzer.analyze_temporal_coverage(
            date_column="transaction_date",
            expected_frequency="D",
        )

        assert temporal_coverage.coverage_percentage == 100.0
        assert temporal_coverage.gaps_within_range == 0

    def test_analyze_temporal_coverage_with_gaps(self):
        """Test temporal coverage with missing dates."""
        # Create data with gaps
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        dates = dates[dates.day != 15]  # Remove all 15ths

        df = pd.DataFrame({
            "date": dates,
            "value": range(len(dates)),
        })

        analyzer = CoverageAnalyzer(df)
        temporal_coverage = analyzer.analyze_temporal_coverage(
            date_column="date",
            expected_start="2024-01-01",
            expected_end="2024-03-31",
            expected_frequency="D",
        )

        assert temporal_coverage.coverage_percentage < 100.0
        assert temporal_coverage.gaps_within_range > 0

    def test_analyze_entity_coverage(self, sample_dataframe):
        """Test entity coverage analysis."""
        analyzer = CoverageAnalyzer(sample_dataframe)

        # Create reference entities (some missing from sample)
        reference_entities = [f"CUST_{i:04d}" for i in range(1, 121)]

        entity_coverage = analyzer.analyze_entity_coverage(
            entity_column="customer_id",
            reference_entities=reference_entities,
        )

        assert entity_coverage.covered_entities == 100
        assert len(entity_coverage.missing_entities) == 20
        assert entity_coverage.coverage_percentage == pytest.approx(83.33, rel=0.01)

    def test_analyze_attribute_coverage(self, sample_dataframe):
        """Test attribute coverage analysis."""
        analyzer = CoverageAnalyzer(sample_dataframe)

        attr_coverage = analyzer.analyze_attribute_coverage(
            required_columns=["id", "customer_id", "amount"],
            recommended_columns=["country", "category"],
            optional_columns=["notes", "metadata"],
        )

        assert attr_coverage.required_coverage_percentage == 100.0
        assert attr_coverage.recommended_coverage_percentage == 100.0
        assert len(attr_coverage.missing_required) == 0
        assert "notes" in [a.name for a in attr_coverage.attribute_details if not a.present]

    def test_analyze_all_coverage(self, sample_dataframe):
        """Test complete coverage analysis."""
        analyzer = CoverageAnalyzer(sample_dataframe)

        coverage = analyzer.analyze_all_coverage(
            location_column="country",
            expected_regions=["US", "UK", "DE", "FR"],
            date_column="transaction_date",
            required_columns=["id", "amount"],
        )

        assert coverage.geographic_coverage is not None
        assert coverage.temporal_coverage is not None
        assert coverage.attribute_coverage is not None

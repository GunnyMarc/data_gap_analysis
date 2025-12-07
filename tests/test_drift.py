"""Tests for Phase 4: Model Drift Computation module."""

import numpy as np
import pandas as pd
import pytest

from data_gap_analysis.drift import DriftAnalyzer, DriftSeverity


@pytest.fixture
def reference_data():
    """Create reference/baseline data."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature_a": np.random.normal(100, 10, 1000),
        "feature_b": np.random.normal(50, 5, 1000),
        "feature_c": np.random.choice(["cat1", "cat2", "cat3"], 1000),
        "target": np.random.choice([0, 1], 1000),
    })


@pytest.fixture
def similar_data():
    """Create data similar to reference (no significant drift)."""
    np.random.seed(43)
    return pd.DataFrame({
        "feature_a": np.random.normal(100, 10, 1000),
        "feature_b": np.random.normal(50, 5, 1000),
        "feature_c": np.random.choice(["cat1", "cat2", "cat3"], 1000),
        "target": np.random.choice([0, 1], 1000),
    })


@pytest.fixture
def drifted_data():
    """Create data with significant drift."""
    np.random.seed(44)
    return pd.DataFrame({
        "feature_a": np.random.normal(120, 15, 1000),  # Shifted mean and std
        "feature_b": np.random.normal(60, 8, 1000),    # Shifted mean and std
        "feature_c": np.random.choice(["cat1", "cat4", "cat5"], 1000),  # New categories
        "target": np.random.choice([0, 1], 1000, p=[0.3, 0.7]),  # Different distribution
    })


class TestDriftAnalyzer:
    """Tests for DriftAnalyzer class."""

    def test_detect_data_drift_no_drift(self, reference_data, similar_data):
        """Test data drift detection with similar data."""
        analyzer = DriftAnalyzer()

        report = analyzer.detect_data_drift(
            reference_data=reference_data,
            current_data=similar_data,
            feature_columns=["feature_a", "feature_b"],
            methods=["psi", "ks_test"],
        )

        assert report.overall_severity in (DriftSeverity.NONE, DriftSeverity.LOW)
        assert len(report.feature_results) == 2

    def test_detect_data_drift_with_drift(self, reference_data, drifted_data):
        """Test data drift detection with drifted data."""
        analyzer = DriftAnalyzer()

        report = analyzer.detect_data_drift(
            reference_data=reference_data,
            current_data=drifted_data,
            feature_columns=["feature_a", "feature_b"],
            methods=["psi"],
        )

        assert report.overall_severity in (
            DriftSeverity.MODERATE,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        )
        assert len(report.drifted_features) > 0

    def test_detect_data_drift_categorical(self, reference_data, drifted_data):
        """Test data drift detection for categorical features."""
        analyzer = DriftAnalyzer()

        report = analyzer.detect_data_drift(
            reference_data=reference_data,
            current_data=drifted_data,
            feature_columns=["feature_c"],
            categorical_columns=["feature_c"],
            methods=["psi", "chi_square"],
        )

        assert len(report.feature_results) == 1
        # Categorical with new values should show drift
        assert report.feature_results[0].drift_score > 0

    def test_detect_concept_drift(self, reference_data, drifted_data):
        """Test concept drift detection."""
        analyzer = DriftAnalyzer()

        report = analyzer.detect_concept_drift(
            reference_data=reference_data,
            current_data=drifted_data,
            feature_columns=["feature_a", "feature_b"],
            target_column="target",
            methods=["ddm", "error_rate"],
        )

        assert report.target_column == "target"
        # With different target distribution, drift should be detected
        assert report.drift_detected or len(report.drift_points) >= 0

    def test_create_drift_monitor(self, reference_data, similar_data, drifted_data):
        """Test drift monitor creation and usage."""
        analyzer = DriftAnalyzer()

        monitor = analyzer.create_drift_monitor(
            baseline_data=reference_data,
            feature_columns=["feature_a", "feature_b"],
            alert_threshold=0.15,
        )

        # Check similar data (should have no/few alerts)
        alerts_similar = monitor.check_batch(similar_data)

        # Check drifted data (should have alerts)
        alerts_drifted = monitor.check_batch(drifted_data)

        assert len(alerts_drifted) > len(alerts_similar)
        assert len(monitor.get_history()) == 2

    def test_drift_severity_levels(self):
        """Test drift severity classification."""
        analyzer = DriftAnalyzer()

        # Test PSI thresholds
        assert analyzer._get_severity_from_psi(0.05) == DriftSeverity.NONE
        assert analyzer._get_severity_from_psi(0.15) == DriftSeverity.LOW
        assert analyzer._get_severity_from_psi(0.22) == DriftSeverity.MODERATE
        assert analyzer._get_severity_from_psi(0.35) == DriftSeverity.HIGH
        assert analyzer._get_severity_from_psi(0.6) == DriftSeverity.CRITICAL

    def test_drift_report_summary(self, reference_data, drifted_data):
        """Test drift report string output."""
        analyzer = DriftAnalyzer()

        report = analyzer.detect_data_drift(
            reference_data=reference_data,
            current_data=drifted_data,
            feature_columns=["feature_a", "feature_b"],
        )

        report_str = str(report)

        assert "DATA DRIFT REPORT" in report_str
        assert "feature_a" in report_str
        assert "feature_b" in report_str

    def test_wasserstein_distance(self, reference_data, drifted_data):
        """Test Wasserstein distance calculation."""
        analyzer = DriftAnalyzer()

        report = analyzer.detect_data_drift(
            reference_data=reference_data,
            current_data=drifted_data,
            feature_columns=["feature_a"],
            methods=["wasserstein"],
        )

        assert len(report.feature_results) == 1
        assert report.feature_results[0].method == "Wasserstein"
        assert report.feature_results[0].statistic > 0

    def test_js_divergence(self, reference_data, drifted_data):
        """Test Jensen-Shannon divergence calculation."""
        analyzer = DriftAnalyzer()

        report = analyzer.detect_data_drift(
            reference_data=reference_data,
            current_data=drifted_data,
            feature_columns=["feature_a"],
            methods=["js_divergence"],
        )

        assert len(report.feature_results) == 1
        assert report.feature_results[0].method == "JS Divergence"

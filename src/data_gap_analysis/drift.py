"""
Phase 4: Model Drift Computation Module

This module provides drift detection capabilities including:
- Data drift detection (changes in input feature distributions)
- Concept drift detection (changes in relationship between features and target)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


class DriftSeverity(Enum):
    """Severity levels for detected drift."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeatureDriftResult:
    """Drift detection result for a single feature."""

    feature: str
    method: str
    statistic: float
    p_value: float | None
    drift_score: float
    severity: DriftSeverity
    reference_stats: dict[str, float] = field(default_factory=dict)
    current_stats: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Feature: {self.feature}\n"
            f"  Method: {self.method}\n"
            f"  Drift Score: {self.drift_score:.4f}\n"
            f"  Severity: {self.severity.value}\n"
            f"  P-value: {self.p_value:.4f}" if self.p_value else ""
        )


@dataclass
class DataDriftReport:
    """Complete data drift analysis report."""

    reference_period: str
    current_period: str
    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    overall_drift_score: float = 0.0
    overall_severity: DriftSeverity = DriftSeverity.NONE
    drifted_features: list[str] = field(default_factory=list)
    recommendation: str = ""
    analysis_timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "DATA DRIFT REPORT",
            "=" * 60,
            f"Reference Period: {self.reference_period}",
            f"Current Period: {self.current_period}",
            f"Overall Drift Score: {self.overall_drift_score:.4f}",
            f"Overall Severity: {self.overall_severity.value}",
            "",
            "Feature Drift Scores:",
            "-" * 40,
        ]

        for result in self.feature_results:
            severity_icon = {
                DriftSeverity.NONE: "✓",
                DriftSeverity.LOW: "○",
                DriftSeverity.MODERATE: "⚠️",
                DriftSeverity.HIGH: "⚠️",
                DriftSeverity.CRITICAL: "🚨",
            }.get(result.severity, "?")

            lines.append(
                f"  {result.feature}: {result.drift_score:.4f} "
                f"({result.severity.value}) {severity_icon}"
            )

        if self.drifted_features:
            lines.extend(["", f"Drifted Features: {self.drifted_features}"])

        if self.recommendation:
            lines.extend(["", f"Recommendation: {self.recommendation}"])

        return "\n".join(lines)


@dataclass
class ConceptDriftPoint:
    """Represents a detected concept drift point."""

    timestamp: datetime | int
    detection_method: str
    severity: DriftSeverity
    metric_before: float
    metric_after: float
    description: str


@dataclass
class ConceptDriftReport:
    """Complete concept drift analysis report."""

    detection_method: str
    target_column: str
    drift_points: list[ConceptDriftPoint] = field(default_factory=list)
    drift_detected: bool = False
    performance_degradation: float = 0.0
    recommendation: str = ""
    analysis_timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "CONCEPT DRIFT REPORT",
            "=" * 60,
            f"Detection Method: {self.detection_method}",
            f"Target Column: {self.target_column}",
            f"Drift Detected: {self.drift_detected}",
            f"Performance Degradation: {self.performance_degradation:.2%}",
            "",
        ]

        if self.drift_points:
            lines.append("Drift Points Detected:")
            lines.append("-" * 40)
            for point in self.drift_points:
                lines.append(f"  {point.timestamp}: {point.description}")

        if self.recommendation:
            lines.extend(["", f"Recommendation: {self.recommendation}"])

        return "\n".join(lines)


class DriftAnalyzer:
    """
    Analyzer for detecting data drift and concept drift.

    This class provides methods to monitor changes in data distributions
    and relationships that can impact ML model performance.
    """

    def __init__(self) -> None:
        """Initialize the DriftAnalyzer."""
        self._psi_thresholds = {
            DriftSeverity.NONE: 0.0,
            DriftSeverity.LOW: 0.1,
            DriftSeverity.MODERATE: 0.2,
            DriftSeverity.HIGH: 0.25,
            DriftSeverity.CRITICAL: float("inf"),
        }

    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: list[str] | None = None,
        methods: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> DataDriftReport:
        """
        Detect data drift between reference and current datasets.

        Args:
            reference_data: Reference/baseline DataFrame (e.g., training data).
            current_data: Current DataFrame to compare (e.g., production data).
            feature_columns: Columns to analyze (all if None).
            methods: Detection methods to use (["psi", "ks_test"] if None).
            categorical_columns: Columns to treat as categorical.

        Returns:
            DataDriftReport with drift analysis results.
        """
        methods = methods or ["psi", "ks_test"]
        categorical_columns = categorical_columns or []

        if feature_columns is None:
            feature_columns = [
                c for c in reference_data.columns
                if c in current_data.columns
            ]

        feature_results = []
        drifted_features = []

        for col in feature_columns:
            if col not in reference_data.columns or col not in current_data.columns:
                continue

            ref_series = reference_data[col].dropna()
            cur_series = current_data[col].dropna()

            if len(ref_series) == 0 or len(cur_series) == 0:
                continue

            is_categorical = (
                col in categorical_columns
                or not pd.api.types.is_numeric_dtype(ref_series)
            )

            # Run drift detection methods
            best_result = None
            for method in methods:
                if method == "psi":
                    result = self._calculate_psi(ref_series, cur_series, col, is_categorical)
                elif method == "ks_test" and not is_categorical:
                    result = self._calculate_ks_test(ref_series, cur_series, col)
                elif method == "wasserstein" and not is_categorical:
                    result = self._calculate_wasserstein(ref_series, cur_series, col)
                elif method == "chi_square" and is_categorical:
                    result = self._calculate_chi_square(ref_series, cur_series, col)
                elif method == "js_divergence":
                    result = self._calculate_js_divergence(
                        ref_series, cur_series, col, is_categorical
                    )
                else:
                    continue

                if best_result is None or result.drift_score > best_result.drift_score:
                    best_result = result

            if best_result:
                feature_results.append(best_result)
                if best_result.severity in (
                    DriftSeverity.MODERATE,
                    DriftSeverity.HIGH,
                    DriftSeverity.CRITICAL,
                ):
                    drifted_features.append(col)

        # Calculate overall drift score
        if feature_results:
            overall_score = np.mean([r.drift_score for r in feature_results])
            overall_severity = self._get_severity_from_psi(overall_score)
        else:
            overall_score = 0.0
            overall_severity = DriftSeverity.NONE

        # Generate recommendation
        recommendation = self._generate_drift_recommendation(
            overall_severity, drifted_features
        )

        return DataDriftReport(
            reference_period="reference",
            current_period="current",
            feature_results=feature_results,
            overall_drift_score=overall_score,
            overall_severity=overall_severity,
            drifted_features=drifted_features,
            recommendation=recommendation,
        )

    def _calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature: str,
        is_categorical: bool,
    ) -> FeatureDriftResult:
        """Calculate Population Stability Index."""
        if is_categorical:
            # Use value counts for categorical
            ref_dist = reference.value_counts(normalize=True)
            cur_dist = current.value_counts(normalize=True)

            # Align distributions
            all_values = set(ref_dist.index) | set(cur_dist.index)
            ref_aligned = pd.Series(
                {v: ref_dist.get(v, 0.0001) for v in all_values}
            )
            cur_aligned = pd.Series(
                {v: cur_dist.get(v, 0.0001) for v in all_values}
            )
        else:
            # Create bins for numeric data
            n_bins = min(10, len(reference.unique()))
            bins = pd.qcut(reference, q=n_bins, duplicates="drop")
            bin_edges = bins.cat.categories

            ref_counts = pd.cut(reference, bins=bin_edges).value_counts(normalize=True)
            cur_counts = pd.cut(current, bins=bin_edges).value_counts(normalize=True)

            # Avoid zeros
            ref_aligned = ref_counts.replace(0, 0.0001)
            cur_aligned = cur_counts.reindex(ref_counts.index, fill_value=0.0001)

        # Calculate PSI
        psi = np.sum(
            (cur_aligned - ref_aligned) * np.log(cur_aligned / ref_aligned)
        )

        severity = self._get_severity_from_psi(psi)

        return FeatureDriftResult(
            feature=feature,
            method="PSI",
            statistic=float(psi),
            p_value=None,
            drift_score=float(psi),
            severity=severity,
            reference_stats={"mean": float(reference.mean()) if not is_categorical else 0},
            current_stats={"mean": float(current.mean()) if not is_categorical else 0},
        )

    def _calculate_ks_test(
        self, reference: pd.Series, current: pd.Series, feature: str
    ) -> FeatureDriftResult:
        """Calculate Kolmogorov-Smirnov test statistic."""
        statistic, p_value = stats.ks_2samp(reference, current)

        # Convert to drift score (higher = more drift)
        drift_score = statistic

        if p_value < 0.001:
            severity = DriftSeverity.HIGH
        elif p_value < 0.01:
            severity = DriftSeverity.MODERATE
        elif p_value < 0.05:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE

        return FeatureDriftResult(
            feature=feature,
            method="KS Test",
            statistic=float(statistic),
            p_value=float(p_value),
            drift_score=float(drift_score),
            severity=severity,
            reference_stats={
                "mean": float(reference.mean()),
                "std": float(reference.std()),
            },
            current_stats={
                "mean": float(current.mean()),
                "std": float(current.std()),
            },
        )

    def _calculate_wasserstein(
        self, reference: pd.Series, current: pd.Series, feature: str
    ) -> FeatureDriftResult:
        """Calculate Wasserstein (Earth Mover's) distance."""
        distance = stats.wasserstein_distance(reference, current)

        # Normalize by reference std for interpretability
        ref_std = reference.std()
        normalized_distance = distance / ref_std if ref_std > 0 else distance

        if normalized_distance > 0.5:
            severity = DriftSeverity.HIGH
        elif normalized_distance > 0.25:
            severity = DriftSeverity.MODERATE
        elif normalized_distance > 0.1:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE

        return FeatureDriftResult(
            feature=feature,
            method="Wasserstein",
            statistic=float(distance),
            p_value=None,
            drift_score=float(normalized_distance),
            severity=severity,
        )

    def _calculate_chi_square(
        self, reference: pd.Series, current: pd.Series, feature: str
    ) -> FeatureDriftResult:
        """Calculate Chi-square test for categorical variables."""
        ref_counts = reference.value_counts()
        cur_counts = current.value_counts()

        # Align categories
        all_categories = set(ref_counts.index) | set(cur_counts.index)
        ref_aligned = np.array([ref_counts.get(c, 0) for c in all_categories])
        cur_aligned = np.array([cur_counts.get(c, 0) for c in all_categories])

        # Scale to same total
        scale_factor = ref_aligned.sum() / cur_aligned.sum()
        cur_scaled = cur_aligned * scale_factor

        # Avoid zeros
        ref_aligned = np.maximum(ref_aligned, 1)
        cur_scaled = np.maximum(cur_scaled, 1)

        statistic, p_value = stats.chisquare(cur_scaled, ref_aligned)

        if p_value < 0.001:
            severity = DriftSeverity.HIGH
        elif p_value < 0.01:
            severity = DriftSeverity.MODERATE
        elif p_value < 0.05:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE

        return FeatureDriftResult(
            feature=feature,
            method="Chi-Square",
            statistic=float(statistic),
            p_value=float(p_value),
            drift_score=float(statistic / len(all_categories)),
            severity=severity,
        )

    def _calculate_js_divergence(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature: str,
        is_categorical: bool,
    ) -> FeatureDriftResult:
        """Calculate Jensen-Shannon divergence."""
        if is_categorical:
            ref_dist = reference.value_counts(normalize=True)
            cur_dist = current.value_counts(normalize=True)

            all_values = sorted(set(ref_dist.index) | set(cur_dist.index))
            p = np.array([ref_dist.get(v, 0) for v in all_values])
            q = np.array([cur_dist.get(v, 0) for v in all_values])
        else:
            # Create histogram bins
            combined = pd.concat([reference, current])
            bins = np.histogram_bin_edges(combined, bins="auto")

            p, _ = np.histogram(reference, bins=bins, density=True)
            q, _ = np.histogram(current, bins=bins, density=True)

        # Avoid zeros
        p = np.maximum(p, 1e-10)
        q = np.maximum(q, 1e-10)

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        # Calculate JS divergence
        m = 0.5 * (p + q)
        js = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))

        if js > 0.2:
            severity = DriftSeverity.HIGH
        elif js > 0.1:
            severity = DriftSeverity.MODERATE
        elif js > 0.05:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE

        return FeatureDriftResult(
            feature=feature,
            method="JS Divergence",
            statistic=float(js),
            p_value=None,
            drift_score=float(js),
            severity=severity,
        )

    def _get_severity_from_psi(self, psi: float) -> DriftSeverity:
        """Get drift severity from PSI score."""
        if psi < 0.1:
            return DriftSeverity.NONE
        elif psi < 0.2:
            return DriftSeverity.LOW
        elif psi < 0.25:
            return DriftSeverity.MODERATE
        elif psi < 0.5:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def _generate_drift_recommendation(
        self, severity: DriftSeverity, drifted_features: list[str]
    ) -> str:
        """Generate recommendation based on drift severity."""
        if severity == DriftSeverity.NONE:
            return "No significant drift detected. Continue monitoring."
        elif severity == DriftSeverity.LOW:
            return "Minor drift detected. Monitor closely for progression."
        elif severity == DriftSeverity.MODERATE:
            return (
                f"Moderate drift detected in {len(drifted_features)} features. "
                "Consider retraining model with recent data."
            )
        elif severity == DriftSeverity.HIGH:
            return (
                f"High drift detected in {len(drifted_features)} features. "
                "Model retraining recommended."
            )
        else:
            return (
                f"Critical drift detected in {len(drifted_features)} features. "
                "Immediate model retraining required."
            )

    def detect_concept_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        methods: list[str] | None = None,
    ) -> ConceptDriftReport:
        """
        Detect concept drift (changes in feature-target relationship).

        Args:
            reference_data: Reference/baseline DataFrame with labels.
            current_data: Current DataFrame to compare with labels.
            feature_columns: Feature columns to analyze.
            target_column: Target/label column.
            methods: Detection methods (["ddm", "adwin"] if None).

        Returns:
            ConceptDriftReport with concept drift analysis.
        """
        methods = methods or ["ddm", "error_rate"]

        if target_column not in reference_data.columns:
            raise ValueError(f"Target column '{target_column}' not in reference data")
        if target_column not in current_data.columns:
            raise ValueError(f"Target column '{target_column}' not in current data")

        drift_points = []
        drift_detected = False

        for method in methods:
            if method == "ddm":
                points = self._detect_ddm(
                    reference_data, current_data, feature_columns, target_column
                )
            elif method == "error_rate":
                points = self._detect_error_rate_drift(
                    reference_data, current_data, feature_columns, target_column
                )
            elif method == "page_hinkley":
                points = self._detect_page_hinkley(
                    reference_data, current_data, feature_columns, target_column
                )
            else:
                continue

            drift_points.extend(points)

        drift_detected = len(drift_points) > 0

        # Calculate performance degradation
        performance_degradation = 0.0
        if drift_points:
            degradations = [
                abs(p.metric_after - p.metric_before) / max(p.metric_before, 0.01)
                for p in drift_points
            ]
            performance_degradation = np.mean(degradations)

        recommendation = self._generate_concept_drift_recommendation(
            drift_detected, performance_degradation
        )

        return ConceptDriftReport(
            detection_method=", ".join(methods),
            target_column=target_column,
            drift_points=drift_points,
            drift_detected=drift_detected,
            performance_degradation=performance_degradation,
            recommendation=recommendation,
        )

    def _detect_ddm(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> list[ConceptDriftPoint]:
        """Detect drift using DDM (Drift Detection Method)."""
        # Calculate error rates
        ref_target = reference_data[target_column]
        cur_target = current_data[target_column]

        # For classification, compare class distributions
        ref_dist = ref_target.value_counts(normalize=True)
        cur_dist = cur_target.value_counts(normalize=True)

        # Calculate distribution difference
        all_classes = set(ref_dist.index) | set(cur_dist.index)
        dist_diff = sum(
            abs(ref_dist.get(c, 0) - cur_dist.get(c, 0)) for c in all_classes
        )

        drift_points = []
        if dist_diff > 0.1:  # Threshold for significant change
            severity = (
                DriftSeverity.HIGH if dist_diff > 0.3
                else DriftSeverity.MODERATE if dist_diff > 0.2
                else DriftSeverity.LOW
            )

            drift_points.append(
                ConceptDriftPoint(
                    timestamp=datetime.now(),
                    detection_method="DDM",
                    severity=severity,
                    metric_before=1.0,
                    metric_after=1.0 - dist_diff,
                    description=f"Target distribution shift: {dist_diff:.2%}",
                )
            )

        return drift_points

    def _detect_error_rate_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> list[ConceptDriftPoint]:
        """Detect drift by comparing error rate proxies."""
        # Use feature correlation with target as a proxy
        ref_correlations = {}
        cur_correlations = {}

        for col in feature_columns:
            if col in reference_data.columns and col in current_data.columns:
                if pd.api.types.is_numeric_dtype(reference_data[col]):
                    try:
                        ref_corr = reference_data[col].corr(
                            reference_data[target_column].astype(float)
                        )
                        cur_corr = current_data[col].corr(
                            current_data[target_column].astype(float)
                        )
                        ref_correlations[col] = ref_corr
                        cur_correlations[col] = cur_corr
                    except (TypeError, ValueError):
                        continue

        drift_points = []
        if ref_correlations:
            # Compare correlation patterns
            correlation_changes = {
                col: abs(ref_correlations[col] - cur_correlations.get(col, 0))
                for col in ref_correlations
            }

            significant_changes = [
                (col, change) for col, change in correlation_changes.items()
                if change > 0.1
            ]

            if significant_changes:
                max_change = max(c[1] for c in significant_changes)
                severity = (
                    DriftSeverity.HIGH if max_change > 0.3
                    else DriftSeverity.MODERATE if max_change > 0.2
                    else DriftSeverity.LOW
                )

                drift_points.append(
                    ConceptDriftPoint(
                        timestamp=datetime.now(),
                        detection_method="Error Rate",
                        severity=severity,
                        metric_before=1.0,
                        metric_after=1.0 - max_change,
                        description=(
                            f"Feature-target correlation shift in "
                            f"{len(significant_changes)} features"
                        ),
                    )
                )

        return drift_points

    def _detect_page_hinkley(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> list[ConceptDriftPoint]:
        """Detect drift using Page-Hinkley test."""
        # Simple implementation using cumulative sum of differences
        ref_mean = reference_data[target_column].astype(float).mean()
        current_values = current_data[target_column].astype(float)

        # Calculate cumulative deviation
        deviations = current_values - ref_mean
        cumsum = np.cumsum(deviations)

        # Detect significant shifts
        threshold = 2 * current_values.std()
        drift_points = []

        if abs(cumsum.iloc[-1]) > threshold:
            drift_points.append(
                ConceptDriftPoint(
                    timestamp=datetime.now(),
                    detection_method="Page-Hinkley",
                    severity=DriftSeverity.MODERATE,
                    metric_before=ref_mean,
                    metric_after=current_values.mean(),
                    description="Mean shift detected in target variable",
                )
            )

        return drift_points

    def _generate_concept_drift_recommendation(
        self, drift_detected: bool, performance_degradation: float
    ) -> str:
        """Generate recommendation for concept drift."""
        if not drift_detected:
            return "No concept drift detected. Continue monitoring."

        if performance_degradation > 0.2:
            return (
                "Significant concept drift detected with major performance impact. "
                "Immediate model retraining with recent labeled data required."
            )
        elif performance_degradation > 0.1:
            return (
                "Concept drift detected with moderate impact. "
                "Consider retraining model or updating feature engineering."
            )
        else:
            return (
                "Minor concept drift detected. "
                "Monitor for progression and plan for model refresh."
            )

    def create_drift_monitor(
        self,
        baseline_data: pd.DataFrame,
        feature_columns: list[str],
        alert_threshold: float = 0.1,
    ) -> DriftMonitor:
        """
        Create a drift monitor for continuous monitoring.

        Args:
            baseline_data: Baseline DataFrame to compare against.
            feature_columns: Columns to monitor for drift.
            alert_threshold: PSI threshold for alerting.

        Returns:
            DriftMonitor instance for ongoing monitoring.
        """
        return DriftMonitor(baseline_data, feature_columns, alert_threshold, self)


class DriftMonitor:
    """Monitor for continuous drift detection."""

    def __init__(
        self,
        baseline_data: pd.DataFrame,
        feature_columns: list[str],
        alert_threshold: float,
        analyzer: DriftAnalyzer,
    ) -> None:
        """
        Initialize the drift monitor.

        Args:
            baseline_data: Baseline DataFrame.
            feature_columns: Columns to monitor.
            alert_threshold: Threshold for alerts.
            analyzer: Parent DriftAnalyzer instance.
        """
        self.baseline_data = baseline_data
        self.feature_columns = feature_columns
        self.alert_threshold = alert_threshold
        self.analyzer = analyzer
        self._history: list[DataDriftReport] = []

    def check_batch(self, new_data: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Check a new batch of data for drift.

        Args:
            new_data: New data batch to compare against baseline.

        Returns:
            List of alert dictionaries if drift exceeds threshold.
        """
        report = self.analyzer.detect_data_drift(
            self.baseline_data,
            new_data,
            self.feature_columns,
        )

        self._history.append(report)

        alerts = []
        for result in report.feature_results:
            if result.drift_score >= self.alert_threshold:
                alerts.append({
                    "feature": result.feature,
                    "drift_score": result.drift_score,
                    "severity": result.severity.value,
                    "method": result.method,
                    "timestamp": datetime.now().isoformat(),
                })

        return alerts

    def get_history(self) -> list[DataDriftReport]:
        """Get history of drift reports."""
        return self._history

    def reset_baseline(self, new_baseline: pd.DataFrame) -> None:
        """Reset the baseline to new data."""
        self.baseline_data = new_baseline
        self._history = []

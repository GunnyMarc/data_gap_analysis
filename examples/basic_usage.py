#!/usr/bin/env python3
"""
Basic Usage Example for Data Gap Analysis Framework

This script demonstrates how to use the data gap analysis framework
to analyze a sample dataset across all four phases.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the framework
from data_gap_analysis import (
    GapAnalyzer,
    GapAnalysisConfig,
    DataProfiler,
    CoverageAnalyzer,
    QualityAnalyzer,
    DriftAnalyzer,
)


def create_sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create sample datasets for demonstration."""
    np.random.seed(42)

    # Main dataset
    n_records = 1000

    main_df = pd.DataFrame({
        "id": range(1, n_records + 1),
        "customer_id": [f"CUST_{i % 200:04d}" for i in range(1, n_records + 1)],
        "email": [
            f"user{i}@example.com" if i % 20 != 0 else None
            for i in range(1, n_records + 1)
        ],
        "country": np.random.choice(["US", "UK", "DE", "FR", "ES"], n_records),
        "transaction_date": pd.date_range("2024-01-01", periods=n_records, freq="h"),
        "amount": np.concatenate([
            np.random.normal(100, 25, n_records - 10),
            [500, 600, 700, 800, 900, -50, -100, 1500, 2000, 2500],  # Outliers
        ]),
        "category": np.random.choice(["Electronics", "Clothing", "Food", "Books"], n_records),
        "last_updated": pd.date_range("2024-01-01", periods=n_records, freq="h") - timedelta(days=60),
    })

    # Reference customer data (for integrity checking)
    reference_customers = pd.DataFrame({
        "id": [f"CUST_{i:04d}" for i in range(0, 180)],
        "name": [f"Customer {i}" for i in range(0, 180)],
    })

    # Historical data (for drift detection)
    historical_df = pd.DataFrame({
        "amount": np.random.normal(95, 20, 1000),
        "category": np.random.choice(["Electronics", "Clothing", "Food", "Books"], 1000),
        "country": np.random.choice(["US", "UK", "DE"], 1000),
    })

    return main_df, reference_customers, historical_df


def run_phase1_profiling(df: pd.DataFrame) -> None:
    """Demonstrate Phase 1: Data Profiling."""
    print("\n" + "=" * 60)
    print("PHASE 1: DATA PROFILING")
    print("=" * 60)

    profiler = DataProfiler(df)

    # Count nulls
    print("\n--- Null Analysis ---")
    null_reports = profiler.count_nulls()
    for report in null_reports:
        if report.null_percentage > 0:
            print(f"  {report.column}: {report.null_percentage:.2f}% null ({report.null_count:,} records)")

    # Assess cardinality
    print("\n--- Cardinality Assessment ---")
    cardinality_reports = profiler.assess_cardinality()
    for report in cardinality_reports:
        print(f"  {report.column}: {report.unique_count:,} unique values ({report.cardinality_type})")

    # Examine data ranges
    print("\n--- Data Ranges (Numeric) ---")
    range_reports = profiler.examine_data_ranges()
    for report in range_reports:
        print(f"  {report.column}: min={report.min_value:.2f}, max={report.max_value:.2f}, "
              f"outliers={report.outliers_count}")

    # Validate data types
    print("\n--- Data Type Validation ---")
    schema = {"id": "int", "email": "email", "amount": "float"}
    type_reports = profiler.validate_data_types(schema)
    for report in type_reports:
        print(f"  {report.column}: {report.validity_percentage:.2f}% valid as {report.expected_type}")


def run_phase2_coverage(df: pd.DataFrame, reference_customers: pd.DataFrame) -> None:
    """Demonstrate Phase 2: Coverage Analysis."""
    print("\n" + "=" * 60)
    print("PHASE 2: COVERAGE ANALYSIS")
    print("=" * 60)

    analyzer = CoverageAnalyzer(df)

    # Geographic coverage
    print("\n--- Geographic Coverage ---")
    geo_coverage = analyzer.analyze_geographic_coverage(
        location_column="country",
        expected_regions=["US", "UK", "DE", "FR", "ES", "IT", "JP"],
    )
    print(f"  Coverage: {geo_coverage.coverage_percentage:.2f}%")
    print(f"  Covered: {geo_coverage.covered_regions}")
    print(f"  Missing: {geo_coverage.missing_regions}")

    # Temporal coverage
    print("\n--- Temporal Coverage ---")
    temporal_coverage = analyzer.analyze_temporal_coverage(
        date_column="transaction_date",
        expected_start="2024-01-01",
        expected_end="2024-02-15",
        expected_frequency="h",
    )
    print(f"  Coverage: {temporal_coverage.coverage_percentage:.2f}%")
    print(f"  Date Range: {temporal_coverage.actual_start.date()} to {temporal_coverage.actual_end.date()}")

    # Entity coverage
    print("\n--- Entity Coverage ---")
    entity_coverage = analyzer.analyze_entity_coverage(
        entity_column="customer_id",
        reference_entities=reference_customers["id"].tolist(),
    )
    print(f"  Coverage: {entity_coverage.coverage_percentage:.2f}%")
    print(f"  Missing Entities: {len(entity_coverage.missing_entities)}")

    # Attribute coverage
    print("\n--- Attribute Coverage ---")
    attr_coverage = analyzer.analyze_attribute_coverage(
        required_columns=["id", "customer_id", "amount", "transaction_date"],
        recommended_columns=["email", "country", "category"],
        optional_columns=["notes", "metadata", "tags"],
    )
    print(f"  Required Coverage: {attr_coverage.required_coverage_percentage:.2f}%")
    print(f"  Recommended Coverage: {attr_coverage.recommended_coverage_percentage:.2f}%")
    print(f"  Missing Required: {attr_coverage.missing_required}")


def run_phase3_quality(df: pd.DataFrame, reference_customers: pd.DataFrame) -> None:
    """Demonstrate Phase 3: Quality Gap Identification."""
    print("\n" + "=" * 60)
    print("PHASE 3: QUALITY GAP IDENTIFICATION")
    print("=" * 60)

    analyzer = QualityAnalyzer(df)

    # Find duplicates
    print("\n--- Duplicate Detection ---")
    dup_report = analyzer.find_duplicates(key_columns=["customer_id"])
    print(f"  Exact Duplicates: {dup_report.exact_duplicate_count} records")
    print(f"  Duplicate Groups: {dup_report.exact_duplicate_groups}")

    # Detect outliers
    print("\n--- Outlier Detection ---")
    outlier_reports = analyzer.detect_outliers(
        numeric_columns=["amount"],
        method="iqr",
        threshold=1.5,
    )
    for report in outlier_reports:
        print(f"  {report.column}: {report.outlier_count} outliers "
              f"({report.outlier_percentage:.2f}%)")
        print(f"    Bounds: [{report.lower_bound:.2f}, {report.upper_bound:.2f}]")

    # Check referential integrity
    print("\n--- Referential Integrity ---")
    ref_issue = analyzer.check_referential_integrity(
        foreign_key="customer_id",
        reference_df=reference_customers,
        reference_key="id",
    )
    print(f"  Orphaned Records: {ref_issue.orphaned_count} ({ref_issue.orphaned_percentage:.2f}%)")

    # Detect stale data
    print("\n--- Stale Data Detection ---")
    stale_report = analyzer.detect_stale_data(
        timestamp_column="last_updated",
        staleness_threshold_days=30,
    )
    print(f"  Stale Records: {stale_report.stale_count} ({stale_report.stale_percentage:.2f}%)")


def run_phase4_drift(df: pd.DataFrame, historical_df: pd.DataFrame) -> None:
    """Demonstrate Phase 4: Model Drift Detection."""
    print("\n" + "=" * 60)
    print("PHASE 4: MODEL DRIFT DETECTION")
    print("=" * 60)

    analyzer = DriftAnalyzer()

    # Data drift detection
    print("\n--- Data Drift Detection ---")
    data_drift = analyzer.detect_data_drift(
        reference_data=historical_df,
        current_data=df[["amount", "category", "country"]],
        feature_columns=["amount"],
        categorical_columns=["category", "country"],
        methods=["psi", "ks_test"],
    )

    print(f"  Overall Drift Score: {data_drift.overall_drift_score:.4f}")
    print(f"  Overall Severity: {data_drift.overall_severity.value}")
    print(f"  Drifted Features: {data_drift.drifted_features}")

    for result in data_drift.feature_results:
        print(f"\n  Feature: {result.feature}")
        print(f"    Method: {result.method}")
        print(f"    Drift Score: {result.drift_score:.4f}")
        print(f"    Severity: {result.severity.value}")

    print(f"\n  Recommendation: {data_drift.recommendation}")


def run_full_analysis() -> None:
    """Run complete gap analysis using the main GapAnalyzer class."""
    print("\n" + "=" * 60)
    print("COMPLETE GAP ANALYSIS")
    print("=" * 60)

    # Create sample data
    df, reference_customers, historical_df = create_sample_data()

    # Initialize the main analyzer
    analyzer = GapAnalyzer(df)

    # Get quick summary
    print("\n--- Quick Summary ---")
    gap_summary = analyzer.get_gap_summary()
    print(f"  Total Records: {gap_summary['total_records']:,}")
    print(f"  Total Columns: {gap_summary['total_columns']}")
    print(f"  Overall Completeness: {gap_summary['overall_completeness']:.2f}%")
    print(f"  Columns with Gaps: {gap_summary['columns_with_gaps']}")

    # Configure full analysis
    config = GapAnalysisConfig(
        # Phase 1 config
        date_columns=["transaction_date"],
        schema={"id": "int", "email": "email", "amount": "float"},
        # Phase 2 config
        location_column="country",
        expected_regions=["US", "UK", "DE", "FR", "ES", "IT", "JP"],
        date_column="transaction_date",
        required_columns=["id", "customer_id", "amount"],
        # Phase 3 config
        duplicate_key_columns=["customer_id"],
        numeric_columns=["amount"],
        timestamp_column="last_updated",
        staleness_threshold_days=30,
        # Phase 4 config
        reference_data=historical_df,
        feature_columns=["amount"],
    )

    # Run full analysis
    report = analyzer.run_full_analysis(config=config)

    # Print summary
    print("\n--- Full Analysis Report ---")
    print(report.summary())

    # Save report
    report.save("gap_analysis_report.json", format="json")
    report.save("gap_analysis_report.html", format="html")
    print("\nReports saved to gap_analysis_report.json and gap_analysis_report.html")


def main():
    """Main function to run all demonstrations."""
    print("=" * 60)
    print("DATA GAP ANALYSIS FRAMEWORK - DEMONSTRATION")
    print("=" * 60)

    # Create sample data
    df, reference_customers, historical_df = create_sample_data()
    print(f"\nSample data created:")
    print(f"  Main dataset: {len(df):,} records, {len(df.columns)} columns")
    print(f"  Reference customers: {len(reference_customers):,} records")
    print(f"  Historical data: {len(historical_df):,} records")

    # Run individual phases
    run_phase1_profiling(df)
    run_phase2_coverage(df, reference_customers)
    run_phase3_quality(df, reference_customers)
    run_phase4_drift(df, historical_df)

    # Run complete analysis
    run_full_analysis()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

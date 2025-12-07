"""
Data Gap Analysis Framework

A comprehensive Python framework for identifying, analyzing, and reporting
data gaps in datasets. This tool helps organizations understand the completeness,
quality, and reliability of their data assets.

Phases:
1. Data Profiling - Analyze null values, distributions, cardinality, ranges
2. Coverage Comparison - Geographic, temporal, entity, and attribute coverage
3. Quality Gap Identification - Format issues, duplicates, outliers, integrity
4. Model Drift Detection - Data drift and concept drift analysis

Example:
    >>> from data_gap_analysis import GapAnalyzer
    >>> import pandas as pd
    >>>
    >>> df = pd.read_csv("your_data.csv")
    >>> analyzer = GapAnalyzer(df)
    >>> report = analyzer.run_full_analysis()
    >>> print(report.summary())
"""

__version__ = "0.1.0"
__author__ = "Data Gap Analysis Team"

from .analyzer import GapAnalyzer, GapAnalysisConfig
from .coverage import (
    AttributeCoverageReport,
    CoverageAnalyzer,
    CoverageReport,
    EntityCoverageReport,
    GeographicCoverageReport,
    TemporalCoverageReport,
)
from .drift import (
    ConceptDriftPoint,
    ConceptDriftReport,
    DataDriftReport,
    DriftAnalyzer,
    DriftMonitor,
    DriftSeverity,
    FeatureDriftResult,
)
from .profiling import (
    CardinalityReport,
    DataProfile,
    DataProfiler,
    DistributionReport,
    NullReport,
    RangeReport,
    TimeSeriesContinuityReport,
    TypeValidationReport,
)
from .quality import (
    DuplicateReport,
    FormatIssue,
    OutlierReport,
    QualityAnalyzer,
    QualityReport,
    ReferentialIntegrityIssue,
    StaleDataReport,
)
from .report import GapAnalysisReport

__all__ = [
    # Main classes
    "GapAnalyzer",
    "GapAnalysisConfig",
    "GapAnalysisReport",
    # Profiling (Phase 1)
    "DataProfiler",
    "DataProfile",
    "NullReport",
    "DistributionReport",
    "CardinalityReport",
    "RangeReport",
    "TimeSeriesContinuityReport",
    "TypeValidationReport",
    # Coverage (Phase 2)
    "CoverageAnalyzer",
    "CoverageReport",
    "GeographicCoverageReport",
    "TemporalCoverageReport",
    "EntityCoverageReport",
    "AttributeCoverageReport",
    # Quality (Phase 3)
    "QualityAnalyzer",
    "QualityReport",
    "FormatIssue",
    "DuplicateReport",
    "OutlierReport",
    "ReferentialIntegrityIssue",
    "StaleDataReport",
    # Drift (Phase 4)
    "DriftAnalyzer",
    "DriftMonitor",
    "DriftSeverity",
    "DataDriftReport",
    "ConceptDriftReport",
    "FeatureDriftResult",
    "ConceptDriftPoint",
]

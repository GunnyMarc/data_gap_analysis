# Data Gap Analysis Framework

A comprehensive Python framework for identifying, analyzing, and reporting data gaps in datasets. This tool helps organizations understand the completeness, quality, and reliability of their data assets.

## What is Gap Analysis in Datasets?

Gap analysis in datasets is the process of identifying:
- **Missing** data - fields or records that should exist but don't
- **Incomplete** data - partial information that lacks required attributes
- **Inconsistent** data - conflicting or contradictory information across sources

The core principle is to **compare what you have against what you need or expect**.

## How to Approach Gap Analysis

### 1. Define Your Reference Standard

Start by establishing what "complete" looks like. This could be a:
- **Schema definition** - the expected structure and data types
- **Business requirements** - what data is needed for operations
- **Regulatory standards** - compliance requirements (GDPR, HIPAA, etc.)
- **Benchmark dataset** - a known-good reference to compare against

You need a baseline to measure gaps against.

### 2. Quantify and Prioritize

Calculate gap metrics like:
- **Completeness percentages** - what proportion of expected data exists
- **Quality scores** - how accurate and consistent the data is
- **Coverage metrics** - geographic, temporal, and entity coverage

Then prioritize based on business impact. A 5% gap in a critical field may be more important than a 30% gap in an optional one.

### 3. Document and Remediate

Create a gap report mapping each issue to its:
- **Source** - where the gap originates
- **Impact** - business consequences of the gap
- **Proposed fix** - whether that's data collection changes, imputation strategies, or accepting certain limitations

---

## Framework Phases

This framework implements gap analysis in four distinct phases:

---

## Phase 1: Data Profiling

The foundation of gap analysis - understanding what data you actually have.

### Capabilities

| Analysis Type | Description |
|--------------|-------------|
| **Null Counting** | Identify missing values per column with percentages and patterns |
| **Value Distribution** | Statistical analysis of value frequencies and distributions |
| **Cardinality Assessment** | Count unique values to understand data diversity |
| **Data Range Examination** | Min/max values, outlier detection for numeric fields |
| **Time Series Continuity** | Detect gaps in temporal sequences |
| **Data Type Validation** | Verify values match expected data types |

### Usage

```python
from data_gap_analysis.profiling import DataProfiler

profiler = DataProfiler(dataframe)
profile = profiler.generate_profile()

# Individual analyses
null_report = profiler.count_nulls()
distribution = profiler.analyze_value_distribution()
cardinality = profiler.assess_cardinality()
ranges = profiler.examine_data_ranges()
time_gaps = profiler.check_time_series_continuity(date_column="timestamp")
type_issues = profiler.validate_data_types(schema)
```

### Output Example

```
Column: customer_email
  - Null Count: 1,234 (5.2%)
  - Cardinality: 18,456 unique values
  - Type Validation: 99.1% valid email format
  - Distribution: Normal (skewness: 0.12)
```

---

## Phase 2: Data Coverage Comparison

Analyze how well your data represents the full scope of what should be covered.

### Coverage Dimensions

| Coverage Type | Description |
|--------------|-------------|
| **Geographic Coverage** | Spatial completeness across regions, countries, coordinates |
| **Temporal Coverage** | Time period completeness, gap detection in date ranges |
| **Entity Coverage** | Completeness of business entities (customers, products, etc.) |
| **Attribute Coverage** | Field-level completeness against schema requirements |

### Usage

```python
from data_gap_analysis.coverage import CoverageAnalyzer

analyzer = CoverageAnalyzer(dataframe)

# Geographic coverage against expected regions
geo_coverage = analyzer.analyze_geographic_coverage(
    location_column="country",
    expected_regions=["US", "UK", "DE", "FR", "JP"]
)

# Temporal coverage
temporal_coverage = analyzer.analyze_temporal_coverage(
    date_column="transaction_date",
    expected_start="2020-01-01",
    expected_end="2024-12-31",
    frequency="daily"
)

# Entity coverage
entity_coverage = analyzer.analyze_entity_coverage(
    entity_column="customer_id",
    reference_entities=master_customer_list
)

# Attribute coverage against schema
attribute_coverage = analyzer.analyze_attribute_coverage(
    required_columns=["id", "name", "email", "created_at"],
    recommended_columns=["phone", "address", "preferences"]
)
```

### Output Example

```
Geographic Coverage Report:
  - Expected regions: 5
  - Covered regions: 4
  - Missing: ["JP"]
  - Coverage percentage: 80.0%

Temporal Coverage Report:
  - Expected date range: 2020-01-01 to 2024-12-31
  - Actual date range: 2020-03-15 to 2024-11-30
  - Missing periods: 74 days at start, 31 days at end
  - Gap days within range: 12
  - Coverage percentage: 93.5%
```

---

## Phase 3: Quality Gap Identification

Detect data quality issues that impact reliability and usability.

### Quality Dimensions

| Issue Type | Description |
|------------|-------------|
| **Inconsistent Formats/Units** | Detect mixed date formats, unit variations, encoding issues |
| **Duplicate Records** | Find exact and fuzzy duplicates |
| **Outliers Indicating Errors** | Statistical outlier detection suggesting data entry errors |
| **Referential Integrity Issues** | Foreign keys that don't match parent tables |
| **Stale Data** | Records that haven't been updated within expected timeframes |

### Usage

```python
from data_gap_analysis.quality import QualityAnalyzer

analyzer = QualityAnalyzer(dataframe)

# Format consistency
format_issues = analyzer.detect_inconsistent_formats(
    columns=["date", "phone", "currency"]
)

# Duplicate detection
duplicates = analyzer.find_duplicates(
    key_columns=["email"],
    fuzzy_columns=["name", "address"],
    similarity_threshold=0.85
)

# Outlier detection
outliers = analyzer.detect_outliers(
    numeric_columns=["amount", "quantity"],
    method="iqr",  # or "zscore", "isolation_forest"
    threshold=1.5
)

# Referential integrity
ref_issues = analyzer.check_referential_integrity(
    foreign_key="customer_id",
    reference_df=customers_df,
    reference_key="id"
)

# Stale data detection
stale_records = analyzer.detect_stale_data(
    timestamp_column="last_updated",
    staleness_threshold_days=90
)
```

### Output Example

```
Quality Gap Report:

  Inconsistent Formats:
    - date: 3 formats detected (YYYY-MM-DD: 85%, MM/DD/YYYY: 12%, DD-MM-YYYY: 3%)
    - phone: 5 formats detected

  Duplicates:
    - Exact duplicates: 234 records
    - Fuzzy duplicates: 89 record groups

  Outliers:
    - amount: 156 outliers detected (0.7% of records)
    - quantity: 23 outliers detected (0.1% of records)

  Referential Integrity:
    - customer_id: 45 orphaned records (foreign keys without parents)

  Stale Data:
    - 1,234 records not updated in 90+ days (5.2% of dataset)
```

---

## Phase 4: Model Drift Computation

Monitor and detect changes in data distributions that can impact ML model performance.

### Drift Types

| Drift Type | Description |
|------------|-------------|
| **Data Drift** | Changes in input feature distributions over time |
| **Concept Drift** | Changes in the relationship between features and target |

### Usage

```python
from data_gap_analysis.drift import DriftAnalyzer

analyzer = DriftAnalyzer()

# Data drift detection
data_drift = analyzer.detect_data_drift(
    reference_data=training_data,
    current_data=production_data,
    feature_columns=["age", "income", "purchase_frequency"],
    methods=["ks_test", "psi", "wasserstein"]
)

# Concept drift detection
concept_drift = analyzer.detect_concept_drift(
    reference_data=historical_labeled_data,
    current_data=recent_labeled_data,
    feature_columns=["age", "income", "purchase_frequency"],
    target_column="churn",
    methods=["ddm", "adwin", "page_hinkley"]
)

# Continuous monitoring
drift_monitor = analyzer.create_drift_monitor(
    baseline_data=training_data,
    feature_columns=features,
    alert_threshold=0.1
)

# Check new batch
alerts = drift_monitor.check_batch(new_data_batch)
```

### Drift Detection Methods

#### Data Drift Methods
- **Kolmogorov-Smirnov Test (KS)** - Non-parametric test for distribution differences
- **Population Stability Index (PSI)** - Measures distribution shift magnitude
- **Wasserstein Distance** - Earth mover's distance between distributions
- **Jensen-Shannon Divergence** - Symmetric measure of distribution similarity

#### Concept Drift Methods
- **DDM (Drift Detection Method)** - Monitors error rate changes
- **ADWIN (Adaptive Windowing)** - Detects changes in data streams
- **Page-Hinkley Test** - Sequential analysis for mean changes

### Output Example

```
Data Drift Report:
  Reference period: 2024-01-01 to 2024-06-30
  Current period: 2024-07-01 to 2024-12-31

  Feature Drift Scores:
    - age: PSI=0.08 (Low drift)
    - income: PSI=0.23 (Moderate drift) ⚠️
    - purchase_frequency: PSI=0.45 (High drift) 🚨

  Overall Assessment: DRIFT DETECTED
  Recommendation: Retrain model with recent data

Concept Drift Report:
  Detection Method: ADWIN

  Drift Points Detected:
    - 2024-08-15: Significant concept shift
    - 2024-10-01: Minor concept adjustment

  Model Performance Impact:
    - Accuracy degradation: -12.3%
    - Recommendation: Investigate feature relationships
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/GunnyMarc/data_gap_analysis.git
cd data_gap_analysis

# Install with pip (Python 3.14+)
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Requirements

- Python 3.14+
- pandas >= 2.0
- numpy >= 1.24
- scipy >= 1.11
- scikit-learn >= 1.3

## Quick Start

```python
from data_gap_analysis import GapAnalyzer
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Initialize the analyzer
analyzer = GapAnalyzer(df)

# Run complete gap analysis
report = analyzer.run_full_analysis(
    schema=expected_schema,
    reference_data=benchmark_df,
    date_column="created_at"
)

# Generate report
report.to_html("gap_analysis_report.html")
report.summary()
```

## Project Structure

```
data_gap_analysis/
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/
│   └── data_gap_analysis/
│       ├── __init__.py
│       ├── profiling.py      # Phase 1: Data Profiling
│       ├── coverage.py       # Phase 2: Coverage Comparison
│       ├── quality.py        # Phase 3: Quality Gaps
│       ├── drift.py          # Phase 4: Model Drift
│       ├── analyzer.py       # Main orchestration
│       ├── report.py         # Report generation
│       └── utils.py          # Utility functions
├── tests/
│   ├── test_profiling.py
│   ├── test_coverage.py
│   ├── test_quality.py
│   └── test_drift.py
└── examples/
    ├── basic_usage.py
    └── sample_data/
```

## License

MIT License

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

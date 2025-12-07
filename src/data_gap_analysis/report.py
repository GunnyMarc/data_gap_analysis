"""
Report Generation Module

This module provides classes and functions for generating
gap analysis reports in various formats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .coverage import CoverageReport
from .drift import ConceptDriftReport, DataDriftReport
from .profiling import DataProfile
from .quality import QualityReport


@dataclass
class GapAnalysisReport:
    """
    Comprehensive gap analysis report combining all phases.

    This class aggregates results from all four phases of gap analysis
    and provides methods for generating reports in various formats.
    """

    data_profile: DataProfile | None = None
    coverage_report: CoverageReport | None = None
    quality_report: QualityReport | None = None
    data_drift_report: DataDriftReport | None = None
    concept_drift_report: ConceptDriftReport | None = None
    report_timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """
        Generate a text summary of the gap analysis.

        Returns:
            Formatted string with analysis summary.
        """
        lines = [
            "=" * 70,
            "GAP ANALYSIS REPORT",
            "=" * 70,
            f"Generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Phase 1: Data Profile Summary
        if self.data_profile:
            lines.extend([
                "PHASE 1: DATA PROFILING",
                "-" * 50,
                f"Total Rows: {self.data_profile.total_rows:,}",
                f"Total Columns: {self.data_profile.total_columns}",
                "",
                "Null Analysis:",
            ])

            high_null_cols = [
                r for r in self.data_profile.null_reports
                if r.null_percentage > 5
            ]
            if high_null_cols:
                for report in sorted(
                    high_null_cols, key=lambda x: x.null_percentage, reverse=True
                )[:5]:
                    lines.append(
                        f"  - {report.column}: {report.null_percentage:.1f}% null"
                    )
            else:
                lines.append("  All columns have <5% null values")

            lines.append("")

        # Phase 2: Coverage Summary
        if self.coverage_report:
            lines.extend([
                "PHASE 2: COVERAGE ANALYSIS",
                "-" * 50,
            ])

            if self.coverage_report.geographic_coverage:
                geo = self.coverage_report.geographic_coverage
                lines.append(
                    f"Geographic Coverage: {geo.coverage_percentage:.1f}% "
                    f"({len(geo.covered_regions)}/{len(geo.expected_regions)} regions)"
                )

            if self.coverage_report.temporal_coverage:
                temp = self.coverage_report.temporal_coverage
                lines.append(
                    f"Temporal Coverage: {temp.coverage_percentage:.1f}%"
                )

            if self.coverage_report.entity_coverage:
                ent = self.coverage_report.entity_coverage
                lines.append(
                    f"Entity Coverage: {ent.coverage_percentage:.1f}% "
                    f"({ent.covered_entities}/{ent.expected_entities} entities)"
                )

            if self.coverage_report.attribute_coverage:
                attr = self.coverage_report.attribute_coverage
                lines.append(
                    f"Attribute Coverage: {attr.required_coverage_percentage:.1f}% "
                    "(required columns)"
                )

            lines.append("")

        # Phase 3: Quality Summary
        if self.quality_report:
            lines.extend([
                "PHASE 3: QUALITY GAPS",
                "-" * 50,
            ])

            if self.quality_report.format_issues:
                lines.append(
                    f"Format Issues: {len(self.quality_report.format_issues)} columns"
                )

            if self.quality_report.duplicate_reports:
                for report in self.quality_report.duplicate_reports:
                    lines.append(
                        f"Duplicates: {report.exact_duplicate_count} exact, "
                        f"{report.fuzzy_duplicate_count} fuzzy"
                    )

            if self.quality_report.outlier_reports:
                total_outliers = sum(
                    r.outlier_count for r in self.quality_report.outlier_reports
                )
                lines.append(f"Outliers: {total_outliers} total across all columns")

            if self.quality_report.referential_integrity_issues:
                total_orphans = sum(
                    i.orphaned_count
                    for i in self.quality_report.referential_integrity_issues
                )
                lines.append(f"Referential Integrity: {total_orphans} orphaned records")

            if self.quality_report.stale_data_reports:
                for report in self.quality_report.stale_data_reports:
                    lines.append(
                        f"Stale Data: {report.stale_count} records "
                        f"({report.stale_percentage:.1f}%)"
                    )

            lines.append("")

        # Phase 4: Drift Summary
        if self.data_drift_report or self.concept_drift_report:
            lines.extend([
                "PHASE 4: MODEL DRIFT",
                "-" * 50,
            ])

            if self.data_drift_report:
                lines.append(
                    f"Data Drift: {self.data_drift_report.overall_severity.value} "
                    f"(score: {self.data_drift_report.overall_drift_score:.4f})"
                )
                if self.data_drift_report.drifted_features:
                    lines.append(
                        f"  Drifted Features: {len(self.data_drift_report.drifted_features)}"
                    )

            if self.concept_drift_report:
                lines.append(
                    f"Concept Drift: {'Detected' if self.concept_drift_report.drift_detected else 'Not Detected'}"
                )
                if self.concept_drift_report.drift_detected:
                    lines.append(
                        f"  Performance Degradation: "
                        f"{self.concept_drift_report.performance_degradation:.1%}"
                    )

        lines.extend([
            "",
            "=" * 70,
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the report to a dictionary.

        Returns:
            Dictionary representation of the report.
        """
        result = {
            "report_timestamp": self.report_timestamp.isoformat(),
            "metadata": self.metadata,
        }

        if self.data_profile:
            result["data_profile"] = {
                "total_rows": self.data_profile.total_rows,
                "total_columns": self.data_profile.total_columns,
                "null_reports": [
                    {
                        "column": r.column,
                        "null_count": r.null_count,
                        "null_percentage": r.null_percentage,
                    }
                    for r in self.data_profile.null_reports
                ],
                "cardinality_reports": [
                    {
                        "column": r.column,
                        "unique_count": r.unique_count,
                        "cardinality_type": r.cardinality_type,
                    }
                    for r in self.data_profile.cardinality_reports
                ],
            }

        if self.coverage_report:
            coverage_dict = {}
            if self.coverage_report.geographic_coverage:
                geo = self.coverage_report.geographic_coverage
                coverage_dict["geographic"] = {
                    "coverage_percentage": geo.coverage_percentage,
                    "missing_regions": geo.missing_regions,
                }
            if self.coverage_report.temporal_coverage:
                temp = self.coverage_report.temporal_coverage
                coverage_dict["temporal"] = {
                    "coverage_percentage": temp.coverage_percentage,
                    "gaps_at_start_days": temp.gaps_at_start_days,
                    "gaps_at_end_days": temp.gaps_at_end_days,
                }
            if self.coverage_report.entity_coverage:
                ent = self.coverage_report.entity_coverage
                coverage_dict["entity"] = {
                    "coverage_percentage": ent.coverage_percentage,
                    "missing_count": len(ent.missing_entities),
                }
            if self.coverage_report.attribute_coverage:
                attr = self.coverage_report.attribute_coverage
                coverage_dict["attribute"] = {
                    "required_coverage": attr.required_coverage_percentage,
                    "missing_required": attr.missing_required,
                }
            result["coverage_report"] = coverage_dict

        if self.quality_report:
            result["quality_report"] = {
                "format_issues": len(self.quality_report.format_issues),
                "duplicates": sum(
                    r.exact_duplicate_count + r.fuzzy_duplicate_count
                    for r in self.quality_report.duplicate_reports
                ),
                "outliers": sum(
                    r.outlier_count for r in self.quality_report.outlier_reports
                ),
                "referential_integrity_issues": sum(
                    i.orphaned_count
                    for i in self.quality_report.referential_integrity_issues
                ),
                "stale_records": sum(
                    r.stale_count for r in self.quality_report.stale_data_reports
                ),
            }

        if self.data_drift_report:
            result["data_drift"] = {
                "overall_score": self.data_drift_report.overall_drift_score,
                "overall_severity": self.data_drift_report.overall_severity.value,
                "drifted_features": self.data_drift_report.drifted_features,
                "recommendation": self.data_drift_report.recommendation,
            }

        if self.concept_drift_report:
            result["concept_drift"] = {
                "detected": self.concept_drift_report.drift_detected,
                "performance_degradation": self.concept_drift_report.performance_degradation,
                "recommendation": self.concept_drift_report.recommendation,
            }

        return result

    def to_json(self, indent: int = 2) -> str:
        """
        Convert the report to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_html(self, output_path: str | Path | None = None) -> str:
        """
        Generate an HTML report.

        Args:
            output_path: Optional path to save the HTML file.

        Returns:
            HTML string.
        """
        html_content = self._generate_html()

        if output_path:
            path = Path(output_path)
            path.write_text(html_content)

        return html_content

    def _generate_html(self) -> str:
        """Generate HTML content for the report."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gap Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .metric {{
            display: inline-block;
            background: #ecf0f1;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 4px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
        }}
        .severity-none {{ color: #27ae60; }}
        .severity-low {{ color: #f39c12; }}
        .severity-moderate {{ color: #e67e22; }}
        .severity-high {{ color: #e74c3c; }}
        .severity-critical {{ color: #c0392b; font-weight: bold; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Gap Analysis Report</h1>
        <p class="timestamp">Generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

        # Phase 1: Data Profile
        if self.data_profile:
            html += """
        <h2>Phase 1: Data Profiling</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            <div class="metric">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total Columns</div>
            </div>
        </div>
""".format(self.data_profile.total_rows, self.data_profile.total_columns)

            # Null analysis table
            if self.data_profile.null_reports:
                high_null = [r for r in self.data_profile.null_reports if r.null_percentage > 0]
                if high_null:
                    html += """
        <h3>Null Value Analysis</h3>
        <table>
            <tr>
                <th>Column</th>
                <th>Null Count</th>
                <th>Null %</th>
            </tr>
"""
                    for report in sorted(high_null, key=lambda x: x.null_percentage, reverse=True)[:10]:
                        html += f"""
            <tr>
                <td>{report.column}</td>
                <td>{report.null_count:,}</td>
                <td>{report.null_percentage:.2f}%</td>
            </tr>
"""
                    html += "        </table>\n"

        # Phase 2: Coverage
        if self.coverage_report:
            html += """
        <h2>Phase 2: Coverage Analysis</h2>
"""
            if self.coverage_report.geographic_coverage:
                geo = self.coverage_report.geographic_coverage
                html += f"""
        <h3>Geographic Coverage</h3>
        <div class="metric">
            <div class="metric-value">{geo.coverage_percentage:.1f}%</div>
            <div class="metric-label">Coverage</div>
        </div>
        <p>Covered {len(geo.covered_regions)} of {len(geo.expected_regions)} expected regions</p>
"""
                if geo.missing_regions:
                    html += f"<p><strong>Missing:</strong> {', '.join(geo.missing_regions)}</p>\n"

            if self.coverage_report.temporal_coverage:
                temp = self.coverage_report.temporal_coverage
                html += f"""
        <h3>Temporal Coverage</h3>
        <div class="metric">
            <div class="metric-value">{temp.coverage_percentage:.1f}%</div>
            <div class="metric-label">Coverage</div>
        </div>
        <p>Date Range: {temp.actual_start.date()} to {temp.actual_end.date()}</p>
"""

        # Phase 3: Quality
        if self.quality_report:
            html += """
        <h2>Phase 3: Quality Gap Identification</h2>
"""
            if self.quality_report.format_issues:
                html += f"""
        <div class="metric">
            <div class="metric-value">{len(self.quality_report.format_issues)}</div>
            <div class="metric-label">Format Issues</div>
        </div>
"""
            if self.quality_report.outlier_reports:
                total_outliers = sum(r.outlier_count for r in self.quality_report.outlier_reports)
                html += f"""
        <div class="metric">
            <div class="metric-value">{total_outliers:,}</div>
            <div class="metric-label">Outliers Detected</div>
        </div>
"""

        # Phase 4: Drift
        if self.data_drift_report:
            severity_class = f"severity-{self.data_drift_report.overall_severity.value}"
            html += f"""
        <h2>Phase 4: Model Drift Detection</h2>
        <h3>Data Drift</h3>
        <div class="metric">
            <div class="metric-value {severity_class}">{self.data_drift_report.overall_severity.value.upper()}</div>
            <div class="metric-label">Drift Severity</div>
        </div>
        <div class="metric">
            <div class="metric-value">{self.data_drift_report.overall_drift_score:.4f}</div>
            <div class="metric-label">Drift Score</div>
        </div>
"""
            if self.data_drift_report.drifted_features:
                html += f"""
        <p><strong>Drifted Features:</strong> {', '.join(self.data_drift_report.drifted_features)}</p>
"""
            if self.data_drift_report.recommendation:
                html += f"""
        <p><strong>Recommendation:</strong> {self.data_drift_report.recommendation}</p>
"""

        html += """
    </div>
</body>
</html>
"""
        return html

    def save(self, output_path: str | Path, format: str = "json") -> None:
        """
        Save the report to a file.

        Args:
            output_path: Path to save the report.
            format: Output format ("json", "html", "txt").
        """
        path = Path(output_path)

        if format == "json":
            path.write_text(self.to_json())
        elif format == "html":
            self.to_html(path)
        elif format == "txt":
            path.write_text(self.summary())
        else:
            raise ValueError(f"Unsupported format: {format}")

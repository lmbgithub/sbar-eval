"""The SBAR Completeness Score.

Two numbers, deliberately kept separate, because collapsing them hides the
failure the SBAR standard exists to prevent.

    section coverage  = recovered sections / 4        (per report, then averaged)
    strict complete   = share of reports with all 4 recovered

Section coverage is the smooth signal you tune a prompt against. Strict
completeness is the one that matters clinically: a handover missing its
Recommendation is not "75% good", it is a handover the receiving nurse cannot
act on. Reporting only the average lets a model that reliably drops one section
look like a strong performer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from sbar_eval.schema import SBARReport, Section

SECTION_COUNT = len(Section.ordered())


@dataclass(frozen=True)
class ReportScore:
    """Completeness of a single report."""

    source_id: str
    recovered: tuple[Section, ...]
    missing: tuple[Section, ...]

    @property
    def coverage(self) -> float:
        """Recovered sections over four."""
        return len(self.recovered) / SECTION_COUNT

    @property
    def is_complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "coverage": round(self.coverage, 4),
            "complete": self.is_complete,
            "recovered": [s.value for s in self.recovered],
            "missing": [s.value for s in self.missing],
        }


@dataclass(frozen=True)
class CorpusScore:
    """Completeness across a set of reports."""

    reports: tuple[ReportScore, ...]

    @property
    def count(self) -> int:
        return len(self.reports)

    @property
    def mean_coverage(self) -> float:
        if not self.reports:
            return 0.0
        return sum(r.coverage for r in self.reports) / len(self.reports)

    @property
    def strict_completeness(self) -> float:
        """Share of reports with all four sections recovered."""
        if not self.reports:
            return 0.0
        return sum(1 for r in self.reports if r.is_complete) / len(self.reports)

    def missing_by_section(self) -> dict[Section, int]:
        """How often each section is the one that goes missing.

        The single most useful diagnostic here: a prompt that drops
        Recommendation on 40% of handovers has a specific, fixable defect, and
        the aggregate score will never tell you which section to go fix.
        """
        counts = {section: 0 for section in Section.ordered()}
        for report in self.reports:
            for section in report.missing:
                counts[section] += 1
        return counts

    def meets(self, threshold: float, *, strict: bool = True) -> bool:
        """Whether the corpus clears a completeness threshold."""
        value = self.strict_completeness if strict else self.mean_coverage
        return value >= threshold

    def to_dict(self, *, include_reports: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "reports": self.count,
            "mean_coverage": round(self.mean_coverage, 4),
            "strict_completeness": round(self.strict_completeness, 4),
            "missing_by_section": {
                s.value: n for s, n in self.missing_by_section().items()
            },
        }
        if include_reports:
            data["per_report"] = [r.to_dict() for r in self.reports]
        return data


def score_report(report: SBARReport) -> ReportScore:
    """Score one report's completeness."""
    return ReportScore(
        source_id=report.source_id,
        recovered=report.populated(),
        missing=report.missing(),
    )


def score_corpus(reports: Sequence[SBARReport]) -> CorpusScore:
    """Score a set of reports."""
    return CorpusScore(reports=tuple(score_report(r) for r in reports))

import pytest

from sbar_eval.completeness import score_corpus, score_report
from sbar_eval.schema import SBARReport, Section

FULL = SBARReport("Bed 4, chest pain", "Hypertension", "Obs stable", "Repeat ECG", source_id="full")
NO_REC = SBARReport("Bed 4, chest pain", "Hypertension", "Obs stable", None, source_id="no-rec")
EMPTY = SBARReport(source_id="empty")


def test_full_report_scores_one():
    score = score_report(FULL)
    assert score.coverage == 1.0
    assert score.is_complete
    assert score.missing == ()


def test_missing_one_section_is_three_quarters():
    score = score_report(NO_REC)
    assert score.coverage == 0.75
    assert not score.is_complete
    assert score.missing == (Section.RECOMMENDATION,)


def test_empty_report_scores_zero():
    assert score_report(EMPTY).coverage == 0.0


def test_corpus_separates_coverage_from_strict_completeness():
    # The headline case for keeping both numbers: every report is 75% covered,
    # so mean coverage looks respectable while not one handover is actionable.
    corpus = score_corpus([NO_REC] * 4)
    assert corpus.mean_coverage == 0.75
    assert corpus.strict_completeness == 0.0


def test_strict_completeness_counts_whole_reports():
    corpus = score_corpus([FULL, FULL, NO_REC, EMPTY])
    assert corpus.strict_completeness == 0.5
    assert corpus.mean_coverage == pytest.approx((1.0 + 1.0 + 0.75 + 0.0) / 4)


def test_missing_by_section_localizes_the_defect():
    corpus = score_corpus([NO_REC, NO_REC, FULL])
    missing = corpus.missing_by_section()
    assert missing[Section.RECOMMENDATION] == 2
    assert missing[Section.SITUATION] == 0


def test_empty_corpus_is_safe():
    corpus = score_corpus([])
    assert corpus.count == 0
    assert corpus.mean_coverage == 0.0
    assert corpus.strict_completeness == 0.0
    assert not corpus.meets(0.8)


def test_meets_threshold_strict_and_soft():
    corpus = score_corpus([NO_REC] * 4)
    assert not corpus.meets(0.8, strict=True)     # 0% complete
    assert corpus.meets(0.7, strict=False)        # 75% coverage
    assert not corpus.meets(0.8, strict=False)


def test_serialization_shape():
    payload = score_corpus([FULL, NO_REC]).to_dict()
    assert payload["reports"] == 2
    assert payload["missing_by_section"]["recommendation"] == 1
    assert len(payload["per_report"]) == 2
    assert "per_report" not in score_corpus([FULL]).to_dict(include_reports=False)

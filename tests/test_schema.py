import pytest

from sbar_eval.schema import SBARReport, Section, is_populated


def test_sections_are_ordered_canonically():
    assert [s.value for s in Section.ordered()] == [
        "situation", "background", "assessment", "recommendation"
    ]
    assert [s.initial for s in Section.ordered()] == ["S", "B", "A", "R"]


@pytest.mark.parametrize("value", ["Chest pain since 06:00", "BP 90/60", "abc"])
def test_real_content_is_populated(value):
    assert is_populated(value)


@pytest.mark.parametrize(
    "value",
    [None, "", "   ", "\n\t", "N/A", "n/a", "none", "Not stated", "unknown",
     "-", "...", "[absent]", "Not applicable.", "NULL"],
)
def test_placeholders_and_blanks_are_not_populated(value):
    assert not is_populated(value)


def test_a_section_holding_a_refusal_does_not_count():
    # The failure mode this guards: a model returns all four keys, three of them
    # saying "not mentioned", and a naive key-count reports 100% completeness.
    report = SBARReport(
        situation="Bed 4, chest pain",
        background="not stated",
        assessment="N/A",
        recommendation="  ",
    )
    assert report.populated() == (Section.SITUATION,)
    assert not report.is_complete


def test_complete_report():
    report = SBARReport("s content", "b content", "a content", "r content")
    assert report.is_complete
    assert report.missing() == ()
    assert len(report.populated()) == 4


def test_missing_preserves_canonical_order():
    report = SBARReport(situation="here", assessment="here")
    assert report.missing() == (Section.BACKGROUND, Section.RECOMMENDATION)


def test_get_by_section():
    report = SBARReport(situation="x")
    assert report.get(Section.SITUATION) == "x"
    assert report.get(Section.BACKGROUND) is None


def test_iteration_yields_all_four_in_order():
    pairs = list(SBARReport(situation="x"))
    assert [s for s, _ in pairs] == list(Section.ordered())
    assert len(pairs) == 4


def test_content_below_the_floor_is_treated_as_absent():
    # A one-character section is not clinical content. The floor keeps a model
    # that emits stray characters from scoring as complete.
    assert not is_populated("s")
    assert is_populated("abc")


def test_from_dict_accepts_full_keys():
    report = SBARReport.from_dict(
        {
            "situation": "Bed 4, chest pain since 06:00",
            "background": "Hypertension, on ramipril",
            "assessment": "Obs stable, pain score 4",
            "recommendation": "Repeat ECG in one hour",
        }
    )
    assert report.is_complete


def test_from_dict_accepts_single_letter_keys():
    report = SBARReport.from_dict(
        {
            "S": "Bed 4, chest pain",
            "B": "Hypertension",
            "A": "Obs stable",
            "R": "Repeat ECG",
        }
    )
    assert report.is_complete


def test_from_dict_accepts_mixed_case_and_abbreviations():
    report = SBARReport.from_dict(
        {
            "Situation": "Bed 4, chest pain",
            "back": "Hypertension",
            "ASSE": "Obs stable",
            "Recommendation": "Repeat ECG",
        }
    )
    assert report.is_complete


def test_from_dict_ignores_unknown_keys():
    report = SBARReport.from_dict({"situation": "s", "notes": "ignored"})
    assert report.situation == "s"
    assert report.background is None


def test_from_dict_handles_null_values():
    report = SBARReport.from_dict({"situation": None, "background": "b"})
    assert not is_populated(report.situation)
    assert report.background == "b"


def test_roundtrip_through_dict():
    original = SBARReport("Bed 4", "Hypertension", "Obs stable", "Repeat ECG", source_id="u1")
    assert SBARReport.from_dict(original.to_dict()).to_dict() == original.to_dict()

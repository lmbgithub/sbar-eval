import json

from sbar_eval.prompts import ALL as ALL_PROMPTS
from sbar_eval.prompts import BASELINE, GROUNDED, get
from sbar_eval.schema import Section
from sbar_eval.structurer import CueStructurer, LLMStructurer, extract_json, structure_all

HANDOVER = (
    "This is bed four, Mr Alvarez, sixty-eight years old, came in with chest pain "
    "at six this morning. He has a history of hypertension and is allergic to "
    "penicillin. His obs are stable, blood pressure one thirty over eighty, sats "
    "ninety-seven on air. Please repeat the ECG in an hour and chase the troponin."
)


def test_cue_structurer_fills_all_four_on_a_canonical_handover():
    report = CueStructurer().structure(HANDOVER, source_id="h1")
    assert report.is_complete
    assert report.source_id == "h1"


def test_cue_structurer_routes_content_to_the_right_sections():
    report = CueStructurer().structure(HANDOVER)
    assert "chest pain" in report.situation.lower()
    assert "hypertension" in report.background.lower()
    assert "obs are stable" in report.assessment.lower()
    assert "repeat the ecg" in report.recommendation.lower()


def test_cue_structurer_on_empty_input_returns_an_empty_report():
    report = CueStructurer().structure("")
    assert not report.populated()


def test_cue_structurer_carries_topic_forward_across_sentences():
    # A sentence with no cue words belongs to whatever the speaker was already
    # talking about, which is how handover speech actually flows.
    text = "Please chase the troponin. Then let the registrar know."
    report = CueStructurer().structure(text)
    assert "registrar" in (report.recommendation or "").lower()


def test_cue_structurer_is_deterministic():
    a = CueStructurer().structure(HANDOVER).to_dict()
    b = CueStructurer().structure(HANDOVER).to_dict()
    assert a == b


def test_structure_all_preserves_ids_and_order():
    pairs = [("a", HANDOVER), ("b", HANDOVER)]
    reports = structure_all(CueStructurer(), pairs)
    assert [r.source_id for r in reports] == ["a", "b"]


# -- JSON extraction -----------------------------------------------------


def test_extract_plain_json():
    assert extract_json('{"situation": "x"}') == {"situation": "x"}


def test_extract_json_from_a_fenced_block():
    raw = 'Here you go:\n```json\n{"situation": "x", "background": "y"}\n```\nHope that helps!'
    assert extract_json(raw) == {"situation": "x", "background": "y"}


def test_extract_json_with_nested_objects():
    raw = 'text {"situation": "x", "meta": {"conf": 0.9}} trailing'
    assert extract_json(raw)["meta"] == {"conf": 0.9}


def test_extract_json_ignores_braces_inside_strings():
    raw = '{"situation": "patient said {weird}", "background": "b"}'
    assert extract_json(raw)["situation"] == "patient said {weird}"


def test_extract_json_handles_escaped_quotes():
    raw = r'{"situation": "he said \"ouch\"", "background": "b"}'
    assert extract_json(raw)["situation"] == 'he said "ouch"'


def test_extract_json_skips_a_malformed_first_object():
    raw = '{not valid} then {"situation": "x"}'
    assert extract_json(raw) == {"situation": "x"}


def test_extract_json_returns_none_when_absent():
    assert extract_json("no json here") is None
    assert extract_json("") is None
    assert extract_json("[1, 2, 3]") is None


# -- LLM structurer ------------------------------------------------------


def test_llm_structurer_parses_a_well_formed_reply():
    payload = {
        "situation": "Bed 4, chest pain",
        "background": "Hypertension",
        "assessment": "Obs stable",
        "recommendation": "Repeat ECG",
    }
    structurer = LLMStructurer(lambda _: json.dumps(payload))
    report = structurer.structure(HANDOVER, source_id="h1")
    assert report.is_complete
    assert report.source_id == "h1"


def test_llm_structurer_degrades_to_an_empty_report_on_garbage():
    # One unparseable generation must show up as a completeness failure in the
    # results, not abort a hundred-transcript evaluation run.
    report = LLMStructurer(lambda _: "I cannot help with that.").structure(HANDOVER)
    assert not report.populated()


def test_llm_structurer_sends_the_rendered_prompt():
    seen = {}

    def generate(prompt: str) -> str:
        seen["prompt"] = prompt
        return "{}"

    LLMStructurer(generate, BASELINE).structure("the transcript body")
    assert "the transcript body" in seen["prompt"]
    assert "SBAR" in seen["prompt"]


def test_llm_structurer_name_records_the_prompt_version():
    assert LLMStructurer(lambda _: "{}", GROUNDED).name == "llm[grounded@3]"


# -- prompts -------------------------------------------------------------


def test_every_prompt_has_a_transcript_slot_and_a_rationale():
    for prompt in ALL_PROMPTS:
        assert "{transcript}" in prompt.template
        assert prompt.rationale.strip()
        assert "SBAR" in prompt.render("x") or "sbar" in prompt.render("x").lower()


def test_prompt_labels_are_unique():
    labels = [p.label for p in ALL_PROMPTS]
    assert len(set(labels)) == len(labels)


def test_prompt_lookup_by_label():
    assert get("grounded@3") is GROUNDED


def test_unknown_prompt_lists_the_alternatives():
    try:
        get("nope@9")
    except KeyError as exc:
        assert "available" in str(exc)
    else:
        raise AssertionError("expected KeyError")

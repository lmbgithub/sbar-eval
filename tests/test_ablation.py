import json

from sbar_eval.ablation import render, run_ablation
from sbar_eval.prompts import ALWAYS_FOUR, BASELINE, GROUNDED

TRANSCRIPTS = [(f"h{i}", "bed four chest pain, please repeat the ecg") for i in range(6)]


def make_generator(behaviour: dict[str, dict]):
    """Return a generator whose output depends on which prompt it receives."""

    def generate(prompt: str) -> str:
        for marker, payload in behaviour.items():
            if marker in prompt:
                return json.dumps(payload)
        return "{}"

    return generate


FULL = {
    "situation": "Bed 4, chest pain",
    "background": "Hypertension",
    "assessment": "Obs stable",
    "recommendation": "Repeat ECG",
}
NO_REC = {k: v for k, v in FULL.items() if k != "recommendation"}


def test_ablation_scores_every_variant():
    generate = make_generator({"Convert the following": FULL})
    ablation = run_ablation(generate, [BASELINE, ALWAYS_FOUR], TRANSCRIPTS)
    assert len(ablation.variants) == 2
    assert {v.label for v in ablation.variants} == {"baseline@1", "always_four@2"}


def test_winner_is_the_variant_with_higher_strict_completeness():
    generate = make_generator({
        "Convert the following": NO_REC,          # baseline drops Recommendation
        "All four keys must be present": FULL,    # the stricter prompts do not
    })
    ablation = run_ablation(generate, [BASELINE, ALWAYS_FOUR], TRANSCRIPTS)
    assert ablation.winner.label == "always_four@2"

    by_label = {v.label: v for v in ablation.variants}
    assert by_label["baseline@1"].score.strict_completeness == 0.0
    assert by_label["baseline@1"].score.mean_coverage == 0.75
    assert by_label["always_four@2"].score.strict_completeness == 1.0


def test_ties_break_toward_the_earlier_variant():
    # A later prompt must beat its predecessor, not merely match it.
    generate = make_generator({"": FULL})
    ablation = run_ablation(generate, [BASELINE, ALWAYS_FOUR, GROUNDED], TRANSCRIPTS)
    assert ablation.winner.label == "baseline@1"


def test_missing_section_counts_localize_the_defect():
    generate = make_generator({"Convert the following": NO_REC})
    ablation = run_ablation(generate, [BASELINE], TRANSCRIPTS)
    payload = ablation.to_dict()["variants"][0]
    assert payload["missing_by_section"]["recommendation"] == len(TRANSCRIPTS)
    assert payload["missing_by_section"]["situation"] == 0


def test_empty_ablation_has_no_winner():
    assert run_ablation(lambda _: "{}", [], TRANSCRIPTS).winner is None
    assert render(run_ablation(lambda _: "{}", [], TRANSCRIPTS)) == "no variants evaluated"


def test_render_includes_labels_and_the_winner():
    generate = make_generator({"All four keys must be present": FULL})
    out = render(run_ablation(generate, [BASELINE, ALWAYS_FOUR], TRANSCRIPTS))
    assert "baseline@1" in out
    assert "always_four@2" in out
    assert "winner:" in out

"""Versioned SBAR prompts.

Prompts are treated as versioned artifacts, not string literals edited in place.
A prompt change is a behaviour change: without a version and a recorded score,
there is no way to say whether the last edit helped, and no way to roll back to
the variant that worked.

The three variants below encode the progression that matters when structuring
spontaneous clinical speech, and `ablation.py` exists to measure the difference
rather than assume it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    """A named, versioned prompt template with one `{transcript}` slot."""

    name: str
    version: str
    rationale: str
    template: str

    def render(self, transcript: str) -> str:
        return self.template.format(transcript=transcript.strip())

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}"


BASELINE = Prompt(
    name="baseline",
    version="1",
    rationale=(
        "Minimal instruction. Establishes the floor: how much structure the model "
        "produces when simply asked for SBAR."
    ),
    template=(
        "Convert the following nursing shift handover into an SBAR report.\n\n"
        "Transcript:\n{transcript}\n\n"
        "Return JSON with keys: situation, background, assessment, recommendation."
    ),
)

ALWAYS_FOUR = Prompt(
    name="always_four",
    version="2",
    rationale=(
        "Spontaneous speech rarely follows the canonical S-B-A-R order, and a model "
        "that only extracts what is stated verbatim leaves sections empty. This "
        "variant requires all four keys and permits inference from context, which is "
        "the change that moves completeness."
    ),
    template=(
        "You are structuring a nursing shift handover into an SBAR report.\n\n"
        "Transcript:\n{transcript}\n\n"
        "Rules:\n"
        "- Return JSON with exactly these keys: situation, background, assessment, "
        "recommendation.\n"
        "- All four keys must be present.\n"
        "- Spoken handovers rarely follow SBAR order. Reorganise the content.\n"
        "- If a section is not stated literally, infer it from the clinical context "
        "of the transcript.\n"
        "- Use only information contained in the transcript."
    ),
)

GROUNDED = Prompt(
    name="grounded",
    version="3",
    rationale=(
        "Adds an explicit absence marker and a no-invention rule. Inference that is "
        "allowed to run unchecked will fabricate a Recommendation to satisfy the "
        "'all four keys' requirement, which trades a completeness failure for a "
        "safety failure. An explicit marker keeps the gap visible."
    ),
    template=(
        "You are structuring a nursing shift handover into an SBAR report.\n\n"
        "Transcript:\n{transcript}\n\n"
        "Rules:\n"
        "- Return JSON with exactly these keys: situation, background, assessment, "
        "recommendation.\n"
        "- All four keys must be present.\n"
        "- Spoken handovers rarely follow SBAR order. Reorganise the content.\n"
        "- If a section is implied but not stated literally, infer it from the "
        "clinical context.\n"
        "- Never invent clinical facts. Do not add vital signs, drug names, doses or "
        "diagnoses that are absent from the transcript.\n"
        "- If a section genuinely cannot be derived, set it to the string "
        '"[absent]" rather than guessing.'
    ),
)

REGISTRY: dict[str, Prompt] = {p.label: p for p in (BASELINE, ALWAYS_FOUR, GROUNDED)}
ALL: tuple[Prompt, ...] = (BASELINE, ALWAYS_FOUR, GROUNDED)


def get(label: str) -> Prompt:
    """Look up a prompt by `name@version`."""
    try:
        return REGISTRY[label]
    except KeyError:
        raise KeyError(
            f"unknown prompt {label!r}; available: {', '.join(sorted(REGISTRY))}"
        ) from None

"""Turning a transcript into an SBAR report.

Two implementations behind one interface:

`LLMStructurer` wraps any callable that takes a prompt and returns text. That is
the production path, and it is deliberately not tied to a vendor SDK — a local
llama.cpp server, an Ollama endpoint and a hosted API all satisfy the same
signature.

`CueStructurer` needs no model at all. It exists so the package is runnable and
testable the moment it is cloned, and so the evaluation harness has a stable
reference point: if a prompt change cannot beat a keyword baseline, the change
is not doing what it claims.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Protocol, Sequence

from sbar_eval.prompts import GROUNDED, Prompt
from sbar_eval.schema import SBARReport, Section

TextGenerator = Callable[[str], str]


class Structurer(Protocol):
    """Anything that turns a transcript into an SBAR report."""

    name: str

    def structure(self, transcript: str, source_id: str = "") -> SBARReport:
        ...


# Cue phrases per section. Ordered by specificity: a sentence is assigned to the
# section whose cues it matches most strongly, so "I would recommend chasing the
# bloods" lands in Recommendation rather than Assessment.
_CUES: dict[Section, tuple[str, ...]] = {
    Section.SITUATION: (
        "this is", "presenting with", "came in", "admitted", "bed ", "patient is",
        "currently", "right now", "complaining of", "arrived", "years old",
        "in for", "brought in",
    ),
    Section.BACKGROUND: (
        "history of", "known", "background", "past medical", "previously",
        "usually takes", "allergic", "allergy", "diagnosed", "chronic",
        "has had", "on regular", "chart shows", "prior",
    ),
    Section.ASSESSMENT: (
        "obs are", "vitals", "saturation", "sats", "blood pressure", "heart rate",
        "temperature", "looks", "appears", "i think", "seems", "stable",
        "deteriorat", "improving", "concerned", "gcs", "pain score", "afebrile",
    ),
    Section.RECOMMENDATION: (
        "needs", "please", "should", "recommend", "chase", "review", "monitor",
        "keep an eye", "escalate", "follow up", "repeat", "continue", "make sure",
        "hourly", "if ", "call ", "next shift", "handover to",
    ),
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    parts = (p.strip() for p in _SENTENCE_SPLIT.split(text or ""))
    return [p for p in parts if p]


def _score_sentence(sentence: str) -> dict[Section, int]:
    lowered = sentence.lower()
    return {
        section: sum(len(cue) for cue in cues if cue in lowered)
        for section, cues in _CUES.items()
    }


class CueStructurer:
    """Deterministic keyword baseline. No model, no network, no credentials.

    Assigns each sentence to its best-matching section. Sentences that match
    nothing fall through to the section carried forward from the previous
    sentence, which mirrors how handover speech actually flows: a speaker stays
    on a topic for several sentences before moving on.
    """

    name = "cue-baseline"

    def __init__(self, default_section: Section = Section.SITUATION) -> None:
        self.default_section = default_section

    def structure(self, transcript: str, source_id: str = "") -> SBARReport:
        buckets: dict[Section, list[str]] = {s: [] for s in Section.ordered()}
        current = self.default_section

        for sentence in _split_sentences(transcript):
            scores = _score_sentence(sentence)
            best = max(scores, key=lambda s: scores[s])
            if scores[best] > 0:
                current = best
            buckets[current].append(sentence)

        return SBARReport(
            situation=" ".join(buckets[Section.SITUATION]) or None,
            background=" ".join(buckets[Section.BACKGROUND]) or None,
            assessment=" ".join(buckets[Section.ASSESSMENT]) or None,
            recommendation=" ".join(buckets[Section.RECOMMENDATION]) or None,
            source_id=source_id,
        )


class LLMStructurer:
    """Prompt a text generator and parse its JSON reply."""

    def __init__(
        self,
        generate: TextGenerator,
        prompt: Prompt = GROUNDED,
        *,
        name: str | None = None,
    ) -> None:
        self.generate = generate
        self.prompt = prompt
        self.name = name or f"llm[{prompt.label}]"

    def structure(self, transcript: str, source_id: str = "") -> SBARReport:
        raw = self.generate(self.prompt.render(transcript))
        payload = extract_json(raw)
        if payload is None:
            # An unparseable reply is an empty report, not an exception. One bad
            # generation in a hundred should show up as a completeness failure
            # in the results, not abort the evaluation run.
            return SBARReport(source_id=source_id)
        report = SBARReport.from_dict(payload)
        return SBARReport(
            situation=report.situation,
            background=report.background,
            assessment=report.assessment,
            recommendation=report.recommendation,
            source_id=source_id,
        )


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply.

    Models wrap JSON in prose, fences, or both, regardless of instructions.
    Scanning for the first balanced object is more reliable than requiring the
    whole reply to parse, and far more reliable than a regex.
    """
    if not text:
        return None

    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(stripped[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = stripped.find("{", start + 1)
    return None


def structure_all(
    structurer: Structurer, transcripts: Sequence[tuple[str, str]]
) -> list[SBARReport]:
    """Structure `(source_id, transcript)` pairs."""
    return [structurer.structure(text, source_id) for source_id, text in transcripts]

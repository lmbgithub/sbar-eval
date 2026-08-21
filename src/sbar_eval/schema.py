"""The SBAR report model.

SBAR (Situation, Background, Assessment, Recommendation) is the WHO-recommended
structure for clinical handover. The four sections are fixed and ordered: the
whole point of the standard is that a receiving clinician always knows where to
look, so this is a closed enum, never a free-form dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator


class Section(str, Enum):
    """The four SBAR sections, in canonical order."""

    SITUATION = "situation"
    BACKGROUND = "background"
    ASSESSMENT = "assessment"
    RECOMMENDATION = "recommendation"

    @property
    def initial(self) -> str:
        return self.value[0].upper()

    @classmethod
    def ordered(cls) -> tuple["Section", ...]:
        return (cls.SITUATION, cls.BACKGROUND, cls.ASSESSMENT, cls.RECOMMENDATION)


# A section that exists as a key but holds a refusal, a placeholder, or an
# apology is not a recovered section. Scoring these as present is the most
# common way an SBAR completeness metric silently inflates itself.
_PLACEHOLDERS = (
    "n/a",
    "na",
    "none",
    "not mentioned",
    "not stated",
    "not provided",
    "not available",
    "not specified",
    "unknown",
    "no information",
    "not applicable",
    "-",
    "--",
    "...",
    "[absent]",
    "absent",
    "null",
)

MIN_CONTENT_CHARS = 3


def is_populated(value: str | None) -> bool:
    """True when a section holds real clinical content.

    Whitespace, placeholders and explicit absence markers all count as absent.
    """
    if value is None:
        return False
    cleaned = " ".join(str(value).split())
    if not cleaned:
        return False
    # Strip surrounding punctuation before comparing, so "N/A.", "[absent]" and
    # a bare "..." all reduce to the same absence signal. A value that is
    # *entirely* punctuation reduces to nothing and is absent by definition.
    core = cleaned.strip(" .-_[]()").lower()
    if len(core) < MIN_CONTENT_CHARS:
        return False
    return core not in _PLACEHOLDERS


@dataclass(frozen=True)
class SBARReport:
    """One structured handover."""

    situation: str | None = None
    background: str | None = None
    assessment: str | None = None
    recommendation: str | None = None
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, section: Section) -> str | None:
        return getattr(self, section.value)

    def populated(self) -> tuple[Section, ...]:
        """Sections that hold real content, in canonical order."""
        return tuple(s for s in Section.ordered() if is_populated(self.get(s)))

    def missing(self) -> tuple[Section, ...]:
        present = set(self.populated())
        return tuple(s for s in Section.ordered() if s not in present)

    @property
    def is_complete(self) -> bool:
        return not self.missing()

    def __iter__(self) -> Iterator[tuple[Section, str | None]]:
        for section in Section.ordered():
            yield section, self.get(section)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {s.value: self.get(s) for s in Section.ordered()}
        if self.source_id:
            data["source_id"] = self.source_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SBARReport":
        """Build a report from a dict, tolerating single-letter and abbreviated keys.

        An LLM asked for SBAR will return `situation`, `Situation`, or `S`
        depending on the prompt and the model. Normalizing here keeps that
        variance out of the scoring code.
        """
        aliases = {}
        for section in Section.ordered():
            aliases[section.value] = section
            aliases[section.initial.lower()] = section
            aliases[section.value[:4]] = section

        resolved: dict[str, str | None] = {}
        for raw_key, value in data.items():
            key = str(raw_key).strip().lower()
            section = aliases.get(key)
            if section is not None and section.value not in resolved:
                resolved[section.value] = None if value is None else str(value)

        return cls(
            situation=resolved.get("situation"),
            background=resolved.get("background"),
            assessment=resolved.get("assessment"),
            recommendation=resolved.get("recommendation"),
            source_id=str(data.get("source_id", "")),
        )

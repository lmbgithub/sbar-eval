"""Prompt ablation.

Runs several prompt variants over the same corpus and reports what each one
actually did. This is the whole argument for versioning prompts: "v3 feels
better" is not a result, and a completeness number attached to a prompt label is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from sbar_eval.completeness import CorpusScore, score_corpus
from sbar_eval.prompts import Prompt
from sbar_eval.schema import Section
from sbar_eval.structurer import LLMStructurer, TextGenerator, structure_all


@dataclass(frozen=True)
class Variant:
    """One prompt's result over the corpus."""

    label: str
    rationale: str
    score: CorpusScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.label,
            "rationale": self.rationale,
            "mean_coverage": round(self.score.mean_coverage, 4),
            "strict_completeness": round(self.score.strict_completeness, 4),
            "missing_by_section": {
                s.value: n for s, n in self.score.missing_by_section().items()
            },
        }


@dataclass(frozen=True)
class Ablation:
    """The full comparison."""

    variants: tuple[Variant, ...]

    @property
    def winner(self) -> Variant | None:
        """Best variant by strict completeness, then mean coverage.

        Ties break toward the earlier variant, so a later prompt has to actually
        beat its predecessor to replace it rather than merely match it.
        """
        if not self.variants:
            return None
        return max(
            self.variants,
            key=lambda v: (v.score.strict_completeness, v.score.mean_coverage),
        )

    def to_dict(self) -> dict[str, Any]:
        winner = self.winner
        return {
            "variants": [v.to_dict() for v in self.variants],
            "winner": winner.label if winner else None,
        }


def run_ablation(
    generate: TextGenerator,
    prompts: Sequence[Prompt],
    transcripts: Sequence[tuple[str, str]],
) -> Ablation:
    """Score each prompt over the same transcripts."""
    variants = []
    for prompt in prompts:
        reports = structure_all(LLMStructurer(generate, prompt), transcripts)
        variants.append(
            Variant(label=prompt.label, rationale=prompt.rationale, score=score_corpus(reports))
        )
    return Ablation(variants=tuple(variants))


def render(ablation: Ablation) -> str:
    """Render the comparison as a plain-text table."""
    if not ablation.variants:
        return "no variants evaluated"

    lines = ["=" * 74, "SBAR PROMPT ABLATION", "=" * 74]
    header = f"{'prompt':<20}{'coverage':>10}{'strict':>10}   missing sections"
    lines.append(header)
    lines.append("-" * 74)

    for variant in ablation.variants:
        missing = variant.score.missing_by_section()
        worst = ", ".join(
            f"{s.initial}:{n}" for s, n in missing.items() if n
        ) or "none"
        lines.append(
            f"{variant.label:<20}"
            f"{variant.score.mean_coverage:>9.1%}"
            f"{variant.score.strict_completeness:>10.1%}"
            f"   {worst}"
        )

    winner = ablation.winner
    if winner is not None:
        lines.append("-" * 74)
        lines.append(f"winner: {winner.label}")
        lines.append(f"why:    {winner.rationale}")
    return "\n".join(lines)

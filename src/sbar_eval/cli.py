"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from sbar_eval import __version__
from sbar_eval.completeness import score_corpus
from sbar_eval.prompts import ALL as ALL_PROMPTS
from sbar_eval.schema import SBARReport, Section
from sbar_eval.structurer import CueStructurer, structure_all

EXIT_OK = 0
EXIT_BELOW_THRESHOLD = 1
EXIT_USAGE = 2


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")
    rows = []
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{line_no}: invalid JSON ({exc.msg})") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{p}:{line_no}: expected a JSON object")
        rows.append(payload)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sbar-eval",
        description="Structure handover transcripts into SBAR and score completeness.",
    )
    parser.add_argument("--version", action="version", version=f"sbar-eval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    st = sub.add_parser("structure", help="structure transcripts with the cue baseline")
    st.add_argument("transcripts", help="JSONL with 'id' and 'transcript' per line")
    st.add_argument("--json", dest="json_out", help="write SBAR reports to this path")

    sc = sub.add_parser("score", help="score existing SBAR reports for completeness")
    sc.add_argument("reports", help="JSONL with situation/background/assessment/recommendation")
    sc.add_argument("--threshold", type=float, default=0.8, help="strict completeness gate")
    sc.add_argument("--soft", action="store_true", help="gate on mean coverage instead")
    sc.add_argument("--json", dest="json_out", help="write the score report to this path")

    sub.add_parser("prompts", help="list the versioned SBAR prompts")
    return parser


def _cmd_structure(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.transcripts)
    pairs = [(str(r.get("id", f"utt-{i:04d}")), str(r.get("transcript", ""))) for i, r in enumerate(rows)]
    reports = structure_all(CueStructurer(), pairs)
    score = score_corpus(reports)

    print(f"structured {len(reports)} transcript(s) with the cue baseline")
    print(f"mean coverage        {score.mean_coverage:.1%}")
    print(f"strict completeness  {score.strict_completeness:.1%}")

    if args.json_out:
        Path(args.json_out).write_text(
            "\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in reports) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")
    return EXIT_OK


def _cmd_score(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.reports)
    reports = []
    for i, row in enumerate(rows):
        row.setdefault("source_id", row.get("id", f"report-{i:04d}"))
        reports.append(SBARReport.from_dict(row))

    score = score_corpus(reports)
    print("=" * 58)
    print("SBAR COMPLETENESS")
    print("=" * 58)
    print(f"reports              {score.count}")
    print(f"mean coverage        {score.mean_coverage:.1%}")
    print(f"strict completeness  {score.strict_completeness:.1%}")
    print("")
    print("missing by section:")
    for section, count in score.missing_by_section().items():
        share = count / score.count if score.count else 0.0
        print(f"  {section.value:<16} {count:>4}  ({share:.0%})")

    incomplete = [r for r in score.reports if not r.is_complete]
    if incomplete:
        print("")
        print(f"--- {min(5, len(incomplete))} of {len(incomplete)} incomplete report(s) ---")
        for report in incomplete[:5]:
            missing = ", ".join(s.value for s in report.missing)
            print(f"  [{report.source_id}] missing: {missing}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(score.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")

    passed = score.meets(args.threshold, strict=not args.soft)
    metric = "mean coverage" if args.soft else "strict completeness"
    value = score.mean_coverage if args.soft else score.strict_completeness
    print("")
    print(f"gate: {metric} {value:.1%} vs threshold {args.threshold:.1%} -> {'PASS' if passed else 'FAIL'}")
    return EXIT_OK if passed else EXIT_BELOW_THRESHOLD


def _cmd_prompts(_: argparse.Namespace) -> int:
    for prompt in ALL_PROMPTS:
        print(f"{prompt.label}\n  {prompt.rationale}\n")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "structure":
            return _cmd_structure(args)
        if args.command == "score":
            return _cmd_score(args)
        if args.command == "prompts":
            return _cmd_prompts(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    parser.error(f"unknown command: {args.command}")
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())

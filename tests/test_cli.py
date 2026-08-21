import json
from pathlib import Path

import pytest

from sbar_eval.cli import EXIT_BELOW_THRESHOLD, EXIT_OK, EXIT_USAGE, main

HANDOVER = (
    "This is bed four, chest pain since six. History of hypertension. "
    "Obs are stable. Please repeat the ECG."
)


def write(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_structure_command(tmp_path, capsys):
    src = write(tmp_path / "t.jsonl", [{"id": "h1", "transcript": HANDOVER}])
    out = tmp_path / "reports.jsonl"
    assert main(["structure", str(src), "--json", str(out)]) == EXIT_OK
    assert "structured 1 transcript" in capsys.readouterr().out
    written = json.loads(out.read_text().strip())
    assert written["source_id"] == "h1"


def test_score_command_passes_above_threshold(tmp_path, capsys):
    rows = [{
        "id": f"r{i}",
        "situation": "Bed 4, chest pain",
        "background": "Hypertension",
        "assessment": "Obs stable",
        "recommendation": "Repeat ECG",
    } for i in range(4)]
    src = write(tmp_path / "r.jsonl", rows)
    assert main(["score", str(src), "--threshold", "0.8"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "strict completeness  100.0%" in out
    assert "PASS" in out


def test_score_command_fails_below_threshold(tmp_path, capsys):
    rows = [{
        "id": f"r{i}",
        "situation": "Bed 4",
        "background": "Hypertension",
        "assessment": "Obs stable",
        "recommendation": "N/A",
    } for i in range(4)]
    src = write(tmp_path / "r.jsonl", rows)
    assert main(["score", str(src)]) == EXIT_BELOW_THRESHOLD
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "incomplete report" in out


def test_soft_gate_uses_mean_coverage(tmp_path):
    rows = [{
        "id": f"r{i}",
        "situation": "Bed 4",
        "background": "Hypertension",
        "assessment": "Obs stable",
        "recommendation": None,
    } for i in range(4)]
    src = write(tmp_path / "r.jsonl", rows)
    assert main(["score", str(src), "--threshold", "0.7", "--soft"]) == EXIT_OK
    assert main(["score", str(src), "--threshold", "0.7"]) == EXIT_BELOW_THRESHOLD


def test_score_writes_json(tmp_path):
    src = write(tmp_path / "r.jsonl", [{"id": "r1", "situation": "Bed 4, chest pain"}])
    out = tmp_path / "score.json"
    main(["score", str(src), "--json", str(out)])
    payload = json.loads(out.read_text())
    assert payload["reports"] == 1
    assert payload["missing_by_section"]["recommendation"] == 1


def test_prompts_command_lists_versions(capsys):
    assert main(["prompts"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "baseline@1" in out and "grounded@3" in out


def test_missing_file_is_a_usage_error(tmp_path, capsys):
    assert main(["score", str(tmp_path / "nope.jsonl")]) == EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_malformed_jsonl_is_reported_with_a_line_number(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a"}\nnot json\n', encoding="utf-8")
    assert main(["score", str(bad)]) == EXIT_USAGE
    assert ":2:" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0

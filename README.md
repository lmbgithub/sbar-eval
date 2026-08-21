# sbar-eval

Structure clinical handover transcripts into SBAR and score their completeness.

SBAR (Situation, Background, Assessment, Recommendation) is the WHO-recommended
structure for clinical handover. This package turns a free-speech handover
transcript into the four sections and measures how much of the structure
actually survived — with a metric designed so that a model reliably dropping one
section cannot hide behind a good average.

Pure standard library. No model, no network, no credentials required to run it.

```
$ sbar-eval score examples/reports.jsonl
==========================================================
SBAR COMPLETENESS
==========================================================
reports              6
mean coverage        91.7%
strict completeness  66.7%

missing by section:
  situation           0  (0%)
  background          0  (0%)
  assessment          0  (0%)
  recommendation      2  (33%)

gate: strict completeness 66.7% vs threshold 80.0% -> FAIL
```

That output is the argument for the whole package. Mean coverage of 91.7% reads
like a working system. Strict completeness of 66.7% says a third of these
handovers reach the receiving nurse with no Recommendation — and the
per-section breakdown says exactly which part of the prompt to go fix.

## Background

This implements the SBAR structuring and completeness-scoring approach from
Module 3 of **UrgeNurse Agent**, a multi-agent, locally-hosted platform for
nursing documentation support in emergency departments, submitted as a Master's
thesis (Máster Universitario en Inteligencia Artificial, UNIR, 2026).

The thesis is co-authored work by a team of three. Module 3 — ASR transcription
and SBAR structuring — was my individual contribution: handover dataset
analysis, Faster-Whisper + Silero VAD implementation, iterative SBAR prompt
design, and the WER and SBAR-completeness evaluation.

This repository is a clean, standalone reimplementation of the _approach_ — the
metric definition, the prompt-versioning discipline, and the evaluation harness.
It is not the thesis codebase and it does not reproduce the thesis results.
Figures quoted from the thesis below are labelled as such.

## Why two completeness numbers

The thesis defines SBAR completeness as recovered sections over four. Reported
alone and averaged across a corpus, that number is misleading, so this package
reports two:

| Metric                | Definition                                  | What it is for                               |
| --------------------- | ------------------------------------------- | -------------------------------------------- |
| `mean_coverage`       | mean of (recovered sections / 4)            | The smooth signal you tune a prompt against  |
| `strict_completeness` | share of reports with **all four** sections | The number that reflects clinical usefulness |

A handover missing its Recommendation is not "75% good" — it is a handover the
receiving nurse cannot act on. The thesis makes the same point: a handover from
which a section cannot be derived counts as an incomplete output, not a partial
hit, which is what makes the metric sensitive to precisely the omissions SBAR
exists to prevent.

**Placeholders do not count as content.** A model asked for four keys will
happily return four keys, three of them saying `"not mentioned"`. `is_populated`
rejects empty strings, whitespace, and absence markers (`N/A`, `none`,
`[absent]`, `unknown`, …) before scoring, because counting those as present is
the most common way an SBAR completeness metric silently inflates itself.

## Install

```bash
git clone https://github.com/<your-username>/sbar-eval.git
cd sbar-eval
pip install -e ".[dev]"
```

Python 3.10+.

## Usage

### Score existing SBAR reports

JSONL, one report per line. Keys may be full names, single letters, or
abbreviations — `situation`, `Situation`, and `S` all resolve:

```jsonl
{
    "id": "r-001",
    "situation": "Bed 4, chest pain since 06:00",
    "background": "Hypertension",
    "assessment": "BP 130/80",
    "recommendation": "Repeat ECG in one hour"
}
```

```bash
sbar-eval score reports.jsonl --threshold 0.8    # exits 1 below threshold
sbar-eval score reports.jsonl --soft             # gate on mean coverage instead
sbar-eval score reports.jsonl --json score.json
```

### Structure transcripts without a model

```bash
sbar-eval structure examples/handovers.jsonl --json structured.jsonl
```

Uses `CueStructurer`, a deterministic cue-phrase baseline. It exists so the
package runs on clone, and so the ablation harness has a stable reference point:
**if a prompt change cannot beat a keyword baseline, the change is not doing
what it claims.**

### Structure with a local LLM

`LLMStructurer` wraps any callable of `str -> str`, so a llama.cpp server, an
Ollama endpoint, and a hosted API all satisfy the same interface:

```python
from sbar_eval import LLMStructurer, score_corpus
from sbar_eval.prompts import GROUNDED

def generate(prompt: str) -> str:
    # any local or hosted model
    return my_llm(prompt)

structurer = LLMStructurer(generate, GROUNDED)
reports = [structurer.structure(t, source_id=i) for i, t in transcripts]
print(score_corpus(reports).to_dict())
```

Model replies are parsed with a brace-matching JSON extractor that tolerates
prose, code fences, nested objects, and braces inside strings. An unparseable
reply becomes an empty report, not an exception: one bad generation in a hundred
should show up as a completeness failure in the results, not abort the run.

## Prompt versioning and ablation

Prompts are versioned artifacts, not string literals edited in place. A prompt
change is a behaviour change, and without a version and a recorded score there
is no way to say whether the last edit helped.

```bash
$ sbar-eval prompts
baseline@1
  Minimal instruction. Establishes the floor.

always_four@2
  Spontaneous speech rarely follows canonical S-B-A-R order... requires all four
  keys and permits inference from context, which is the change that moves
  completeness.

grounded@3
  Adds an explicit absence marker and a no-invention rule. Inference allowed to
  run unchecked will fabricate a Recommendation to satisfy the 'all four keys'
  requirement, trading a completeness failure for a safety failure.
```

That progression is the real lesson from the module. Requiring four sections
raises completeness; requiring them _without_ a no-invention rule raises
completeness by inventing clinical facts. The third variant exists because the
second one is dangerous on its own.

```python
from sbar_eval import run_ablation, render
from sbar_eval.prompts import ALL

print(render(run_ablation(generate, ALL, transcripts)))
```

```
==========================================================================
SBAR PROMPT ABLATION
==========================================================================
prompt                coverage    strict   missing sections
--------------------------------------------------------------------------
baseline@1               75.0%      0.0%   R:6
always_four@2           100.0%    100.0%   none
--------------------------------------------------------------------------
winner: always_four@2
```

Ties break toward the earlier variant, so a later prompt has to actually beat
its predecessor to replace it rather than merely match it.

## Thesis results (context, not a claim about this repo)

Measured in the thesis on 100 synthetic nursing handover recordings, CPU-only,
under a 2 GB RAM budget:

| Metric                    | Target      | Measured        |
| ------------------------- | ----------- | --------------- |
| WER (global)              | < 20%       | 14.2%           |
| WER (nursing terminology) | informative | 22.5%           |
| Real-time factor          | < 1.0       | 0.32            |
| SBAR completeness         | ≥ 80%       | above threshold |

Selected configuration: `faster-whisper base.en`, INT8 quantization via
CTranslate2, Silero VAD pre-filter, SBAR structuring by a locally-served
quantized LLM. These numbers come from that evaluation, on that dataset and
hardware — this repository does not reproduce them.

## Tests

```bash
pytest -q     # 75 tests
```

Covers the placeholder/absence rules, key-alias normalization, the
coverage-versus-strict distinction, per-section defect localization, JSON
extraction against adversarial model output (fences, nested braces, escaped
quotes, malformed leading objects), ablation tie-breaking, and CLI exit codes.

## Scope

This is a documentation-support and evaluation tool. It structures and measures
text that already exists. It does not diagnose, prescribe, or make clinical
decisions, and any output is intended for human review before use.

No real patient data is present in this repository. The example handovers are
synthetic.

## License

MIT — see [LICENSE](LICENSE).

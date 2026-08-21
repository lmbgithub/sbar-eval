"""Structure clinical handover transcripts into SBAR and score their completeness."""

from sbar_eval.ablation import Ablation, Variant, render, run_ablation
from sbar_eval.completeness import CorpusScore, ReportScore, score_corpus, score_report
from sbar_eval.prompts import ALL as PROMPTS
from sbar_eval.prompts import Prompt, get
from sbar_eval.schema import SBARReport, Section, is_populated
from sbar_eval.structurer import CueStructurer, LLMStructurer, extract_json, structure_all

__version__ = "0.1.0"

__all__ = [
    "Ablation", "Variant", "render", "run_ablation",
    "CorpusScore", "ReportScore", "score_corpus", "score_report",
    "PROMPTS", "Prompt", "get",
    "SBARReport", "Section", "is_populated",
    "CueStructurer", "LLMStructurer", "extract_json", "structure_all",
    "__version__",
]

# dashboard/reasoning_utils.py
from __future__ import annotations
import re
from dataclasses import dataclass

_STEP_PATTERN = re.compile(r"^Étape\s+(\d+)\s*[–-]\s*(.*)$")

# (icon, short label) per step number — matches the LangGraph node order
# confirmed in src/agent/nodes.py. Step 2 is included for robustness, even
# though it never reaches this tab in practice (see the guide's §2.3).
_STEP_META = {
    1: ("🔬", "Diagnostic"),
    2: ("⚠️", "Confidence check"),
    3: ("📚", "RAG retrieval"),
    4: ("🌦️", "Weather check"),
    5: ("✅", "Decision"),
    6: ("📝", "Synthesis"),
}
_DEFAULT_META = ("🔹", "Step")


@dataclass
class ReasoningStep:
    number: int
    icon: str
    label: str
    detail: str   # text after "Étape N – ", shown inside the expander body
    raw: str      # original line, untouched — kept in case it's needed elsewhere


def parse_trace(trace: list[str]) -> list[ReasoningStep]:
    """Turns the flat `trace` list into structured steps for the accordion.
    Falls back gracefully (sequential numbering, default icon, full line as
    detail) if a line doesn't match the expected "Étape N – ..." shape — so
    a future backend format change degrades instead of crashing the tab."""
    steps = []
    for i, line in enumerate(trace, start=1):
        match = _STEP_PATTERN.match(line.strip())
        if match:
            number = int(match.group(1))
            detail = match.group(2)
        else:
            number = i
            detail = line
        icon, label = _STEP_META.get(number, _DEFAULT_META)
        steps.append(ReasoningStep(number=number, icon=icon, label=label, detail=detail, raw=line))
    return steps
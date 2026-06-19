import re

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above) instructions",
    r"you are now",
    r"system prompt",
    r"disregard (the )?(rules|instructions)",
]

def is_suspicious(question: str) -> bool:
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)

def sanitize_question(question: str) -> str:
    cleaned = question.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)          # collapse whitespace
    cleaned = re.sub(r"[<>{}]", "", cleaned)         # strip characters with no place in a question
    return cleaned
"""Whitespace cleanup for analyst report text.

Some LLM providers (notably Google Gemini) occasionally pad a report's
tail with hundreds of thousands of blank lines. When the debate stage
concatenates all analyst reports into one prompt, that padding blows up
the input past the model's context window (observed: ARKB report tails
reaching ~99% whitespace, debate input hitting ~1.9M tokens vs Gemini's
1,048,576 limit).

The cleanup is intentionally conservative: it only strips line-trailing
spaces and collapses 3+ consecutive newlines down to a single blank
line, so ordinary Markdown paragraph breaks are preserved.
"""

import re

_TRAILING_WS = re.compile(r"[ \t]+(?=\n)")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def clean_report_whitespace(text: str) -> str:
    """Collapse runaway trailing / blank whitespace from an LLM report."""
    if not text:
        return text
    text = _TRAILING_WS.sub("", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()

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

Trailing spaces are stripped line-by-line with str.rstrip, NOT with a
regex: the previous r"[ \t]+(?=\n)" pattern backtracked quadratically on
a long space run not followed by a newline (Gemini's runaway padding can
end mid-line or at end-of-string), and a few hundred thousand chars was
enough to burn generate_report's entire 60-minute budget (daily picks
2026-07-16..18 all timed out inside this function).
"""

import re

_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def clean_report_whitespace(text: str) -> str:
    """Collapse runaway trailing / blank whitespace from an LLM report."""
    if not text:
        return text
    text = "\n".join(line.rstrip(" \t") for line in text.split("\n"))
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()

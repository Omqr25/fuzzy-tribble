"""Shared text-matching, Arabic-normalization, corpus-building, and index-validation utilities.

Used by both ``app/services/constraints.py`` and ``app/services/scorer.py`` so
hard eligibility checks and hybrid lexical scoring share identical Unicode/Arabic
normalization, boundary-aware phrase matching, candidate corpus segmentation, and
NumPy-safe index validation.
"""

from __future__ import annotations

from collections.abc import Sequence
import re
import unicodedata

import numpy as np

from app.schemas.candidate import CandidateForAllocation

_ARABIC_DIACRITICS_RE = re.compile(r"[\u0640\u064b-\u065f\u0670]")
_ARABIC_CHAR_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ة": "ه",
        "ى": "ي",
    }
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """Strip Arabic diacritics/tatweel and unify alef, teh marbuta, and alef maksura."""
    stripped = _ARABIC_DIACRITICS_RE.sub("", text)
    return stripped.translate(_ARABIC_CHAR_MAP)


def normalize_text(text: str | None) -> str:
    """Normalize text for deterministic Arabic/English comparison."""
    if not text:
        return ""
    composed = unicodedata.normalize("NFC", text)
    arabic_normalized = normalize_arabic(composed)
    collapsed = _WHITESPACE_RE.sub(" ", arabic_normalized).strip()
    return collapsed.casefold()


def dedupe_normalized_items(items: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Return ``(display_label, normalized_value)`` pairs, ignoring blanks."""
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in items:
        display = _WHITESPACE_RE.sub(" ", raw or "").strip()
        norm = normalize_text(display)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        pairs.append((display, norm))
    return tuple(pairs)


def compile_phrase_pattern(normalized_phrase: str) -> re.Pattern[str]:
    """Compile a boundary-aware pattern supporting English, Arabic, and tech symbols.

    Prevents partial token leaks (e.g. ``Java`` inside ``JavaScript``, ``C`` inside
    ``Computer``/``C++``/``Objective-C``, ``R`` inside ``Docker``/``R&D``,
    ``.NET`` inside ``ASP.NET``, ``React`` inside ``React-Native``) while allowing
    punctuation such as ``Python.`` and slash-separated tokens such as ``C/C++``.
    """
    escaped = re.escape(normalized_phrase)
    return re.compile(
        rf"(?<![\w+#.])(?<![\w+#][&-]){escaped}(?![\w+#]|[.&-][\w+#])"
    )


def contains_phrase(
    normalized_text: str,
    pattern: re.Pattern[str],
    normalized_phrase: str,
) -> bool:
    """Return True if ``normalized_phrase`` matches exactly or on token boundaries."""
    if not normalized_text or not normalized_phrase:
        return False
    if normalized_text == normalized_phrase:
        return True
    return pattern.search(normalized_text) is not None


def build_candidate_corpus_segments(
    candidate: CandidateForAllocation,
) -> tuple[str, ...]:
    """Extract, normalize, and deduplicate all 11 candidate text fields (excluding city)."""
    meta = candidate.metadata
    cv = candidate.parsed_cv
    raw_fields: list[str] = [
        *meta.technical_skills,
        meta.program,
        meta.field,
        meta.major,
        meta.additional_info,
        *cv.skills,
        cv.experience_summary,
        cv.education_summary,
        cv.projects_summary,
        *cv.certifications,
        *cv.languages,
    ]
    return tuple(norm for _, norm in dedupe_normalized_items(raw_fields))


def validate_index(index: int | np.integer, size: int, label: str) -> int:
    """Validate that ``index`` is an integer in ``[0, size)`` and return a Python ``int``."""
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise IndexError(f"{label} index must be an integer (got {index!r}).")
    idx = int(index)
    if idx < 0 or idx >= size:
        raise IndexError(f"{label} index {idx} out of bounds for size {size}.")
    return idx

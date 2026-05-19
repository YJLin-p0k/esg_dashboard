from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


def normalize_text(text: str) -> str:
    """Normalize PDF text while preserving sentence boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_chinese_sentences(text: str, min_length: int = 6) -> list[str]:
    """Split Chinese or mixed Chinese/English text into sentence-like chunks."""
    normalized = normalize_text(text)
    candidates = _SENTENCE_RE.findall(normalized)
    sentences: list[str] = []

    for candidate in candidates:
        sentence = _WHITESPACE_RE.sub(" ", candidate).strip()
        if len(sentence) >= min_length:
            sentences.append(sentence)

    return sentences

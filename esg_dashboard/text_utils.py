from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
_BLANK_LINE_RE = re.compile(r"\n\s*\n+")


def normalize_text(text: str) -> str:
    """Normalize PDF text while preserving sentence boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_chinese_paragraphs(text: str, min_length: int = 12, soft_max_length: int = 600) -> list[str]:
    """Split PDF text into paragraph-like chunks while keeping sentence order."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    blocks = [
        _WHITESPACE_RE.sub(" ", block).strip()
        for block in _BLANK_LINE_RE.split(text.strip())
        if block.strip()
    ]
    paragraphs = [block for block in blocks if len(block) >= min_length]
    if len(paragraphs) > 1:
        return paragraphs

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    fallback_paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        current.append(line)
        joined = _WHITESPACE_RE.sub(" ", " ".join(current).strip())
        if len(joined) >= soft_max_length and (
            re.search(r"[。！？!?；;]$", line) or len(joined) >= soft_max_length * 1.5
        ):
            fallback_paragraphs.append(joined)
            current = []

    if current:
        joined = _WHITESPACE_RE.sub(" ", " ".join(current).strip())
        if len(joined) >= min_length:
            fallback_paragraphs.append(joined)

    return fallback_paragraphs


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

from __future__ import annotations

import re
from dataclasses import dataclass


_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
_PARAGRAPH_END_RE = re.compile(r"[。！？!?；;]$")


@dataclass(frozen=True)
class SentenceUnit:
    sentence: str
    paragraph_id: int
    paragraph_text: str
    paragraph_context: str


def normalize_text(text: str) -> str:
    """Normalize PDF text while preserving sentence boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_chinese_sentences(text: str, min_length: int = 6) -> list[str]:
    """Split Chinese or mixed Chinese/English text into sentence-like chunks."""
    normalized = normalize_text(text)
    sentences: list[str] = []

    for candidate in _SENTENCE_RE.findall(normalized):
        sentence = _WHITESPACE_RE.sub(" ", candidate).strip()
        if len(sentence) >= min_length:
            sentences.append(sentence)

    return sentences


def split_chinese_sentence_units(
    text: str,
    min_length: int = 6,
    max_sentences_per_paragraph: int = 4,
    max_paragraph_chars: int = 520,
) -> list[SentenceUnit]:
    """Split text into sentence units while keeping paragraph-level context."""
    paragraphs = _split_paragraphs(text, max_sentences_per_paragraph, max_paragraph_chars)
    units: list[SentenceUnit] = []

    for paragraph_index, paragraph in enumerate(paragraphs, start=1):
        sentences = split_chinese_sentences(paragraph, min_length=min_length)
        if not sentences:
            continue

        context_parts = []
        if paragraph_index > 1:
            context_parts.append(paragraphs[paragraph_index - 2])
        context_parts.append(paragraph)
        if paragraph_index < len(paragraphs):
            context_parts.append(paragraphs[paragraph_index])
        paragraph_context = "\n".join(context_parts)

        for sentence in sentences:
            units.append(
                SentenceUnit(
                    sentence=sentence,
                    paragraph_id=paragraph_index,
                    paragraph_text=paragraph,
                    paragraph_context=paragraph_context,
                )
            )

    return units


def _split_paragraphs(
    text: str,
    max_sentences_per_paragraph: int,
    max_paragraph_chars: int,
) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]

    paragraphs: list[str] = []
    for block in raw_blocks:
        lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in block.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            continue

        current_lines: list[str] = []
        current_sentence_count = 0
        for line in lines:
            current_lines.append(line)
            current_sentence_count += len(_SENTENCE_RE.findall(line))
            current_text = _WHITESPACE_RE.sub(" ", " ".join(current_lines)).strip()
            should_flush = (
                len(current_text) >= max_paragraph_chars
                or current_sentence_count >= max_sentences_per_paragraph
                or (_PARAGRAPH_END_RE.search(line) and len(current_text) >= 120)
            )
            if should_flush:
                paragraphs.append(current_text)
                current_lines = []
                current_sentence_count = 0

        if current_lines:
            paragraphs.append(_WHITESPACE_RE.sub(" ", " ".join(current_lines)).strip())

    return paragraphs

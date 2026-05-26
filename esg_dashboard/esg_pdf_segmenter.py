from __future__ import annotations

# Install notes:
#   pip install pymupdf pandas
#
# Optional fallback if PyMuPDF is unavailable:
#   pip install pypdf

from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path
import argparse
import re
from typing import Any

import pandas as pd


DEFAULT_ESG_KEYWORDS = [
    "承諾",
    "目標",
    "預計",
    "將於",
    "計畫",
    "持續",
    "減碳",
    "淨零",
    "碳中和",
    "再生能源",
    "永續",
    "ESG",
    "治理",
    "社會責任",
    "查證",
    "驗證",
    "第三方",
    "認證",
    "達成",
    "完成",
    "揭露",
    "報告",
]

SENTENCE_END_RE = re.compile(r"([。！？；!?;])")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:第\s*)?(?:page\s*)?\d{1,4}(?:\s*/\s*\d{1,4}|\s+of\s+\d{1,4})?\s*(?:頁)?\s*$",
    re.IGNORECASE,
)
DOT_LEADER_RE = re.compile(r"\.{4,}|…{2,}|[．.]\s*[．.]\s*[．.]")
REFERENCE_HEADING_RE = re.compile(
    r"^\s*(參考文獻|參考資料|附錄|appendix|references?|bibliography)\s*$",
    re.IGNORECASE,
)
TOC_HEADING_RE = re.compile(r"^\s*(目錄|contents?|table of contents)\s*$", re.IGNORECASE)


def extract_pdf_text(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract readable text from each PDF page and keep page numbers.

    Returns:
        A list of {"page": int, "text": str}. Unreadable pages are skipped
        instead of stopping the whole job.
    """
    pdf_path = Path(pdf_path)
    pages: list[dict[str, Any]] = []

    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None

    if fitz is not None:
        try:
            with fitz.open(pdf_path) as doc:
                for page_index in range(doc.page_count):
                    try:
                        text = doc.load_page(page_index).get_text("text") or ""
                    except Exception as exc:
                        print(f"[warn] Failed to read page {page_index + 1}: {exc}")
                        continue
                    pages.append({"page": page_index + 1, "text": text})
            return pages
        except Exception as exc:
            print(f"[warn] PyMuPDF failed for {pdf_path.name}: {exc}")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Please install PyMuPDF first: pip install pymupdf pandas") from exc

    with pdf_path.open("rb") as file_obj:
        reader = PdfReader(file_obj)
        for page_index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                print(f"[warn] Failed to read page {page_index}: {exc}")
                continue
            pages.append({"page": page_index, "text": text})

    return pages


def clean_text(text: str) -> str:
    """Clean PDF text while keeping Chinese paragraphs reasonably intact."""
    if not text:
        return ""

    text = text.replace("\u3000", " ").replace("\xa0", " ")
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    kept_lines: list[str] = []

    for line in lines:
        if not line:
            kept_lines.append("")
            continue
        if PAGE_NUMBER_RE.match(line):
            continue
        if _looks_like_table_or_footer(line):
            continue
        kept_lines.append(line)

    paragraphs = _join_wrapped_lines(kept_lines)
    cleaned = "\n\n".join(paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    """Split text into Chinese-aware sentence units."""
    text = clean_text(text)
    if not text:
        return []

    sentences: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        parts = SENTENCE_END_RE.split(paragraph)
        buffer = ""
        for part in parts:
            if not part:
                continue
            buffer += part
            if SENTENCE_END_RE.fullmatch(part):
                _append_sentence(sentences, buffer)
                buffer = ""
        if buffer.strip():
            _append_sentence(sentences, buffer)

    return _merge_short_sentences(sentences)


def build_candidate_segments(
    sentences: Sequence[str],
    keywords: Sequence[str],
    window_size: int = 1,
) -> list[dict[str, Any]]:
    """Build ESG candidate paragraphs from matched sentences plus context."""
    normalized_keywords = [kw.strip() for kw in keywords if kw and kw.strip()]
    segments: list[dict[str, Any]] = []
    used_ranges: list[tuple[int, int]] = []

    for index, sentence in enumerate(sentences):
        matched = _matched_keywords(sentence, normalized_keywords)
        if not matched:
            continue

        start = max(0, index - window_size)
        end = min(len(sentences), index + window_size + 1)
        if used_ranges and start <= used_ranges[-1][1]:
            old_start, old_end = used_ranges.pop()
            start = min(start, old_start)
            end = max(end, old_end)
        used_ranges.append((start, end))

    for start, end in used_ranges:
        context_sentences = list(sentences[start:end])
        text = _fit_segment_length("".join(context_sentences), normalized_keywords)
        matched = _matched_keywords(text, normalized_keywords)
        if matched:
            segments.append(
                {
                    "text": text,
                    "matched_keywords": matched,
                    "char_count": len(text),
                }
            )

    return segments


def deduplicate_segments(
    segments: Sequence[dict[str, Any] | str],
    similarity_threshold: float = 0.92,
) -> list[dict[str, Any] | str]:
    """Remove exact and near-duplicate segments."""
    unique: list[dict[str, Any] | str] = []
    seen: set[str] = set()

    for segment in segments:
        text = segment["text"] if isinstance(segment, dict) else str(segment)
        normalized = _normalize_for_dedupe(text)
        if not normalized or normalized in seen:
            continue

        is_duplicate = False
        for existing in unique:
            existing_text = existing["text"] if isinstance(existing, dict) else str(existing)
            existing_normalized = _normalize_for_dedupe(existing_text)
            if _similarity(normalized, existing_normalized) >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            seen.add(normalized)
            unique.append(segment)

    return unique


def process_pdf(pdf_path: str | Path, keywords: Sequence[str] | None = None) -> pd.DataFrame:
    """Process a PDF into ESG candidate segments ready for NLP classification."""
    pdf_path = Path(pdf_path)
    keywords = list(keywords or DEFAULT_ESG_KEYWORDS)
    rows: list[dict[str, Any]] = []

    for page_record in extract_pdf_text(pdf_path):
        page = int(page_record["page"])
        raw_text = page_record.get("text", "")
        if _is_non_body_page(raw_text):
            continue

        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        for paragraph in re.split(r"\n{2,}", cleaned):
            sentences = split_sentences(paragraph)
            if not sentences:
                continue

            for segment in build_candidate_segments(sentences, keywords, window_size=1):
                rows.append(
                    {
                        "doc_name": pdf_path.name,
                        "page": page,
                        "paragraph_id": 0,
                        "text": segment["text"],
                        "matched_keywords": ", ".join(segment["matched_keywords"]),
                        "char_count": segment["char_count"],
                    }
                )

    rows = deduplicate_segments(rows)
    for paragraph_id, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            row["paragraph_id"] = paragraph_id

    return pd.DataFrame(
        rows,
        columns=["doc_name", "page", "paragraph_id", "text", "matched_keywords", "char_count"],
    )


def save_results(df: pd.DataFrame, output_csv_path: str | Path) -> None:
    """Save results as UTF-8 with BOM so Excel can open Chinese text correctly."""
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")


def _append_sentence(sentences: list[str], sentence: str) -> None:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if sentence:
        sentences.append(sentence)


def _join_wrapped_lines(lines: Sequence[str]) -> list[str]:
    paragraphs: list[str] = []
    buffer = ""

    for line in lines:
        if not line:
            if buffer.strip():
                paragraphs.append(buffer.strip())
                buffer = ""
            continue

        if not buffer:
            buffer = line
            continue

        if _should_join_without_space(buffer, line):
            buffer += line
        else:
            buffer += " " + line

    if buffer.strip():
        paragraphs.append(buffer.strip())

    return paragraphs


def _should_join_without_space(previous: str, current: str) -> bool:
    if re.search(r"[A-Za-z0-9)]$", previous) and re.search(r"^[A-Za-z0-9(]", current):
        return False
    if previous.endswith(("。", "！", "？", "；", ".", "!", "?", ";", "：", ":")):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]$", previous) or re.search(r"^[\u4e00-\u9fff]", current))


def _merge_short_sentences(sentences: Sequence[str], min_chars: int = 40) -> list[str]:
    if not sentences:
        return []

    merged: list[str] = []
    buffer = ""
    for sentence in sentences:
        if not buffer:
            buffer = sentence
            continue
        if len(buffer) < min_chars:
            buffer += sentence
        else:
            merged.append(buffer)
            buffer = sentence

    if buffer:
        if merged and len(buffer) < min_chars:
            merged[-1] += buffer
        else:
            merged.append(buffer)
    return merged


def _fit_segment_length(
    text: str,
    keywords: Sequence[str] | None = None,
    min_chars: int = 100,
    target_chars: int = 300,
    max_chars: int = 500,
) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text

    sentences = split_sentences(text)
    if not sentences:
        return _trim_around_keyword(text, keywords or [], max_chars)

    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(buffer) + len(sentence) <= target_chars or len(buffer) < min_chars:
            buffer += sentence
        else:
            chunks.append(buffer)
            buffer = sentence
    if buffer:
        chunks.append(buffer)

    keyword_chunks = [chunk for chunk in chunks if _matched_keywords(chunk, keywords or [])]
    best_pool = keyword_chunks or chunks
    best = max(best_pool, key=lambda item: min(len(item), max_chars))
    if len(best) > max_chars:
        return _trim_around_keyword(best, keywords or [], max_chars)
    return best[:max_chars].rstrip()


def _matched_keywords(text: str, keywords: Sequence[str]) -> list[str]:
    lowered = text.lower()
    matched: list[str] = []
    for keyword in keywords:
        if keyword.lower() in lowered:
            matched.append(keyword)
    return matched


def _trim_around_keyword(text: str, keywords: Sequence[str], max_chars: int) -> str:
    lowered = text.lower()
    hit_positions = [lowered.find(keyword.lower()) for keyword in keywords if keyword and keyword.lower() in lowered]
    if not hit_positions:
        return text[:max_chars].rstrip()

    center = min(position for position in hit_positions if position >= 0)
    start = max(0, center - max_chars // 2)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)

    trimmed = text[start:end].strip()
    trimmed = re.sub(r"^[^。！？；!?;]{0,40}[。！？；!?;]", "", trimmed).strip() or trimmed
    return trimmed.rstrip()


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if len(left) < 120 and len(right) < 120:
        return SequenceMatcher(None, left, right).ratio()
    left_set = set(_char_ngrams(left))
    right_set = set(_char_ngrams(right))
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    if len(text) <= n:
        return [text]
    return [text[index : index + n] for index in range(len(text) - n + 1)]


def _looks_like_table_or_footer(line: str) -> bool:
    if len(line) <= 2:
        return True
    if DOT_LEADER_RE.search(line):
        return True
    if re.match(r"^\s*(資料來源|來源|註|備註|copyright|版權)\s*[:：]", line, re.IGNORECASE):
        return True

    separators = sum(line.count(char) for char in "|│┆")
    digits = sum(char.isdigit() for char in line)
    if separators >= 3:
        return True
    if len(line) >= 12 and digits / max(len(line), 1) > 0.55:
        return True
    return False


def _is_non_body_page(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True

    first_lines = lines[:8]
    if any(TOC_HEADING_RE.match(line) or REFERENCE_HEADING_RE.match(line) for line in first_lines):
        return True

    dot_leader_count = sum(1 for line in lines if DOT_LEADER_RE.search(line))
    if len(lines) >= 5 and dot_leader_count / len(lines) > 0.35:
        return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ESG candidate paragraphs from a PDF.")
    parser.add_argument("pdf_path", help="Path to the PDF file.")
    parser.add_argument(
        "-o",
        "--output",
        default="esg_candidate_segments.csv",
        help="Output CSV path. Default: esg_candidate_segments.csv",
    )
    args = parser.parse_args()

    df = process_pdf(args.pdf_path)
    save_results(df, args.output)
    print(f"Saved {len(df)} ESG candidate segments to {args.output}")


if __name__ == "__main__":
    main()

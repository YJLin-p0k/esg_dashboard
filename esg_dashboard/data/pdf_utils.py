from __future__ import annotations

# Google Colab install notes:
#   !pip install -q pymupdf pandas

from collections import Counter
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
import re
from typing import Any, BinaryIO

import pandas as pd


SENTENCE_END = "。！？；"
SOFT_END_WORDS = ("，", "、", "及", "與", "的", "之", "為", "在", "將", "並", "或")
BAD_START_WORDS = ("的", "及", "與", "並", "或", "而", "且")
BAD_END_WORDS = ("將", "於", "在", "為", "之", "與", "及")

WHITESPACE_RE = re.compile(r"[ \t\r\f\v\u3000]+")
PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?\d{1,4}(?:\s*/\s*\d{1,4}|\s+of\s+\d{1,4})?\s*$",
    re.IGNORECASE,
)
TABLE_FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?:(?:\u5716|\u8868)\s*[A-Za-z]?\s*\d+(?:[-–.]\d+)*|(?:fig(?:ure)?|table)\s*[A-Za-z]?\s*\d+(?:[-–.]\d+)*)"
    r"\s*[:：.\-、]?\s*.{0,100}$",
    re.IGNORECASE,
)
BULLET_RE = re.compile(r"^\s*(?:[一二三四五六七八九十]+、|\d+[.)、]|[（(]\d+[）)]|[•●○\-])\s*")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
ABBR_RE = re.compile(r"\b(?:Inc|Ltd|Co|Corp|Dr|Mr|Ms|No|Fig|e\.g|i\.e)\.$", re.IGNORECASE)
CHINESE_PUNCT_RE = re.compile(r"[。！？；]")
NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|％|tCO2e|CO2e|kWh|MWh|GWh|噸|公噸|人|件|家|次|元|萬元|億元)?", re.IGNORECASE)
TABLE_UNIT_RE = re.compile(r"^(?:單位|unit)\s*[:：]|(?:仟元|千元|萬元|億元|噸|公噸|%|％|kWh|MWh|GWh|tCO2e|CO2e)$", re.IGNORECASE)
YEAR_HEADER_RE = re.compile(r"^(?:20\d{2}|19\d{2})(?:\s*[/-]\s*(?:20\d{2}|19\d{2}))*$")


def extract_pdf_blocks(pdf_source: str | Path | BinaryIO | BytesIO) -> list[dict[str, Any]]:
    """Extract text blocks with coordinates from a PDF using PyMuPDF dict data."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("Please install PyMuPDF first: pip install pymupdf") from exc

    doc_name = getattr(pdf_source, "name", None) or (Path(pdf_source).name if isinstance(pdf_source, (str, Path)) else "uploaded.pdf")
    if isinstance(pdf_source, (str, Path)):
        doc = fitz.open(str(pdf_source))
    else:
        pdf_source.seek(0)
        doc = fitz.open(stream=pdf_source.read(), filetype="pdf")

    blocks: list[dict[str, Any]] = []
    with doc:
        for page_index, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            page_height = float(page.rect.height)
            page_width = float(page.rect.width)
            text_blocks: list[dict[str, Any]] = []

            for raw_block in page_dict.get("blocks", []):
                if raw_block.get("type") != 0:
                    continue

                block_lines: list[dict[str, Any]] = []
                span_sizes: list[float] = []
                for line_index, raw_line in enumerate(raw_block.get("lines", [])):
                    spans = raw_line.get("spans", [])
                    for segment_index, segment in enumerate(_split_spans_into_line_segments(spans)):
                        line_text = _clean_inline_text(segment["text"])
                        if not line_text:
                            continue

                        sizes = segment["sizes"]
                        span_sizes.extend(sizes)
                        block_lines.append(
                            {
                                "line_no": line_index * 100 + segment_index,
                                "text": line_text,
                                "bbox": segment["bbox"],
                                "font_size": sum(sizes) / len(sizes) if sizes else 0.0,
                            }
                        )

                if not block_lines:
                    continue
                block_lines = _sort_line_segments_reading_order(block_lines, page_width)

                bbox = tuple(float(value) for value in raw_block.get("bbox", (0, 0, 0, 0)))
                text_blocks.append(
                    {
                        "doc_name": doc_name,
                        "page": page_index,
                        "page_width": page_width,
                        "page_height": page_height,
                        "block_id": int(raw_block.get("number", len(text_blocks))),
                        "bbox": bbox,
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3],
                        "font_size": sum(span_sizes) / len(span_sizes) if span_sizes else 0.0,
                        "lines": block_lines,
                    }
                )

            blocks.extend(_sort_blocks_reading_order(text_blocks))

    return blocks


def normalize_lines(blocks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten block lines, remove repeated headers/footers, page numbers, and TOC noise."""
    common_edge_lines = _detect_repeated_edge_lines(blocks)
    repeated_short_lines = _detect_repeated_short_lines(blocks)
    pages_to_skip = _detect_non_body_pages(blocks)
    normalized_lines: list[dict[str, Any]] = []
    previous_block_by_page: dict[int, dict[str, Any]] = {}

    for block in blocks:
        page = int(block["page"])
        if page in pages_to_skip:
            continue
        if _looks_like_table_or_chart_block(block):
            continue
        previous_block = previous_block_by_page.get(page)
        block_gap = float(block["y0"] - previous_block["y1"]) if previous_block else 0.0
        previous_block_by_page[page] = block

        for line in block["lines"]:
            text = _clean_inline_text(line["text"])
            normalized = _normalize_for_repetition(text)
            if not text:
                continue
            if normalized in common_edge_lines:
                continue
            if normalized in repeated_short_lines:
                continue
            if PAGE_NUMBER_RE.match(text):
                continue
            if _looks_like_toc_line(text):
                continue
            if _looks_like_table_or_figure_caption(text):
                continue
            if _looks_like_numeric_table_line(text):
                continue
            if _looks_like_table_fragment(text):
                continue

            bbox = line["bbox"]
            normalized_lines.append(
                {
                    "doc_name": block["doc_name"],
                    "page": page,
                    "page_width": float(block.get("page_width") or 0.0),
                    "block_id": block["block_id"],
                    "line_no": line["line_no"],
                    "text": text,
                    "x0": float(bbox[0]),
                    "y0": float(bbox[1]),
                    "x1": float(bbox[2]),
                    "y1": float(bbox[3]),
                    "font_size": float(line.get("font_size") or block.get("font_size") or 0.0),
                    "block_font_size": float(block.get("font_size") or 0.0),
                    "block_gap": block_gap,
                }
            )

    return _sort_lines_reading_order(normalized_lines)


def merge_broken_lines(lines: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair PDF hard line breaks without joining obvious titles to body text."""
    merged: list[dict[str, Any]] = []

    for line in lines:
        current = dict(line)
        text = current["text"]
        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        if _should_merge_lines(previous, current):
            separator = "" if previous["text"].endswith("-") else " "
            previous["text"] = _clean_inline_text(previous["text"].rstrip("-") + separator + text)
            previous["y1"] = current["y1"]
            previous["sentence_id_end_line"] = current["line_no"]
        else:
            merged.append(current)

    return merged


def rebuild_paragraphs(lines: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild natural paragraphs from repaired lines and PDF layout hints."""
    paragraphs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in lines:
        text = line["text"]
        if current is None:
            current = _new_paragraph(line)
            continue

        if _should_start_new_paragraph(current, line):
            paragraphs.append(_finalize_paragraph(current))
            current = _new_paragraph(line)
        else:
            current["texts"].append(text)
            current["page_end"] = line["page"]
            current["block_id_end"] = line["block_id"]
            current["font_sizes"].append(line["font_size"])

    if current is not None:
        paragraphs.append(_finalize_paragraph(current))

    return [paragraph for paragraph in paragraphs if paragraph["text"]]


def split_chinese_sentences(text: str) -> list[str]:
    """Split rebuilt paragraphs by Chinese sentence punctuation without cutting common false boundaries."""
    text = _clean_inline_text(text)
    if not text:
        return []

    sentences: list[str] = []
    buffer: list[str] = []
    quote_depth = 0

    for index, char in enumerate(text):
        buffer.append(char)
        if char in "「『“\"'":
            quote_depth += 1
        elif char in "」』”\"'":
            quote_depth = max(0, quote_depth - 1)

        if char not in "。！？；.!?;":
            continue
        if _is_false_sentence_boundary(text, index):
            continue
        if quote_depth > 0 and char not in "。！？；":
            continue

        sentence = _clean_inline_text("".join(buffer))
        if sentence:
            sentences.append(sentence)
        buffer = []

    tail = _clean_inline_text("".join(buffer))
    if tail:
        sentences.append(tail)
    return sentences


def build_chunks_by_sentence(
    paragraphs: Sequence[dict[str, Any]],
    target_min_chars: int = 150,
    target_max_chars: int = 350,
    hard_max_chars: int = 500,
    context_sentences: int = 1,
) -> list[dict[str, Any]]:
    """Build chunks by merging complete sentences, not by fixed character cuts."""
    sentence_rows: list[dict[str, Any]] = []
    sentence_id = 1

    for paragraph in paragraphs:
        sentences = split_chinese_sentences(paragraph["text"])
        for sentence in sentences:
            sentence_rows.append(
                {
                    "sentence_id": sentence_id,
                    "text": sentence,
                    "page": paragraph["page"],
                    "block_id": paragraph["block_id"],
                }
            )
            sentence_id += 1

    chunks: list[dict[str, Any]] = []
    index = 0
    while index < len(sentence_rows):
        start = index
        pieces = [sentence_rows[index]["text"]]
        total = len(pieces[0])
        index += 1

        while index < len(sentence_rows):
            next_sentence = sentence_rows[index]["text"]
            should_add = total < target_min_chars or total + len(next_sentence) <= target_max_chars
            if not should_add and total + len(next_sentence) > hard_max_chars:
                break
            if not should_add:
                break
            pieces.append(next_sentence)
            total += len(next_sentence)
            index += 1

        end = index - 1
        context_start = max(0, start - context_sentences)
        context_end = min(len(sentence_rows), end + context_sentences + 1)
        chunk_text = _clean_inline_text("".join(pieces))
        chunks.append(
            {
                "page": sentence_rows[start]["page"],
                "block_id": sentence_rows[start]["block_id"],
                "sentence_id_start": sentence_rows[start]["sentence_id"],
                "sentence_id_end": sentence_rows[end]["sentence_id"],
                "chunk_text": chunk_text,
                "context_text": _clean_inline_text("".join(row["text"] for row in sentence_rows[context_start:context_end])),
                "quality_warning": "",
            }
        )

    return chunks


def fix_bad_chunks(chunks: Sequence[dict[str, Any]], min_chars: int = 80, max_chars: int = 650) -> list[dict[str, Any]]:
    """Merge suspicious chunks with neighbors when the split quality is poor."""
    fixed: list[dict[str, Any]] = []

    for raw_chunk in chunks:
        chunk = dict(raw_chunk)
        warnings = _chunk_quality_warnings(chunk["chunk_text"], min_chars=min_chars)
        chunk["quality_warning"] = "; ".join(warnings)

        if fixed and warnings and len(fixed[-1]["chunk_text"]) + len(chunk["chunk_text"]) <= max_chars:
            previous = fixed[-1]
            previous["chunk_text"] = _clean_inline_text(previous["chunk_text"] + chunk["chunk_text"])
            previous["context_text"] = _clean_inline_text(previous["context_text"] + chunk["context_text"])
            previous["sentence_id_end"] = chunk["sentence_id_end"]
            previous["quality_warning"] = "; ".join(_chunk_quality_warnings(previous["chunk_text"], min_chars=min_chars))
            continue

        fixed.append(chunk)

    index = 0
    while index < len(fixed) - 1:
        warnings = _chunk_quality_warnings(fixed[index]["chunk_text"], min_chars=min_chars)
        if warnings and len(fixed[index]["chunk_text"]) + len(fixed[index + 1]["chunk_text"]) <= max_chars:
            fixed[index]["chunk_text"] = _clean_inline_text(fixed[index]["chunk_text"] + fixed[index + 1]["chunk_text"])
            fixed[index]["context_text"] = _clean_inline_text(fixed[index]["context_text"] + fixed[index + 1]["context_text"])
            fixed[index]["sentence_id_end"] = fixed[index + 1]["sentence_id_end"]
            fixed[index]["quality_warning"] = "; ".join(_chunk_quality_warnings(fixed[index]["chunk_text"], min_chars=min_chars))
            fixed.pop(index + 1)
        else:
            index += 1

    return fixed


def process_pdf(pdf_source: str | Path | BinaryIO | BytesIO) -> pd.DataFrame:
    """Run the full PDF text reconstruction and sentence-safe chunking pipeline."""
    doc_name = getattr(pdf_source, "name", None) or (Path(pdf_source).name if isinstance(pdf_source, (str, Path)) else "uploaded.pdf")
    blocks = extract_pdf_blocks(pdf_source)
    lines = normalize_lines(blocks)
    merged_lines = merge_broken_lines(lines)
    paragraphs = rebuild_paragraphs(merged_lines)
    chunks = fix_bad_chunks(build_chunks_by_sentence(paragraphs))

    rows = []
    for chunk_id, chunk in enumerate(chunks, start=1):
        rows.append(
            {
                "doc_name": doc_name,
                "page": chunk["page"],
                "chunk_id": chunk_id,
                "block_id": chunk["block_id"],
                "sentence_id_start": chunk["sentence_id_start"],
                "sentence_id_end": chunk["sentence_id_end"],
                "chunk_text": chunk["chunk_text"],
                "context_text": chunk["context_text"],
                "char_count": len(chunk["chunk_text"]),
                "quality_warning": chunk["quality_warning"],
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "doc_name",
            "page",
            "chunk_id",
            "block_id",
            "sentence_id_start",
            "sentence_id_end",
            "chunk_text",
            "context_text",
            "char_count",
            "quality_warning",
        ],
    )


def save_results(df: pd.DataFrame, output_csv_path: str | Path) -> None:
    """Save chunk results as UTF-8 with BOM for Chinese Excel compatibility."""
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")


def extract_pdf_text(file: BinaryIO | BytesIO) -> str:
    """Compatibility helper for the Streamlit app: return reconstructed chunk text."""
    df = process_pdf(file)
    if df.empty:
        return ""
    return "\n\n".join(df["chunk_text"].astype(str).tolist())


def _sort_blocks_reading_order(blocks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not blocks:
        return []
    page_width = float(blocks[0].get("page_width") or 1.0)
    left_blocks = [block for block in blocks if block["x0"] < page_width * 0.55]
    right_blocks = [block for block in blocks if block["x0"] >= page_width * 0.55]
    if len(left_blocks) >= 3 and len(right_blocks) >= 3:
        return sorted(left_blocks, key=lambda item: (item["y0"], item["x0"])) + sorted(right_blocks, key=lambda item: (item["y0"], item["x0"]))
    return sorted(blocks, key=lambda item: (round(item["y0"] / 8) * 8, item["x0"]))


def _split_spans_into_line_segments(spans: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_spans = [
        span
        for span in sorted(spans, key=lambda item: float(item.get("bbox", (0, 0, 0, 0))[0]))
        if str(span.get("text", "")).strip()
    ]
    if not clean_spans:
        return []

    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_x1: float | None = None
    previous_size = 0.0

    for span in clean_spans:
        bbox = tuple(float(value) for value in span.get("bbox", (0, 0, 0, 0)))
        size = float(span.get("size", 0.0) or 0.0)
        gap = bbox[0] - previous_x1 if previous_x1 is not None else 0.0
        large_gap = previous_x1 is not None and gap > max(24.0, max(previous_size, size) * 3.0)
        if current and large_gap:
            segments.append(current)
            current = []
        current.append(span)
        previous_x1 = bbox[2]
        previous_size = size

    if current:
        segments.append(current)

    return [_line_segment_from_spans(segment) for segment in segments]


def _sort_line_segments_reading_order(lines: Sequence[dict[str, Any]], page_width: float) -> list[dict[str, Any]]:
    if not lines:
        return []
    left_lines = [line for line in lines if float(line["bbox"][0]) < page_width * 0.55]
    right_lines = [line for line in lines if float(line["bbox"][0]) >= page_width * 0.55]
    if len(left_lines) >= 3 and len(right_lines) >= 3:
        return sorted(left_lines, key=lambda item: (item["bbox"][1], item["bbox"][0])) + sorted(
            right_lines,
            key=lambda item: (item["bbox"][1], item["bbox"][0]),
        )
    return sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _sort_lines_reading_order(lines: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_lines: list[dict[str, Any]] = []
    lines_by_page: dict[int, list[dict[str, Any]]] = {}
    for line in lines:
        lines_by_page.setdefault(int(line["page"]), []).append(line)

    for page in sorted(lines_by_page):
        page_lines = lines_by_page[page]
        if not page_lines:
            continue
        page_width = max(float(line.get("page_width") or 0.0) for line in page_lines) or 1.0
        body_lines = [
            line
            for line in page_lines
            if not (
                float(line["x0"]) < page_width * 0.18
                and len(str(line["text"])) <= 36
                and not CHINESE_PUNCT_RE.search(str(line["text"]))
            )
        ]
        working_lines = body_lines or page_lines
        columns = _cluster_lines_by_x(working_lines, min_gap=max(70.0, page_width * 0.1))

        if len(columns) >= 2 and sum(len(column) >= 4 for column in columns) >= 2:
            for column in columns:
                sorted_lines.extend(sorted(column, key=lambda item: (item["y0"], item["x0"])))
        else:
            sorted_lines.extend(sorted(working_lines, key=lambda item: (round(item["y0"] / 8) * 8, item["x0"])))

    return sorted_lines


def _cluster_lines_by_x(lines: Sequence[dict[str, Any]], min_gap: float) -> list[list[dict[str, Any]]]:
    ordered = sorted(lines, key=lambda item: (float(item["x0"]), float(item["y0"])))
    clusters: list[list[dict[str, Any]]] = []
    centers: list[float] = []

    for line in ordered:
        x0 = float(line["x0"])
        if not clusters or x0 - centers[-1] > min_gap:
            clusters.append([line])
            centers.append(x0)
            continue
        clusters[-1].append(line)
        centers[-1] = sum(float(item["x0"]) for item in clusters[-1]) / len(clusters[-1])

    return clusters


def _line_segment_from_spans(spans: Sequence[dict[str, Any]]) -> dict[str, Any]:
    bboxes = [tuple(float(value) for value in span.get("bbox", (0, 0, 0, 0))) for span in spans]
    sizes = [float(span.get("size", 0.0) or 0.0) for span in spans if str(span.get("text", "")).strip()]
    return {
        "text": "".join(str(span.get("text", "")) for span in spans),
        "sizes": sizes,
        "bbox": (
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        ),
    }


def _clean_inline_text(text: str) -> str:
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = re.sub(r"\s+([，。！？；：、,.!?;:%）)])", r"\1", text)
    text = re.sub(r"([（(])\s+", r"\1", text)
    return text


def _detect_repeated_edge_lines(blocks: Sequence[dict[str, Any]], edge_count: int = 3) -> set[str]:
    page_lines: dict[int, list[str]] = {}
    for block in blocks:
        page_lines.setdefault(int(block["page"]), [])
        page_lines[int(block["page"])].extend(_normalize_for_repetition(line["text"]) for line in block["lines"] if line["text"].strip())

    counter: Counter[str] = Counter()
    for lines in page_lines.values():
        for line in lines[:edge_count] + lines[-edge_count:]:
            if line and not PAGE_NUMBER_RE.match(line):
                counter[line] += 1

    min_count = max(3, int(len(page_lines) * 0.3))
    return {line for line, count in counter.items() if count >= min_count and len(line) <= 80}


def _detect_repeated_short_lines(blocks: Sequence[dict[str, Any]]) -> set[str]:
    page_hits: dict[str, set[int]] = {}
    page_count = len({int(block["page"]) for block in blocks})
    for block in blocks:
        page = int(block["page"])
        for line in block["lines"]:
            text = _normalize_for_repetition(line["text"])
            if not text or len(text) > 24 or CHINESE_PUNCT_RE.search(text):
                continue
            page_hits.setdefault(text, set()).add(page)

    min_pages = max(4, int(page_count * 0.18))
    return {text for text, pages in page_hits.items() if len(pages) >= min_pages}


def _detect_non_body_pages(blocks: Sequence[dict[str, Any]]) -> set[int]:
    page_lines: dict[int, list[str]] = {}
    for block in blocks:
        page_lines.setdefault(int(block["page"]), [])
        page_lines[int(block["page"])].extend(line["text"] for line in block["lines"] if line["text"].strip())

    skip_pages: set[int] = set()
    for page, lines in page_lines.items():
        normalized = [_clean_inline_text(line) for line in lines if _clean_inline_text(line)]
        if not normalized:
            skip_pages.add(page)
            continue
        first_lines = normalized[:10]
        toc_heading = any(line.lower() in {"目錄", "contents", "table of contents"} for line in first_lines)
        toc_count = sum(1 for line in normalized if _looks_like_toc_line(line))
        short_heading_count = sum(1 for line in normalized if len(line) <= 24 and not CHINESE_PUNCT_RE.search(line))
        if toc_heading and (toc_count >= 3 or short_heading_count / max(len(normalized), 1) > 0.5):
            skip_pages.add(page)
        elif len(normalized) >= 12 and toc_count / len(normalized) > 0.25:
            skip_pages.add(page)
    return skip_pages


def _normalize_for_repetition(text: str) -> str:
    return _clean_inline_text(text).lower()


def _looks_like_toc_line(text: str) -> bool:
    return bool(re.search(r"(\.{3,}|…{2,})\s*\d{1,4}$", text) or re.match(r"^\d+(?:\.\d+)*\s+.+\s+\d{1,4}$", text))


def _looks_like_table_or_figure_caption(text: str) -> bool:
    normalized = _clean_inline_text(text)
    if not TABLE_FIGURE_CAPTION_RE.match(normalized):
        return False
    return len(normalized) <= 120 and normalized.count("。") == 0


def _looks_like_table_or_chart_block(block: dict[str, Any]) -> bool:
    lines = [_clean_inline_text(line["text"]) for line in block.get("lines", [])]
    lines = [line for line in lines if line]
    if len(lines) < 3:
        return False

    numeric_lines = sum(1 for line in lines if _looks_like_numeric_table_line(line) or _looks_like_table_fragment(line))
    caption_lines = sum(1 for line in lines if _looks_like_table_or_figure_caption(line))
    no_sentence_lines = sum(1 for line in lines if not CHINESE_PUNCT_RE.search(line))
    short_lines = sum(1 for line in lines if len(line) <= 28)
    repeated_x_positions = _has_repeated_column_positions(block.get("lines", []))

    if caption_lines and numeric_lines:
        return True
    if numeric_lines >= 3 and numeric_lines / len(lines) >= 0.45:
        return True
    if repeated_x_positions and no_sentence_lines / len(lines) >= 0.75 and short_lines / len(lines) >= 0.55:
        return True
    return False


def _looks_like_numeric_table_line(text: str) -> bool:
    normalized = _clean_inline_text(text)
    if not normalized:
        return True
    if TABLE_UNIT_RE.search(normalized):
        return True
    if YEAR_HEADER_RE.match(normalized):
        return True

    tokens = NUMERIC_TOKEN_RE.findall(normalized)
    digit_count = sum(char.isdigit() for char in normalized)
    has_sentence_punct = bool(CHINESE_PUNCT_RE.search(normalized))
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", normalized))

    if len(tokens) >= 3 and not has_sentence_punct:
        return True
    if len(tokens) >= 2 and len(normalized) <= 36 and not has_sentence_punct:
        return True
    if digit_count >= 4 and digit_count / max(len(normalized), 1) >= 0.45 and not has_sentence_punct:
        return True
    if not has_cjk and len(tokens) >= 1 and len(normalized) <= 18:
        return True
    return False


def _has_repeated_column_positions(lines: Sequence[dict[str, Any]]) -> bool:
    if len(lines) < 4:
        return False

    rounded_positions = [round(float(line.get("bbox", (0, 0, 0, 0))[0]) / 8) * 8 for line in lines]
    counts = Counter(rounded_positions)
    return sum(1 for count in counts.values() if count >= 2) >= 2


def _looks_like_table_fragment(text: str) -> bool:
    if len(text) < 2:
        return True
    digits = sum(char.isdigit() for char in text)
    separators = sum(text.count(char) for char in "|｜")
    if separators >= 3:
        return True
    return len(text) >= 12 and digits / len(text) > 0.65 and not CHINESE_PUNCT_RE.search(text)


def _looks_like_title(line: dict[str, Any] | str) -> bool:
    text = line if isinstance(line, str) else str(line.get("text", ""))
    if not text:
        return False
    if BULLET_RE.match(text):
        return False
    if len(text) <= 32 and not CHINESE_PUNCT_RE.search(text):
        return True
    return bool(re.match(r"^\d+(?:\.\d+)*\s+\S.{0,40}$", text) and not CHINESE_PUNCT_RE.search(text))


def _should_merge_lines(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    previous_text = previous["text"]
    current_text = current["text"]
    if previous["page"] != current["page"]:
        return False
    if _looks_like_title(previous):
        return False
    if BULLET_RE.match(current_text):
        return False
    if previous["block_id"] != current["block_id"] and float(current.get("block_gap", 0.0)) > max(12.0, previous.get("font_size", 0.0) * 1.3):
        return False
    if previous_text.endswith(tuple(SENTENCE_END + "：:")) and not current_text[:1].islower():
        return False
    if previous_text.endswith(SOFT_END_WORDS):
        return True
    if re.match(r"^[a-z0-9%（(]", current_text):
        return True
    return not previous_text.endswith(tuple(SENTENCE_END + "：:"))


def _new_paragraph(line: dict[str, Any]) -> dict[str, Any]:
    return {
        "page": line["page"],
        "page_end": line["page"],
        "block_id": line["block_id"],
        "block_id_end": line["block_id"],
        "texts": [line["text"]],
        "font_sizes": [line["font_size"]],
    }


def _finalize_paragraph(paragraph: dict[str, Any]) -> dict[str, Any]:
    text = _clean_inline_text(" ".join(paragraph["texts"]))
    return {
        "page": paragraph["page"],
        "page_end": paragraph["page_end"],
        "block_id": paragraph["block_id"],
        "block_id_end": paragraph["block_id_end"],
        "text": text,
        "font_size": sum(paragraph["font_sizes"]) / len(paragraph["font_sizes"]),
    }


def _should_start_new_paragraph(current: dict[str, Any], line: dict[str, Any]) -> bool:
    previous_text = current["texts"][-1]
    current_text = line["text"]
    if line["page"] != current["page_end"]:
        return True
    if BULLET_RE.match(current_text):
        return True
    if _looks_like_title(current_text):
        return True
    if float(line.get("block_gap", 0.0)) > max(14.0, line.get("font_size", 0.0) * 1.5):
        return True
    current_font = sum(current["font_sizes"]) / len(current["font_sizes"])
    if current_font and abs(line["font_size"] - current_font) >= 2.0 and _looks_like_title(current_text):
        return True
    return bool(previous_text.endswith(tuple(SENTENCE_END)) and len("".join(current["texts"])) >= 120)


def _is_false_sentence_boundary(text: str, index: int) -> bool:
    char = text[index]
    previous_char = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""
    if char == ".":
        if previous_char.isdigit() and next_char.isdigit():
            return True
        before = text[max(0, index - 8) : index + 1]
        if ABBR_RE.search(before):
            return True
        if URL_RE.search(text[max(0, index - 30) : min(len(text), index + 30)]):
            return True
    if char in "/-" and previous_char.isdigit() and next_char.isdigit():
        return True
    return False


def _chunk_quality_warnings(text: str, min_chars: int) -> list[str]:
    stripped = text.strip()
    warnings: list[str] = []
    if len(stripped) < min_chars:
        warnings.append("too short")
    if stripped.startswith(BAD_START_WORDS):
        warnings.append("odd start")
    if stripped.endswith(BAD_END_WORDS):
        warnings.append("odd end")
    if not CHINESE_PUNCT_RE.search(stripped):
        warnings.append("no Chinese sentence punctuation")
    return warnings

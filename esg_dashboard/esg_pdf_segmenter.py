from __future__ import annotations

# Google Colab install notes:
#   !pip install -q pymupdf pandas numpy scikit-learn sentence-transformers torch
#
# Example:
#   df = process_pdf("/content/company_esg_report.pdf")
#   save_results(df, "/content/esg_candidate_segments.csv")

from collections import Counter
from collections.abc import Sequence
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
import argparse
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"

DEFAULT_QUERIES = [
    "企業提出明確 ESG 承諾或未來目標",
    "文本包含減碳、淨零、永續發展目標",
    "文本包含第三方驗證、認證、查證或審核",
    "文本包含具體數據、年份、比例、達成成果",
    "文本包含承諾但缺少具體證據",
]

ESG_KEYWORDS = [
    "ESG", "永續", "永續發展", "永續經營", "企業社會責任", "CSR", "GRI", "SASB", "TCFD",
    "氣候", "氣候變遷", "溫室氣體", "碳", "碳排", "碳排放", "減碳", "淨零", "碳中和",
    "再生能源", "綠電", "能源", "節能", "水資源", "廢棄物", "污染", "循環經濟",
    "人權", "勞工", "職安", "職業安全", "供應鏈", "員工", "多元", "平等", "包容",
    "公司治理", "董事會", "風險管理", "法遵", "誠信經營", "反貪腐", "資訊安全",
    "sustainability", "sustainable", "net zero", "carbon neutral", "emission", "emissions",
    "greenhouse gas", "GHG", "renewable energy", "human rights", "occupational safety",
    "corporate governance", "risk management", "anti-corruption",
]

COMMITMENT_TERMS = [
    "將", "預計", "目標", "承諾", "規劃", "致力於", "持續", "推動", "導入", "提升",
    "達成", "實現", "願景", "藍圖", "里程碑", "plan", "target", "goal", "commit",
    "commitment", "aim", "will", "strive", "roadmap",
]

EVIDENCE_TERMS = [
    "完成", "達成", "取得", "通過", "驗證", "認證", "第三方", "查證", "確信", "審核",
    "稽核", "盤查", "揭露", "符合", "ISO", "BSI", "SGS", "DNV", "KPMG", "PwC", "EY",
    "Deloitte", "verified", "verification", "certified", "assurance", "audited", "reviewed",
]

SLOGAN_TERMS = ["願景", "使命", "核心價值", "標語", "口號", "我們的承諾", "永續願景"]

WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?\d{1,4}(?:\s*/\s*\d{1,4}|\s+of\s+\d{1,4})?\s*$",
    re.IGNORECASE,
)
TOC_LINE_RE = re.compile(r"(\.{3,}|…{2,})\s*\d{1,4}$|^\s*\d+(?:\.\d+)*\s+.+\s+\d{1,4}$")
TOC_HEADING_RE = re.compile(r"^\s*(目錄|contents?|table of contents)\s*$", re.IGNORECASE)
REFERENCE_HEADING_RE = re.compile(
    r"^\s*(參考文獻|參考資料|附錄|references?|bibliography|appendix)\s*$",
    re.IGNORECASE,
)
NUMBER_SIGNAL_RE = re.compile(
    r"(\d{4}\s*年|\d{1,2}\s*月|\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*(?:噸|公噸|tCO2e|CO2e|度|kWh|MWh|GWh|人|件|家|次|億元|萬元))",
    re.IGNORECASE,
)
SENTENCE_END_RE = re.compile(r"([。！？!?；;])")


def extract_pdf_text(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract page text from a PDF with PyMuPDF and keep page numbers."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("Please install PyMuPDF first: pip install pymupdf") from exc

    pdf_path = Path(pdf_path)
    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            pages.append({"page": page_index + 1, "text": text})
    return pages


def clean_text(text: str, common_lines: set[str] | None = None) -> str:
    """Remove PDF noise and normalize whitespace while preserving paragraphs."""
    if not text:
        return ""

    common_lines = common_lines or set()
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    raw_lines = text.splitlines()
    cleaned_lines: list[str] = []

    for raw_line in raw_lines:
        line = WHITESPACE_RE.sub(" ", raw_line).strip()
        normalized = _normalize_line(line)
        if not line:
            cleaned_lines.append("")
            continue
        if normalized in common_lines:
            continue
        if PAGE_NUMBER_RE.match(line):
            continue
        if _looks_like_noise_line(line):
            continue
        cleaned_lines.append(line)

    paragraphs = _join_wrapped_lines(cleaned_lines)
    cleaned = "\n\n".join(paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def split_into_paragraphs(
    pages: Sequence[dict[str, Any]],
    min_chars: int = 80,
    max_chars: int = 1_200,
    context_window: int = 1,
) -> list[dict[str, Any]]:
    """Split cleaned page text into semantic paragraphs and attach nearby context."""
    common_lines = _detect_common_header_footer_lines(pages)
    paragraphs: list[dict[str, Any]] = []
    paragraph_id = 1
    stop_after_references = False

    for page_record in pages:
        page = int(page_record["page"])
        raw_text = str(page_record.get("text", ""))
        if stop_after_references:
            break
        if _is_toc_page(raw_text):
            continue
        if _starts_reference_section(raw_text):
            stop_after_references = True
            continue

        cleaned = clean_text(raw_text, common_lines=common_lines)
        for paragraph in re.split(r"\n{2,}", cleaned):
            paragraph = _normalize_paragraph(paragraph)
            if not paragraph:
                continue
            for chunk in _split_long_paragraph(paragraph, max_chars=max_chars):
                if len(chunk) < min_chars and not _has_any(chunk, ESG_KEYWORDS):
                    continue
                paragraphs.append(
                    {
                        "page": page,
                        "paragraph_id": paragraph_id,
                        "text": chunk,
                    }
                )
                paragraph_id += 1

    for index, paragraph in enumerate(paragraphs):
        start = max(0, index - context_window)
        end = min(len(paragraphs), index + context_window + 1)
        paragraph["context_text"] = _merge_context([item["text"] for item in paragraphs[start:end]])

    return paragraphs


def build_embeddings(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    is_query: bool = False,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Build multilingual E5 embeddings with sentence-transformers."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = _load_sentence_transformer(model_name)
    prefix = "query: " if is_query else "passage: "
    encoded_texts = [prefix + str(text).strip() for text in texts]
    embeddings = model.encode(
        encoded_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return embeddings.astype(np.float32)


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Please install sentence-transformers: pip install sentence-transformers") from exc

    return SentenceTransformer(model_name)


def retrieve_candidates(
    paragraphs: Sequence[dict[str, Any]],
    queries: Sequence[str] | None = None,
    keywords: Sequence[str] | None = None,
    similarity_threshold: float = 0.78,
    top_k_per_query: int = 20,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Recall candidate paragraphs by ESG keywords and embedding similarity."""
    queries = list(queries or DEFAULT_QUERIES)
    keywords = list(keywords or ESG_KEYWORDS)
    if not paragraphs:
        return []

    paragraph_texts = [str(item["context_text"]) for item in paragraphs]
    paragraph_embeddings = build_embeddings(paragraph_texts, model_name=model_name, batch_size=batch_size)
    query_embeddings = build_embeddings(queries, model_name=model_name, batch_size=batch_size, is_query=True)
    scores = cosine_similarity(paragraph_embeddings, query_embeddings)

    selected_indices: set[int] = set()
    reasons_by_index: dict[int, set[str]] = {}

    for index, paragraph in enumerate(paragraphs):
        matched = _matched_keywords(paragraph["context_text"], keywords)
        if matched:
            selected_indices.add(index)
            reasons_by_index.setdefault(index, set()).add("keyword")

        best_score = float(scores[index].max())
        if best_score >= similarity_threshold:
            selected_indices.add(index)
            reasons_by_index.setdefault(index, set()).add("similarity")

    for query_index, _query in enumerate(queries):
        top_indices = np.argsort(scores[:, query_index])[-top_k_per_query:]
        for index in top_indices:
            selected_indices.add(int(index))
            reasons_by_index.setdefault(int(index), set()).add("top_k")

    rows: list[dict[str, Any]] = []
    for index in sorted(selected_indices):
        paragraph = paragraphs[index]
        query_scores = scores[index]
        best_query_index = int(np.argmax(query_scores))
        text = str(paragraph["context_text"])
        matched_keywords = _matched_keywords(text, keywords)
        rule_result = filter_candidate_with_rules(text)
        keep_reasons = sorted(reasons_by_index.get(index, set()))
        if rule_result["keep_reason"]:
            keep_reasons.extend(rule_result["keep_reason"].split("; "))

        rows.append(
            {
                "page": paragraph["page"],
                "paragraph_id": paragraph["paragraph_id"],
                "text": text,
                "matched_keywords": ", ".join(matched_keywords),
                "similarity_score": round(float(query_scores[best_query_index]), 4),
                "matched_query": queries[best_query_index],
                "rule_score": int(rule_result["rule_score"]),
                "char_count": len(text),
                "keep_reason": "; ".join(dict.fromkeys(keep_reasons)),
                "_keep_by_rules": bool(rule_result["keep"]),
            }
        )

    return [row for row in rows if row["_keep_by_rules"] or row["matched_keywords"] or row["similarity_score"] >= similarity_threshold]


def filter_candidate_with_rules(text: str) -> dict[str, Any]:
    """Score whether a segment is worth keeping for downstream ESG NLP tasks."""
    normalized = _normalize_paragraph(text)
    char_count = len(normalized)
    matched_esg = _matched_keywords(normalized, ESG_KEYWORDS)
    has_commitment = _has_any(normalized, COMMITMENT_TERMS)
    has_evidence = _has_any(normalized, EVIDENCE_TERMS)
    has_number = bool(NUMBER_SIGNAL_RE.search(normalized) or re.search(r"\b20\d{2}\b|\b19\d{2}\b", normalized))
    is_title = _looks_like_title_or_slogan(normalized)

    score = 0
    reasons: list[str] = []
    if matched_esg:
        score += 2
        reasons.append("ESG topic")
    if has_commitment:
        score += 2
        reasons.append("commitment language")
    if has_evidence:
        score += 2
        reasons.append("evidence signal")
    if has_number:
        score += 1
        reasons.append("number/timeline")
    if 120 <= char_count <= 1_800:
        score += 1
        reasons.append("complete length")
    if char_count < 80:
        score -= 3
        reasons.append("too short")
    if char_count > 2_500:
        score -= 2
        reasons.append("too long")
    if is_title:
        score -= 3
        reasons.append("title/slogan")

    keep = score >= 3 and not (is_title and score < 5)
    return {
        "keep": keep,
        "rule_score": max(0, score),
        "keep_reason": "; ".join(reasons),
    }


def deduplicate_segments(
    rows: Sequence[dict[str, Any]],
    similarity_threshold: float = 0.9,
) -> list[dict[str, Any]]:
    """Remove exact and near-duplicate candidate segments, keeping stronger rows."""
    sorted_rows = sorted(
        rows,
        key=lambda row: (float(row.get("rule_score", 0)), float(row.get("similarity_score", 0)), len(str(row.get("text", "")))),
        reverse=True,
    )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in sorted_rows:
        text = str(row.get("text", ""))
        normalized = _normalize_for_dedupe(text)
        if not normalized or normalized in seen:
            continue
        if any(_text_similarity(normalized, _normalize_for_dedupe(str(old.get("text", "")))) >= similarity_threshold for old in unique):
            continue
        seen.add(normalized)
        unique.append(dict(row))

    unique.sort(key=lambda row: (int(row["page"]), int(row["paragraph_id"])))
    for new_id, row in enumerate(unique, start=1):
        row["paragraph_id"] = new_id
        row.pop("_keep_by_rules", None)
    return unique


def process_pdf(
    pdf_path: str | Path,
    output_csv_path: str | Path | None = None,
    queries: Sequence[str] | None = None,
    keywords: Sequence[str] | None = None,
    similarity_threshold: float = 0.78,
    top_k_per_query: int = 20,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Run the full three-stage ESG PDF candidate extraction pipeline."""
    pdf_path = Path(pdf_path)
    pages = extract_pdf_text(pdf_path)
    paragraphs = split_into_paragraphs(pages)
    candidates = retrieve_candidates(
        paragraphs,
        queries=queries,
        keywords=keywords,
        similarity_threshold=similarity_threshold,
        top_k_per_query=top_k_per_query,
        model_name=model_name,
        batch_size=batch_size,
    )
    candidates = deduplicate_segments(candidates)

    rows = [
        {
            "doc_name": pdf_path.name,
            "page": row["page"],
            "paragraph_id": row["paragraph_id"],
            "text": row["text"],
            "matched_keywords": row["matched_keywords"],
            "similarity_score": row["similarity_score"],
            "matched_query": row["matched_query"],
            "rule_score": row["rule_score"],
            "char_count": row["char_count"],
            "keep_reason": row["keep_reason"],
        }
        for row in candidates
    ]
    df = pd.DataFrame(rows, columns=_output_columns())
    if output_csv_path is not None:
        save_results(df, output_csv_path)
    return df


def save_results(df: pd.DataFrame, output_csv_path: str | Path) -> None:
    """Save results as UTF-8 with BOM so Excel handles Chinese correctly."""
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")


def _output_columns() -> list[str]:
    return [
        "doc_name",
        "page",
        "paragraph_id",
        "text",
        "matched_keywords",
        "similarity_score",
        "matched_query",
        "rule_score",
        "char_count",
        "keep_reason",
    ]


def _detect_common_header_footer_lines(pages: Sequence[dict[str, Any]], edge_lines: int = 4) -> set[str]:
    counter: Counter[str] = Counter()
    for page_record in pages:
        lines = [_normalize_line(line) for line in str(page_record.get("text", "")).splitlines() if line.strip()]
        for line in lines[:edge_lines] + lines[-edge_lines:]:
            if line and not PAGE_NUMBER_RE.match(line):
                counter[line] += 1
    min_count = max(3, int(len(pages) * 0.3))
    return {line for line, count in counter.items() if count >= min_count and len(line) <= 80}


def _normalize_line(line: str) -> str:
    return WHITESPACE_RE.sub(" ", line).strip().lower()


def _normalize_paragraph(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([，。；：、！？,.!?;:%])", r"\1", text)
    return text


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
        elif _should_join_without_space(buffer, line):
            buffer += line
        else:
            buffer += " " + line
    if buffer.strip():
        paragraphs.append(buffer.strip())
    return paragraphs


def _should_join_without_space(previous: str, current: str) -> bool:
    if previous.endswith(("。", "！", "？", ".", "!", "?", "；", ";", "：", ":")):
        return False
    if re.search(r"[A-Za-z0-9)]$", previous) and re.search(r"^[A-Za-z0-9(]", current):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]$", previous) or re.search(r"^[\u4e00-\u9fff]", current))


def _split_long_paragraph(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = _split_sentences(text)
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if not buffer or len(buffer) + len(sentence) <= max_chars:
            buffer += sentence
        else:
            chunks.append(buffer.strip())
            buffer = sentence
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks or [text[:max_chars].strip()]


def _split_sentences(text: str) -> list[str]:
    parts = SENTENCE_END_RE.split(text)
    sentences: list[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        buffer += part
        if SENTENCE_END_RE.fullmatch(part):
            sentences.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        sentences.append(buffer.strip())
    return sentences


def _merge_context(paragraphs: Sequence[str], max_chars: int = 1_800) -> str:
    text = "\n".join(_normalize_paragraph(paragraph) for paragraph in paragraphs if paragraph)
    if len(text) <= max_chars:
        return text
    sentences = _split_sentences(text)
    kept: list[str] = []
    total = 0
    for sentence in sentences:
        if total + len(sentence) > max_chars and kept:
            break
        kept.append(sentence)
        total += len(sentence)
    return "".join(kept).strip()


def _matched_keywords(text: str, keywords: Sequence[str]) -> list[str]:
    lowered = text.lower()
    matched = []
    for keyword in keywords:
        if keyword and keyword.lower() in lowered:
            matched.append(keyword)
    return list(dict.fromkeys(matched))


def _has_any(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms if term)


def _looks_like_noise_line(line: str) -> bool:
    if len(line) <= 1:
        return True
    if TOC_LINE_RE.search(line):
        return True
    if re.fullmatch(r"[\W_]+", line):
        return True
    digits = sum(char.isdigit() for char in line)
    if len(line) >= 12 and digits / len(line) > 0.65:
        return True
    return False


def _is_toc_page(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    if any(TOC_HEADING_RE.match(line) for line in lines[:8]):
        return True
    toc_lines = sum(1 for line in lines if TOC_LINE_RE.search(line))
    return len(lines) >= 8 and toc_lines / len(lines) > 0.35


def _starts_reference_section(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return any(REFERENCE_HEADING_RE.match(line) for line in lines[:10])


def _looks_like_title_or_slogan(text: str) -> bool:
    if len(text) <= 40 and not NUMBER_SIGNAL_RE.search(text):
        return True
    if len(text) <= 80 and _has_any(text, SLOGAN_TERMS) and text.count("。") == 0:
        return True
    return False


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


def _text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if len(left) < 500 and len(right) < 500:
        return SequenceMatcher(None, left, right).ratio()
    left_set = set(_char_ngrams(left))
    right_set = set(_char_ngrams(right))
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _char_ngrams(text: str, n: int = 5) -> list[str]:
    if len(text) <= n:
        return [text]
    return [text[index : index + n] for index in range(len(text) - n + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract high-quality ESG candidate paragraphs from a PDF.")
    parser.add_argument("pdf_path", help="Path to the ESG PDF.")
    parser.add_argument("-o", "--output", default="esg_candidate_segments.csv", help="Output CSV path.")
    parser.add_argument("--threshold", type=float, default=0.78, help="Embedding similarity threshold.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k paragraphs to keep per query.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="sentence-transformers model name.")
    args = parser.parse_args()

    df = process_pdf(
        args.pdf_path,
        output_csv_path=args.output,
        similarity_threshold=args.threshold,
        top_k_per_query=args.top_k,
        model_name=args.model,
    )
    print(f"Saved {len(df)} ESG candidate segments to {args.output}")


if __name__ == "__main__":
    main()

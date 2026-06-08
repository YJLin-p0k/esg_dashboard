from __future__ import annotations

import importlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from esg_dashboard.core.config import ESG_CATEGORY_ORDER, ESG_TYPE_TO_CATEGORY
from esg_dashboard.core.hybrid_model import ESG_CATEGORIES, HybridESGAnalyzer, normalize_task_outputs
from esg_dashboard.core.scoring import calculate_greenwashing_risk, compute_reference_trust_score
from esg_dashboard.core.taxonomy import detect_company, detect_topic
from esg_dashboard.data import pdf_utils
from esg_dashboard.data.text_utils import split_chinese_sentence_units


APP_DIR = Path(__file__).resolve().parents[2]
TRAINING_DATA_PATH = APP_DIR / "data" / "vpesg_4k_train_1000.json"


def process_pdf_chunks(pdf_source):
    """Load the PDF processor lazily so Streamlit cannot keep a stale symbol."""
    if not hasattr(pdf_utils, "process_pdf"):
        importlib.reload(pdf_utils)
    return pdf_utils.process_pdf(pdf_source)


@st.cache_resource
def load_analyzer() -> HybridESGAnalyzer:
    return HybridESGAnalyzer()


def normalize_task_columns(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows

    normalized = rows.copy()
    task_rows = pd.DataFrame(
        [
            normalize_task_outputs(
                promise_status=str(row.get("promise_status", "No")),
                evidence_status=str(row.get("evidence_status", "N/A")),
                verification_timeline=str(row.get("verification_timeline", "N/A")),
                evidence_quality=str(row.get("evidence_quality", "N/A")),
            )
            for _, row in normalized.iterrows()
        ],
        columns=["promise_status", "evidence_status", "verification_timeline", "evidence_quality"],
        index=normalized.index,
    )
    for column in task_rows.columns:
        normalized[column] = task_rows[column]
    return normalized


@st.cache_data
def load_training_peer_rows() -> pd.DataFrame:
    rows = pd.read_json(path_or_buf=str(TRAINING_DATA_PATH))
    rows["company"] = rows["company"].astype(str).str.lower()
    rows["esg_category"] = rows["esg_type"].map(ESG_TYPE_TO_CATEGORY)
    rows["evidence_quality"] = rows["evidence_quality"].replace("", "N/A").fillna("N/A")
    rows["evidence_status"] = rows["evidence_status"].replace("", "N/A").fillna("N/A")
    rows["verification_timeline"] = rows["verification_timeline"].replace("", "N/A").fillna("N/A")
    rows["promise_status"] = rows["promise_status"].replace("", "No").fillna("No")
    rows = normalize_task_columns(rows)
    rows["overall_trust_score"] = rows.apply(compute_reference_trust_score, axis=1)
    rows["sentence"] = rows["data"].astype(str)
    rows["topic"] = rows.apply(lambda row: detect_topic(str(row["sentence"]), str(row["esg_category"])), axis=1)
    rows = rows[rows["topic"].notna()].copy()
    return rows


def build_results(uploaded_files) -> pd.DataFrame:
    analyzer = load_analyzer()
    rows: list[dict[str, object]] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        chunk_df = process_pdf_chunks(BytesIO(file_bytes))
        text = "\n\n".join(chunk_df["chunk_text"].astype(str).tolist()) if not chunk_df.empty else ""
        company = detect_company(uploaded_file.name, text)
        chunk_units: list[tuple[dict[str, object], object]] = []
        for chunk in chunk_df.to_dict("records"):
            for unit in split_chinese_sentence_units(str(chunk["chunk_text"])):
                chunk_units.append((chunk, unit))

        sentences = [str(getattr(unit, "sentence", "")) for _, unit in chunk_units]
        predictions = analyzer.predict(sentences)

        for sentence_id, ((chunk, unit), prediction) in enumerate(zip(chunk_units, predictions), start=1):
            sentence = str(getattr(unit, "sentence", ""))
            paragraph_id = int(getattr(unit, "paragraph_id", 0) or 0)
            paragraph_text = str(getattr(unit, "paragraph_text", ""))
            paragraph_context = str(getattr(unit, "paragraph_context", ""))
            if prediction.esg_category not in ESG_CATEGORIES or prediction.esg_category == "Other":
                continue
            topic = detect_topic(sentence, prediction.esg_category)
            if topic is None:
                continue
            promise_status, evidence_status, verification_timeline, evidence_quality = normalize_task_outputs(
                promise_status=prediction.promise_status,
                evidence_status=prediction.evidence_status,
                verification_timeline=prediction.verification_timeline,
                evidence_quality=prediction.evidence_quality,
            )

            row = {
                "file_name": uploaded_file.name,
                "company": company,
                "page": int(chunk.get("page", 0) or 0),
                "paragraph_id": int(chunk.get("chunk_id", paragraph_id) or paragraph_id),
                "paragraph_text": paragraph_text,
                "paragraph_context": str(chunk.get("context_text", paragraph_context)),
                "sentence_id": sentence_id,
                "sentence": sentence,
                "esg_category": prediction.esg_category,
                "topic": topic,
                "overall_trust_score": prediction.overall_trust_score,
                "confidence": prediction.confidence,
                "promise_status": promise_status,
                "verification_timeline": verification_timeline,
                "evidence_status": evidence_status,
                "evidence_quality": evidence_quality,
            }
            row.update(calculate_greenwashing_risk(pd.Series(row)))
            rows.append(row)

    return pd.DataFrame(rows)


def build_issue_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = result_df.sort_values(
        ["file_name", "company", "esg_category", "overall_trust_score", "topic"],
        ascending=[True, True, True, True, True],
    )

    summary = (
        sorted_df.groupby(["file_name", "company", "esg_category", "topic"], as_index=False)
        .agg(
            overall_trust_score=("overall_trust_score", "mean"),
            avg_confidence=("confidence", "min"),
            evidence_count=("sentence", "count"),
            promise_rate=("promise_status", lambda values: round((values == "Yes").mean() * 100, 1)),
            clear_evidence_rate=("evidence_quality", lambda values: round((values == "Clear").mean() * 100, 1)),
            timeline_milestone=("verification_timeline", lambda values: values.mode().iat[0] if not values.mode().empty else "N/A"),
            representative_sentence=("sentence", "first"),
        )
    )
    summary["_esg_order"] = summary["esg_category"].map(ESG_CATEGORY_ORDER).fillna(99)
    return (
        summary.sort_values(
            ["file_name", "_esg_order", "overall_trust_score", "topic"],
            ascending=[True, True, True, True],
        )
        .drop(columns="_esg_order")
        .reset_index(drop=True)
    )

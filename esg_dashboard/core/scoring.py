from __future__ import annotations

import pandas as pd

from esg_dashboard.core.hybrid_model import OFFICIAL_WEIGHTS

from .config import PEER_CATEGORY_LABELS, TRUST_COLOR_HIGH, TRUST_COLOR_LOW, TRUST_COLOR_MEDIUM, TRUST_LOW_THRESHOLD, TRUST_STABLE_THRESHOLD


def compute_reference_trust_score(row: pd.Series) -> float:
    promise_status = str(row.get("promise_status", "No"))
    evidence_status = str(row.get("evidence_status", "N/A"))
    evidence_quality = str(row.get("evidence_quality", "N/A"))
    verification_timeline = str(row.get("verification_timeline", "N/A"))

    promise_score = 1.0 if promise_status == "Yes" else 0.55
    evidence_score = {"Yes": 1.0, "No": 0.25, "N/A": 0.55}.get(evidence_status, 0.55)
    quality_score = {"Clear": 1.0, "Not Clear": 0.35, "Misleading": 0.0, "N/A": 0.55}.get(evidence_quality, 0.55)
    timeline_score = {
        "already": 1.0,
        "within_2_years": 0.85,
        "between_2_and_5_years": 0.65,
        "longer_than_5_years": 0.35,
        "N/A": 0.55,
    }.get(verification_timeline, 0.55)

    weighted = (
        promise_score * OFFICIAL_WEIGHTS["promise_status"]
        + evidence_score * OFFICIAL_WEIGHTS["evidence_status"]
        + quality_score * OFFICIAL_WEIGHTS["evidence_quality"]
        + timeline_score * OFFICIAL_WEIGHTS["verification_timeline"]
    )
    return round(max(0, min(100, weighted * 100)), 2)


def format_score_metric(value: float | None) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value:.1f}"


def calculate_overall_trust_score(rows: pd.DataFrame) -> float | None:
    if rows.empty or "overall_trust_score" not in rows:
        return None
    return round(float(rows["overall_trust_score"].mean()), 2)


def severity_from_trust(score: float) -> tuple[str, str]:
    if score < TRUST_LOW_THRESHOLD:
        return "低信任", TRUST_COLOR_LOW
    if score < TRUST_STABLE_THRESHOLD:
        return "需追蹤", TRUST_COLOR_MEDIUM
    return "穩健", TRUST_COLOR_HIGH


def peer_score_comment(score: float) -> str:
    if score < TRUST_LOW_THRESHOLD:
        return "信任分數偏低，建議優先檢查證據品質與時程揭露。"
    if score < TRUST_STABLE_THRESHOLD:
        return "信任分數中等，揭露基礎尚可，但仍需要追蹤證據完整性。"
    return "信任分數較穩定，承諾、證據與時程訊號大致一致。"


def calculate_esg_trust_scores(rows: pd.DataFrame) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for category in PEER_CATEGORY_LABELS:
        category_rows = rows[rows["esg_category"].eq(category)]
        scores[category] = None if category_rows.empty else round(float(category_rows["overall_trust_score"].mean()), 2)
    return scores

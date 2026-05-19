from __future__ import annotations

import re


COMMITMENT_PATTERNS = [
    r"承諾",
    r"保證",
    r"確保",
    r"必(?:須|定)",
    r"將(?:於|在|會)?",
    r"預計",
    r"目標",
    r"達成",
    r"實現",
    r"完成",
]

ESG_RISK_PATTERNS = [
    r"淨零",
    r"碳中和",
    r"減碳",
    r"零排放",
    r"再生能源",
    r"供應鏈",
    r"人權",
    r"童工",
    r"強迫勞動",
    r"職安",
    r"董事會",
    r"反貪腐",
    r"法遵",
    r"揭露",
]

TIME_BOUND_PATTERNS = [
    r"\d{4}\s*年",
    r"\d+\s*(?:個月|年|天)",
    r"年底",
    r"短期",
    r"中期",
    r"長期",
]


def flag_high_risk_commitment(sentence: str, model_risk_score: float = 0.0) -> tuple[bool, str]:
    """Flag sentences that combine ESG topics with concrete commitments."""
    commitment_hits = _matches(sentence, COMMITMENT_PATTERNS)
    esg_hits = _matches(sentence, ESG_RISK_PATTERNS)
    time_hits = _matches(sentence, TIME_BOUND_PATTERNS)

    reasons: list[str] = []
    if commitment_hits:
        reasons.append(f"承諾語氣: {', '.join(commitment_hits[:3])}")
    if esg_hits:
        reasons.append(f"ESG 議題: {', '.join(esg_hits[:3])}")
    if time_hits:
        reasons.append(f"時程/量化: {', '.join(time_hits[:3])}")
    if model_risk_score >= 0.72:
        reasons.append(f"模型風險分數 {model_risk_score:.2f}")

    is_high_risk = bool(commitment_hits and esg_hits and (time_hits or model_risk_score >= 0.60))
    is_high_risk = is_high_risk or model_risk_score >= 0.82

    return is_high_risk, "；".join(reasons) if reasons else ""


def _matches(text: str, patterns: list[str]) -> list[str]:
    hits: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            hits.append(match.group(0))
    return hits

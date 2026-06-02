from __future__ import annotations

import re


COMMITMENT_PATTERNS = [
    r"承諾",
    r"保證",
    r"確保",
    r"必(?:須|定)",
    r"將(?:於|在|會|持續|逐步|開始|導入|推動|完成|達成)?",
    r"預計",
    r"預期",
    r"預定",
    r"目標",
    r"規劃",
    r"計畫",
    r"致力於",
    r"力求",
    r"持續推動",
    r"aims?\s+to",
    r"plans?\s+to",
    r"targets?\s+to",
    r"commits?\s+to",
    r"will\s+(?:achieve|reduce|increase|improve|implement|establish)",
]

ESG_RISK_PATTERNS = [
    r"ESG",
    r"永續",
    r"永續發展",
    r"企業社會責任",
    r"氣候",
    r"氣候變遷",
    r"溫室氣體",
    r"碳排",
    r"碳排放",
    r"減碳",
    r"淨零",
    r"碳中和",
    r"再生能源",
    r"綠電",
    r"能源",
    r"水資源",
    r"廢棄物",
    r"污染",
    r"循環經濟",
    r"人權",
    r"勞工",
    r"職業安全",
    r"職安",
    r"供應鏈",
    r"公司治理",
    r"董事會",
    r"風險管理",
    r"法遵",
    r"誠信經營",
    r"反貪腐",
    r"sustainab(?:le|ility)",
    r"net\s*zero",
    r"carbon\s+neutral",
    r"greenhouse\s+gas",
    r"GHG",
    r"renewable\s+energy",
    r"human\s+rights",
    r"corporate\s+governance",
]

EVIDENCE_PATTERNS = [
    r"完成",
    r"達成",
    r"實現",
    r"取得",
    r"通過",
    r"驗證",
    r"認證",
    r"查證",
    r"確信",
    r"第三方",
    r"審核",
    r"稽核",
    r"盤查",
    r"揭露",
    r"ISO\s*\d+",
    r"SGS",
    r"BSI",
    r"DNV",
    r"verified",
    r"certified",
    r"assurance",
    r"audited",
]

TIME_BOUND_PATTERNS = [
    r"\d{4}\s*年",
    r"\d{1,2}\s*月",
    r"\d{1,2}\s*日",
    r"\b20\d{2}\b",
    r"\b19\d{2}\b",
    r"\d+(?:\.\d+)?\s*%",
    r"\d+(?:\.\d+)?\s*(?:噸|公噸|tCO2e|CO2e|度|kWh|MWh|GWh|人|件|家|次)",
    r"短期",
    r"中期",
    r"長期",
    r"年度",
    r"每年",
    r"年底",
]


def flag_high_risk_commitment(sentence: str, model_risk_score: float = 0.0) -> tuple[bool, str]:
    """Flag ESG statements that make promises needing evidence or timeline review."""
    commitment_hits = _matches(sentence, COMMITMENT_PATTERNS)
    esg_hits = _matches(sentence, ESG_RISK_PATTERNS)
    evidence_hits = _matches(sentence, EVIDENCE_PATTERNS)
    time_hits = _matches(sentence, TIME_BOUND_PATTERNS)

    reasons: list[str] = []
    if commitment_hits:
        reasons.append(f"承諾語氣: {', '.join(commitment_hits[:3])}")
    if esg_hits:
        reasons.append(f"ESG 主題: {', '.join(esg_hits[:3])}")
    if time_hits:
        reasons.append(f"數字/時程: {', '.join(time_hits[:3])}")
    if evidence_hits:
        reasons.append(f"證據線索: {', '.join(evidence_hits[:3])}")
    if model_risk_score >= 0.72:
        reasons.append(f"模型風險分數 {model_risk_score:.2f}")

    has_commitment_risk = bool(commitment_hits and esg_hits and (time_hits or model_risk_score >= 0.60))
    has_unsupported_commitment = bool(commitment_hits and esg_hits and not evidence_hits and model_risk_score >= 0.55)
    is_high_risk = has_commitment_risk or has_unsupported_commitment or model_risk_score >= 0.82

    return is_high_risk, "；".join(reasons) if reasons else ""


def _matches(text: str, patterns: list[str]) -> list[str]:
    hits: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            hits.append(match.group(0))
    return hits

from __future__ import annotations

from .config import COMPANY_ALIASES, FIXED_ESG_TOPICS, NORMALIZED_TOPIC_KEYWORDS, PEER_GROUPS


def detect_company(file_name: str, text: str) -> str:
    haystack = f"{file_name}\n{text[:5000]}".lower()
    for company, aliases in COMPANY_ALIASES.items():
        if any(alias.lower() in haystack for alias in aliases):
            return company
    return "unknown"


def get_peer_group(company: str) -> set[str]:
    normalized_company = str(company).lower()
    for peer_companies in PEER_GROUPS.values():
        if normalized_company in peer_companies:
            return peer_companies
    return set()


def detect_topic(sentence: str, category: str) -> str | None:
    topic_scores: dict[str, int] = {}
    lowered = sentence.lower()
    for topic, keywords in NORMALIZED_TOPIC_KEYWORDS.get(category, {}).items():
        topic_scores[topic] = sum(1 for keyword in keywords if keyword in lowered)

    best_topic, best_score = max(topic_scores.items(), key=lambda item: item[1], default=("", 0))
    return best_topic if best_score > 0 and best_topic in FIXED_ESG_TOPICS else None

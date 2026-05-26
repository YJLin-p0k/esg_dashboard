from __future__ import annotations

import re
from dataclasses import dataclass


ESG_CATEGORIES = ["Environment", "Social", "Governance", "Other"]

OFFICIAL_WEIGHTS = {
    "promise_status": 0.10,
    "evidence_status": 0.35,
    "evidence_quality": 0.35,
    "verification_timeline": 0.20,
}

LABEL_DEFINITIONS = {
    "promise_status": ["Yes", "No"],
    "evidence_status": ["Yes", "No", "N/A"],
    "verification_timeline": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "longer_than_5_years",
        "N/A",
    ],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}

CATEGORY_KEYWORDS = {
    "Environment": [
        "environment", "emission", "emissions", "carbon", "net zero", "renewable", "energy",
        "waste", "water", "climate", "ghg", "greenhouse", "recycle", "pollution", "biodiversity",
        "環境", "排放", "碳", "減碳", "淨零", "再生能源", "能源", "廢棄物", "水資源",
        "氣候", "溫室氣體", "回收", "污染", "生物多樣性",
    ],
    "Social": [
        "employee", "employees", "health", "safety", "human rights", "diversity", "inclusion",
        "training", "community", "labor", "supply chain", "supplier", "員工", "職安", "安全",
        "人權", "多元", "包容", "訓練", "社區", "勞工", "供應鏈", "供應商",
    ],
    "Governance": [
        "governance", "board", "director", "ethics", "compliance", "anti-corruption",
        "anticorruption", "audit", "risk management", "privacy", "security", "治理", "董事",
        "誠信", "法遵", "反貪腐", "稽核", "風險管理", "隱私", "資安",
    ],
}

PROMISE_KEYWORDS = [
    "target", "goal", "commit", "commitment", "pledge", "aim", "plan", "will", "by 20",
    "achieve", "reduce", "increase", "目標", "承諾", "計畫", "預計", "將於", "達成",
    "完成", "降低", "提升", "持續", "推動",
]

EVIDENCE_KEYWORDS = [
    "certified", "verified", "audited", "assurance", "iso", "scope 1", "scope 2", "scope 3",
    "kpi", "third-party", "third party", "認證", "驗證", "查證", "稽核", "第三方",
    "指標", "揭露", "報告", "數據", "資料",
]

MISLEADING_KEYWORDS = [
    "world-class", "best-in-class", "leading", "green", "sustainable", "eco-friendly",
    "世界級", "最佳", "領先", "綠色", "永續", "環保",
]


@dataclass(frozen=True)
class HybridPrediction:
    esg_category: str
    confidence: float
    promise_status: str
    verification_timeline: str
    evidence_status: str
    evidence_quality: str
    overall_trust_score: float


class HybridESGAnalyzer:
    """Offline inference layer adapted from the notebook's Hybrid v4 contract.

    The notebook combines a RoBERTa model for Task 1/3 and GPT/RAG for Task 2/4.
    This class preserves the same output schema with deterministic rules, so the
    dashboard can run locally today and later swap in trained checkpoints/API calls.
    """

    def predict(self, sentences: list[str]) -> list[HybridPrediction]:
        return [self.predict_one(sentence) for sentence in sentences]

    def predict_one(self, sentence: str) -> HybridPrediction:
        category, confidence = self._classify_category(sentence)
        promise_status = self._predict_promise(sentence)
        evidence_status = self._predict_evidence_status(sentence, promise_status)
        verification_timeline = self._predict_timeline(sentence, promise_status)
        evidence_quality = self._predict_evidence_quality(
            sentence=sentence,
            promise_status=promise_status,
            evidence_status=evidence_status,
            timeline=verification_timeline,
        )
        trust_score = self._score(
            promise_status=promise_status,
            evidence_status=evidence_status,
            verification_timeline=verification_timeline,
            evidence_quality=evidence_quality,
            confidence=confidence,
        )

        return HybridPrediction(
            esg_category=category,
            confidence=confidence,
            promise_status=promise_status,
            verification_timeline=verification_timeline,
            evidence_status=evidence_status,
            evidence_quality=evidence_quality,
            overall_trust_score=trust_score,
        )

    def _classify_category(self, sentence: str) -> tuple[str, float]:
        scores = {
            category: _keyword_hits(sentence, keywords)
            for category, keywords in CATEGORY_KEYWORDS.items()
        }
        category, hits = max(scores.items(), key=lambda item: item[1])
        if hits == 0:
            return "Other", 0.45

        total_hits = sum(scores.values())
        confidence = min(0.98, 0.52 + 0.12 * hits + 0.04 * total_hits)
        return category, round(confidence, 4)

    def _predict_promise(self, sentence: str) -> str:
        has_future_year = any(int(year) >= 2026 for year in _years(sentence))
        return "Yes" if _contains_any(sentence, PROMISE_KEYWORDS) or has_future_year else "No"

    def _predict_evidence_status(self, sentence: str, promise_status: str) -> str:
        has_quantified_data = bool(re.search(r"\d+(?:\.\d+)?\s*(?:%|t|tons?|tonnes?|kwh|mwh|gwh|人|件|次)", sentence, re.I))
        has_evidence_keyword = _contains_any(sentence, EVIDENCE_KEYWORDS)
        has_year = bool(_years(sentence))

        if has_quantified_data or has_evidence_keyword:
            return "Yes"
        if promise_status == "No" and not has_year:
            return "N/A"
        return "No"

    def _predict_timeline(self, sentence: str, promise_status: str) -> str:
        if promise_status == "No":
            return "N/A"

        years = _years(sentence)
        if not years:
            return "N/A"

        current_year = 2026
        nearest = min(years, key=lambda year: abs(year - current_year))
        if nearest <= current_year:
            return "already"
        if nearest <= current_year + 2:
            return "within_2_years"
        if nearest <= current_year + 5:
            return "between_2_and_5_years"
        return "longer_than_5_years"

    def _predict_evidence_quality(
        self,
        sentence: str,
        promise_status: str,
        evidence_status: str,
        timeline: str,
    ) -> str:
        if evidence_status == "N/A":
            return "N/A"
        if evidence_status == "Yes":
            return "Clear"
        if promise_status == "Yes" and timeline == "longer_than_5_years":
            return "Not Clear"
        if promise_status == "Yes" and _contains_any(sentence, MISLEADING_KEYWORDS):
            return "Misleading"
        return "Not Clear"

    def _score(
        self,
        promise_status: str,
        evidence_status: str,
        verification_timeline: str,
        evidence_quality: str,
        confidence: float,
    ) -> float:
        promise_score = 1.0 if promise_status == "Yes" else 0.55
        evidence_score = {"Yes": 1.0, "No": 0.25, "N/A": 0.55}[evidence_status]
        quality_score = {"Clear": 1.0, "Not Clear": 0.35, "Misleading": 0.0, "N/A": 0.55}[evidence_quality]
        timeline_score = {
            "already": 1.0,
            "within_2_years": 0.85,
            "between_2_and_5_years": 0.65,
            "longer_than_5_years": 0.35,
            "N/A": 0.55,
        }[verification_timeline]

        weighted = (
            promise_score * OFFICIAL_WEIGHTS["promise_status"]
            + evidence_score * OFFICIAL_WEIGHTS["evidence_status"]
            + quality_score * OFFICIAL_WEIGHTS["evidence_quality"]
            + timeline_score * OFFICIAL_WEIGHTS["verification_timeline"]
        )
        trust_score = round(max(0, min(100, weighted * 100)), 2)

        return trust_score


def _keyword_hits(sentence: str, keywords: list[str]) -> int:
    lowered = sentence.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def _contains_any(sentence: str, keywords: list[str]) -> bool:
    return _keyword_hits(sentence, keywords) > 0


def _years(sentence: str) -> list[int]:
    return [int(match) for match in re.findall(r"\b20[2-5][0-9]\b", sentence)]

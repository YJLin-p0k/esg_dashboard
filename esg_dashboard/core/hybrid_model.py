from __future__ import annotations

import json
import re
from dataclasses import dataclass

from esg_dashboard.core.final_gpt_rag import FinalGPTRAGPredictor
from esg_dashboard.core.final_rag_assets import FinalRAGAssets
from esg_dashboard.core.final_roberta_model import (
    FinalRoBERTaEnsemblePredictor,
    FinalRoBERTaUnavailable,
)


ESG_CATEGORIES = ["Environment", "Social", "Governance", "Other"]

OFFICIAL_WEIGHTS = {
    "promise_status": 0.20,
    "verification_timeline": 0.15,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
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
        "環境", "氣候", "氣候變遷", "溫室氣體", "碳排", "碳排放", "減碳", "淨零", "碳中和",
        "再生能源", "綠電", "能源", "水資源", "廢棄物", "污染", "循環經濟", "生物多樣性",
    ],
    "Social": [
        "employee", "employees", "health", "safety", "human rights", "diversity", "inclusion",
        "training", "community", "labor", "supply chain", "supplier",
        "社會", "員工", "勞工", "人權", "職安", "職業安全", "安全衛生", "健康", "多元",
        "平等", "包容", "訓練", "社區", "公益", "供應鏈", "供應商",
    ],
    "Governance": [
        "governance", "board", "director", "ethics", "compliance", "anti-corruption",
        "anticorruption", "audit", "risk management", "privacy", "security",
        "治理", "公司治理", "董事", "董事會", "獨立董事", "誠信", "法遵", "合規",
        "反貪腐", "稽核", "內控", "風險管理", "資訊安全", "資安", "隱私",
    ],
}

PROMISE_KEYWORDS = [
    "target", "goal", "commit", "commitment", "pledge", "aim", "plan", "will", "by 20",
    "achieve", "reduce", "increase", "承諾", "保證", "確保", "將", "預計", "預期",
    "預定", "目標", "規劃", "計畫", "致力於", "力求", "持續推動",
]

EVIDENCE_KEYWORDS = [
    "certified", "verified", "audited", "assurance", "iso", "scope 1", "scope 2", "scope 3",
    "kpi", "third-party", "third party", "完成", "達成", "實現", "取得", "通過",
    "驗證", "認證", "查證", "確信", "第三方", "審核", "稽核", "盤查", "揭露",
]

MISLEADING_KEYWORDS = [
    "world-class", "best-in-class", "leading", "green", "sustainable", "eco-friendly",
    "世界級", "最佳", "領先", "環保", "綠色", "永續", "友善環境",
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


def normalize_task_outputs(
    promise_status: str,
    evidence_status: str,
    verification_timeline: str,
    evidence_quality: str,
) -> tuple[str, str, str, str]:
    promise_status = promise_status if promise_status in LABEL_DEFINITIONS["promise_status"] else "No"
    evidence_status = evidence_status if evidence_status in LABEL_DEFINITIONS["evidence_status"] else "N/A"
    verification_timeline = (
        verification_timeline
        if verification_timeline in LABEL_DEFINITIONS["verification_timeline"]
        else "N/A"
    )
    evidence_quality = evidence_quality if evidence_quality in LABEL_DEFINITIONS["evidence_quality"] else "N/A"

    if promise_status == "No":
        evidence_status = "N/A"
        verification_timeline = "N/A"
        evidence_quality = "N/A"
    elif evidence_status in {"No", "N/A"}:
        evidence_quality = "N/A"

    return promise_status, evidence_status, verification_timeline, evidence_quality


class HybridESGAnalyzer:
    """Final notebook inference layer with a local fallback.

    If Colab-exported RoBERTa checkpoints are present under
    models/final_roberta_task13_A_stable_baseline, task 1 and task 3 use the
    final notebook's soft-voting RoBERTa ensemble. Task 2 and task 4 keep local
    deterministic inference plus the notebook's final post-processing rules.
    """

    def __init__(self) -> None:
        self.roberta = FinalRoBERTaEnsemblePredictor()
        self.rag_assets = FinalRAGAssets()
        self.gpt_rag = FinalGPTRAGPredictor(self.rag_assets)

    def predict(self, sentences: list[str]) -> list[HybridPrediction]:
        roberta_rows = self._predict_roberta_tasks(sentences)
        return [
            self.predict_one(sentence, roberta_row=roberta_row)
            for sentence, roberta_row in zip(sentences, roberta_rows)
        ]

    def predict_one(self, sentence: str, roberta_row: dict[str, object] | None = None) -> HybridPrediction:
        category, confidence = self._classify_category(sentence)
        if roberta_row:
            promise_status = str(roberta_row.get("promise_status", self._predict_promise(sentence)))
            evidence_status = str(
                roberta_row.get("evidence_status", self._predict_evidence_status(sentence, promise_status))
            )
            confidence = min(confidence, float(roberta_row.get("confidence", confidence)))
        else:
            promise_status = self._predict_promise(sentence)
            evidence_status = self._predict_evidence_status(sentence, promise_status)

        verification_timeline = self._predict_timeline(sentence, promise_status)
        evidence_quality = self._predict_evidence_quality(
            sentence=sentence,
            promise_status=promise_status,
            evidence_status=evidence_status,
            timeline=verification_timeline,
        )
        rag_row = self._predict_gpt_rag_tasks(sentence, roberta_row) or self._predict_rag_tasks(sentence)
        if rag_row:
            verification_timeline = str(rag_row.get("verification_timeline", verification_timeline))
            evidence_quality = str(rag_row.get("evidence_quality", evidence_quality))

        promise_status, evidence_status, verification_timeline, evidence_quality = self._postprocess_final_tasks(
            promise_status=promise_status,
            evidence_status=evidence_status,
            verification_timeline=verification_timeline,
            evidence_quality=evidence_quality,
        )
        trust_score = self._score(
            promise_status=promise_status,
            evidence_status=evidence_status,
            verification_timeline=verification_timeline,
            evidence_quality=evidence_quality,
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

    def _predict_roberta_tasks(self, sentences: list[str]) -> list[dict[str, object] | None]:
        if not self.roberta.is_available:
            return [None] * len(sentences)
        try:
            return self.roberta.predict(sentences)
        except FinalRoBERTaUnavailable:
            return [None] * len(sentences)

    def _predict_rag_tasks(self, sentence: str) -> dict[str, object] | None:
        try:
            return self.rag_assets.predict_task24(sentence)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _predict_gpt_rag_tasks(
        self,
        sentence: str,
        roberta_row: dict[str, object] | None,
    ) -> dict[str, object] | None:
        try:
            return self.gpt_rag.predict_task24(sentence, roberta_row=roberta_row)
        except Exception:
            return None

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
        has_future_year = any(year >= 2026 for year in _years(sentence))
        return "Yes" if _contains_any(sentence, PROMISE_KEYWORDS) or has_future_year else "No"

    def _predict_evidence_status(self, sentence: str, promise_status: str) -> str:
        has_quantified_data = bool(
            re.search(
                r"\d+(?:\.\d+)?\s*(?:%|％|t|tons?|tonnes?|kwh|mwh|gwh|噸|公噸|人|件|家|次)",
                sentence,
                re.I,
            )
        )
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

    def _postprocess_final_tasks(
        self,
        promise_status: str,
        evidence_status: str,
        verification_timeline: str,
        evidence_quality: str,
    ) -> tuple[str, str, str, str]:
        return normalize_task_outputs(
            promise_status=promise_status,
            evidence_status=evidence_status,
            verification_timeline=verification_timeline,
            evidence_quality=evidence_quality,
        )

    def _score(
        self,
        promise_status: str,
        evidence_status: str,
        verification_timeline: str,
        evidence_quality: str,
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
        return round(max(0, min(100, weighted * 100)), 2)


def _keyword_hits(sentence: str, keywords: list[str]) -> int:
    lowered = sentence.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def _contains_any(sentence: str, keywords: list[str]) -> bool:
    return _keyword_hits(sentence, keywords) > 0


def _years(sentence: str) -> list[int]:
    return [int(match) for match in re.findall(r"\b20[2-5][0-9]\b", sentence)]

from __future__ import annotations

import json
import os
from typing import Any

from esg_dashboard.core.final_rag_assets import FinalRAGAssets


DEFAULT_GPT_MODEL = "gpt-5"
DEFAULT_TOP_K = 6

TIMELINE_LABELS = {
    "already",
    "within_2_years",
    "between_2_and_5_years",
    "longer_than_5_years",
    "N/A",
}

QUALITY_LABELS = {
    "Clear",
    "Not Clear",
    "Misleading",
    "N/A",
}


class FinalGPTRAGPredictor:
    """GPT+RAG predictor for the final notebook's task 2 and task 4."""

    def __init__(
        self,
        rag_assets: FinalRAGAssets,
        model: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.rag_assets = rag_assets
        self.model = model or _secret_value("OPENAI_MODEL") or DEFAULT_GPT_MODEL
        self.top_k = top_k

    @property
    def is_available(self) -> bool:
        return bool(_openai_api_key()) and self.rag_assets.is_available

    def predict_task24(
        self,
        sentence: str,
        roberta_row: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if not self.is_available:
            return None

        examples = self.rag_assets.retrieve_top_k(sentence, top_k=self.top_k)
        if not examples:
            return None

        timeline = self._predict_timeline(sentence, examples, roberta_row)
        quality = self._predict_quality(sentence, examples, roberta_row)
        if timeline is None and quality is None:
            return None

        return {
            "verification_timeline": timeline or "N/A",
            "evidence_quality": quality or "N/A",
            "gpt_model": self.model,
            "top_k": self.top_k,
        }

    def _predict_timeline(
        self,
        sentence: str,
        examples: list[dict[str, object]],
        roberta_row: dict[str, object] | None,
    ) -> str | None:
        parsed = self._call_json(_build_timeline_prompt(sentence, examples, roberta_row))
        if not parsed:
            return None
        return _normalize_label(str(parsed.get("verification_timeline", "N/A")), TIMELINE_LABELS, "N/A")

    def _predict_quality(
        self,
        sentence: str,
        examples: list[dict[str, object]],
        roberta_row: dict[str, object] | None,
    ) -> str | None:
        parsed = self._call_json(_build_quality_prompt(sentence, examples, roberta_row))
        if not parsed:
            return None
        return _normalize_label(str(parsed.get("evidence_quality", "N/A")), QUALITY_LABELS, "N/A")

    def _call_json(self, prompt: str) -> dict[str, Any] | None:
        try:
            from openai import OpenAI
        except ImportError:
            return None

        api_key = _openai_api_key()
        if not api_key:
            return None

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an ESG promise verification classifier. Return valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def _openai_api_key() -> str | None:
    return _secret_value("OPENAI_API_KEY")


def _secret_value(name: str) -> str | None:
    env_value = os.getenv(name)
    if env_value:
        return env_value
    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        return str(secret_value) if secret_value else None
    except Exception:
        return None


def _build_timeline_prompt(
    sentence: str,
    examples: list[dict[str, object]],
    roberta_row: dict[str, object] | None,
) -> str:
    return f"""
請根據測試文本、RoBERTa 輔助預測、標籤定義與相似案例，只判斷 verification_timeline。
請只輸出 JSON，例如 {{"verification_timeline": "already"}}。

可選標籤：
- already：承諾或行動已經完成、已取得、已執行或已有結果。
- within_2_years：承諾時程在未來 2 年內。
- between_2_and_5_years：承諾時程在未來 2 到 5 年。
- longer_than_5_years：承諾時程超過未來 5 年。
- N/A：沒有 ESG 承諾或沒有可判斷時程。

RoBERTa 輔助預測：
{_format_roberta(roberta_row)}

相似案例：
{_format_examples(examples, ["verification_timeline"])}

測試文本：
{sentence}
""".strip()


def _build_quality_prompt(
    sentence: str,
    examples: list[dict[str, object]],
    roberta_row: dict[str, object] | None,
) -> str:
    return f"""
請根據測試文本、RoBERTa 輔助預測、標籤定義與相似案例，只判斷 evidence_quality。
請只輸出 JSON，例如 {{"evidence_quality": "Clear"}}。

可選標籤：
- Clear：有清楚、可驗證、具體或量化的證據。
- Not Clear：有證據或宣稱，但證據不夠清楚、具體或可驗證。
- Misleading：文字可能誤導，或以模糊正面詞彙包裝但缺乏支撐。
- N/A：文本不適用 evidence_quality 判斷；尤其沒有 ESG 承諾、沒有 evidence_status=Yes 的情況。

重要規則：
- 如果 RoBERTa 判斷 promise_status=No，通常 evidence_quality 應為 N/A。
- 如果 RoBERTa 判斷 evidence_status=No 或 N/A，通常 evidence_quality 應為 N/A。

RoBERTa 輔助預測：
{_format_roberta(roberta_row)}

相似案例：
{_format_examples(examples, ["evidence_quality"])}

測試文本：
{sentence}
""".strip()


def _format_roberta(roberta_row: dict[str, object] | None) -> str:
    if not roberta_row:
        return "N/A"
    return json.dumps(
        {
            "promise_status": roberta_row.get("promise_status"),
            "evidence_status": roberta_row.get("evidence_status"),
        },
        ensure_ascii=False,
    )


def _format_examples(examples: list[dict[str, object]], labels: list[str]) -> str:
    rows: list[str] = []
    for index, example in enumerate(examples, start=1):
        label_text = ", ".join(f"{label}={example.get(label, 'N/A')}" for label in labels)
        rows.append(
            f"[Example {index}] similarity={example.get('similarity', 'N/A')} | {label_text}\n"
            f"text={example.get('retrieval_text') or example.get('data') or ''}"
        )
    return "\n\n".join(rows)


def _normalize_label(value: str, allowed: set[str], default: str) -> str:
    normalized = value.strip()
    aliases = {
        "na": "N/A",
        "n/a": "N/A",
        "not applicable": "N/A",
        "not_clear": "Not Clear",
        "not clear": "Not Clear",
        "misleading": "Misleading",
        "clear": "Clear",
    }
    normalized = aliases.get(normalized.lower(), normalized)
    return normalized if normalized in allowed else default

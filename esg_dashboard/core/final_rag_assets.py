from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path("models") / "final_roberta_task13_A_stable_baseline"
DEFAULT_POOL_PATH = DEFAULT_MODEL_DIR / "final_retrieval_pool.json"
DEFAULT_EMBEDDING_PATH = DEFAULT_MODEL_DIR / "final_retrieval_embeddings.npy"


class FinalRAGAssets:
    """Local access to final notebook retrieval assets.

    The original final notebook uses these examples with GPT+RAG. In the app we
    can still use the pool as a conservative local fallback when a new sentence
    is lexically close to a training example.
    """

    def __init__(
        self,
        pool_path: str | Path = DEFAULT_POOL_PATH,
        embedding_path: str | Path = DEFAULT_EMBEDDING_PATH,
        min_similarity: float = 0.22,
    ) -> None:
        self.pool_path = Path(pool_path)
        self.embedding_path = Path(embedding_path)
        self.min_similarity = min_similarity
        self._loaded = False
        self._examples: list[dict[str, Any]] = []
        self._example_tokens: list[Counter[str]] = []

    @property
    def is_available(self) -> bool:
        return self.pool_path.exists() and self.embedding_path.exists()

    def predict_task24(self, sentence: str) -> dict[str, object] | None:
        retrieved = self.retrieve_top_k(sentence, top_k=1)
        if not retrieved:
            return None
        best = retrieved[0]
        if float(best.get("similarity", 0.0)) < self.min_similarity:
            return None

        return {
            "verification_timeline": best.get("verification_timeline", "N/A"),
            "evidence_quality": best.get("evidence_quality", "N/A"),
            "rag_similarity": best.get("similarity", 0.0),
            "rag_example_id": best.get("id"),
        }

    def retrieve_top_k(self, sentence: str, top_k: int = 6) -> list[dict[str, object]]:
        if not self.is_available:
            return []
        self._load()
        query_tokens = _token_counter(sentence)
        if not query_tokens:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for example, example_tokens in zip(self._examples, self._example_tokens):
            score = _weighted_jaccard(query_tokens, example_tokens)
            scored.append((score, example))

        scored.sort(key=lambda item: item[0], reverse=True)
        rows: list[dict[str, object]] = []
        for score, example in scored[:top_k]:
            row = dict(example)
            row["similarity"] = round(score, 4)
            rows.append(row)
        return rows

    def _load(self) -> None:
        if self._loaded:
            return
        data = json.loads(self.pool_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"RAG pool must be a list: {self.pool_path}")
        self._examples = [example for example in data if isinstance(example, dict)]
        self._example_tokens = [
            _token_counter(str(example.get("retrieval_text") or example.get("data") or ""))
            for example in self._examples
        ]
        self._loaded = True


def _token_counter(text: str) -> Counter[str]:
    normalized = text.lower()
    latin_tokens = re.findall(r"[a-z0-9][a-z0-9_\-/.%]*", normalized)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    cjk_bigrams = [
        "".join(pair)
        for pair in zip(cjk_chars, cjk_chars[1:])
    ]
    return Counter(latin_tokens + cjk_bigrams)


def _weighted_jaccard(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    intersection = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    return 0.0 if union == 0 else intersection / union

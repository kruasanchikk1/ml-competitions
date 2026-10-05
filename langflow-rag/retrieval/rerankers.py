"""
Реранкер: второй этап двухэтапного поиска.

top-100 из векторного поиска (дёшево, грубо) -> реранкер пересчитывает
релевантность пары (запрос, документ) точнее и сужает до top-3/5.

Как и с эмбеддерами, реальная модель (BGE reranker, кросс-энкодер) и
офлайн-заглушка (лексическое пересечение слов) спрятаны за одним
интерфейсом Reranker, чтобы пайплайн можно было тестировать без
Hugging Face и без изменений в остальном коде.
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, documents: list[str]) -> list[float]: ...

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        scores = self.score(query, [d["text"] for d in documents])
        for d, s in zip(documents, scores):
            d["rerank_score"] = s
        return sorted(documents, key=lambda d: d["rerank_score"], reverse=True)[:top_k]


class BGERerankerCrossEncoder(Reranker):
    """BAAI/bge-reranker-v2-m3 через sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.name = "bge-reranker-v2-m3"

    def score(self, query: str, documents: list[str]) -> list[float]:
        pairs = [(query, doc) for doc in documents]
        return [float(s) for s in self.model.predict(pairs)]


class LexicalOverlapReranker(Reranker):
    """
    Офлайн-заглушка: скор = доля слов запроса, встретившихся в документе,
    с лёгким бонусом за точные фразы. Не кросс-энкодер, но детерминирован
    и достаточен, чтобы протестировать место реранкера в пайплайне и
    измерить *эффект* двухэтапного поиска на синтетических вопросах.
    """

    def __init__(self):
        self.name = "lexical-overlap-dev"

    def score(self, query: str, documents: list[str]) -> list[float]:
        q_words = set(w.lower() for w in _WORD_RE.findall(query) if len(w) > 2)
        scores = []
        for doc in documents:
            d_words = set(w.lower() for w in _WORD_RE.findall(doc) if len(w) > 2)
            if not q_words:
                scores.append(0.0)
                continue
            overlap = len(q_words & d_words) / len(q_words)
            phrase_bonus = 0.1 if query.lower()[:20] in doc.lower() else 0.0
            scores.append(overlap + phrase_bonus)
        return scores


def get_reranker(backend: str | None = None) -> Reranker:
    backend = (backend or os.getenv("RERANK_BACKEND", "hash")).lower()
    if backend in ("bge", "bge-reranker", "cross-encoder"):
        return BGERerankerCrossEncoder()
    if backend in ("hash", "dev", "offline", "lexical"):
        return LexicalOverlapReranker()
    raise ValueError(f"Неизвестный backend реранкера: {backend!r}")

"""
Единый интерфейс эмбеддера + несколько реализаций.

Почему так: сравнение BGE-M3 vs Qwen3-Embedding — это буквально то, что
хочется показать на собеседовании как side-by-side, а не декларировать
на словах. Чтобы это было реальным кодом, а не единственным жёстко
зашитым вызовом одной модели, эмбеддер спрятан за интерфейсом Embedder,
и обе модели подключаются через один и тот же sentence-transformers API
(они обе поддерживают Sentence Transformers из коробки).

HashEmbedder — не про качество поиска, а про CI/офлайн-разработку:
в песочнице, где собирался этот репозиторий, нет доступа к Hugging Face
(сетевая политика блокирует huggingface.co), поэтому весь пайплайн ниже
написан и протестирован end-to-end на HashEmbedder — детерминированном
мешке слов с хэш-трюком (bag-of-words + feature hashing), который даёт
*настоящую* косинусную близость по общим словам, но не является
семантическим эмбеддингом. Это позволяет проверить механику (чанкинг →
индексация → поиск → реранкинг → eval), не выкачивая веса. При запуске
на машине с интернетом просто переключите EMBEDDING_BACKEND=bge-m3 (или
qwen3) в .env — код не меняется.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod


class Embedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class SentenceTransformerEmbedder(Embedder):
    """Обёртка над sentence-transformers для реальных моделей."""

    def __init__(self, model_name: str, name: str, dim: int, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer  # локальный импорт

        self.model = SentenceTransformer(model_name)
        self.name = name
        self.dim = dim
        self.query_prefix = query_prefix

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([self.query_prefix + text])[0]


def BGEM3Embedder() -> Embedder:
    # BGE-M3: открытая, мультиязычная (включая русский), разворачивается
    # локально — обоснование выбора для закрытого контура банка, см. README.
    return SentenceTransformerEmbedder("BAAI/bge-m3", name="bge-m3", dim=1024)


def Qwen3Embedder() -> Embedder:
    # Qwen3-Embedding-0.6B: более новая мультиязычная модель (Alibaba),
    # сопоставимый размер с BGE-M3, часто выше в MTEB-мультиязычных
    # бенчмарках на момент написания — альтернатива для side-by-side.
    return SentenceTransformerEmbedder(
        "Qwen/Qwen3-Embedding-0.6B",
        name="qwen3-embedding-0.6b",
        dim=1024,
        query_prefix="Instruct: Given a query, retrieve relevant passages\nQuery: ",
    )


_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "для", "об", "также", "иных", "иные", "настоящего", "настоящем",
}


class HashEmbedder(Embedder):
    """
    Детерминированный bag-of-words эмбеддер на хэш-трюке.

    НЕ семантический (синонимы не сближаются), но даёт настоящую
    косинусную близость по пересечению слов — достаточно, чтобы
    end-to-end проверить пайплайн офлайн, без обращения к Hugging Face.
    """

    def __init__(self, dim: int = 384):
        self.name = "hash-bow-dev"
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = [w.lower() for w in _WORD_RE.findall(text)]
        words = [w for w in words if w not in _STOPWORDS and len(w) > 2]
        if not words:
            return vec
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def get_embedder(backend: str | None = None) -> Embedder:
    backend = (backend or os.getenv("EMBEDDING_BACKEND", "hash")).lower()
    if backend in ("bge-m3", "bge", "bgem3"):
        return BGEM3Embedder()
    if backend in ("qwen3", "qwen", "qwen3-embedding"):
        return Qwen3Embedder()
    if backend in ("hash", "dev", "offline"):
        return HashEmbedder()
    raise ValueError(f"Неизвестный backend эмбеддера: {backend!r}")

"""
Двухэтапный поиск:
  1. Фильтрация по метаданным (article/chapter) — ДО векторного поиска,
     если фильтр передан. Это дешевле, чем фильтровать после: незачем
     считать косинусное расстояние с точками, которые заведомо не
     подходят (см. Qdrant Filter в query()).
  2. Векторный поиск даёт top-N кандидатов (по умолчанию top-100) —
     дёшево и грубо.
  3. Реранкер пересчитывает релевантность пары (запрос, документ) точнее
     и сужает до top-K (по умолчанию top-5).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from ingestion.build_index import COLLECTION
from ingestion.embedders import Embedder
from retrieval.rerankers import Reranker


def _build_filter(metadata_filter: dict[str, str] | None) -> Filter | None:
    if not metadata_filter:
        return None
    return Filter(
        must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in metadata_filter.items()]
    )


def two_stage_search(
    client: QdrantClient,
    embedder: Embedder,
    reranker: Reranker,
    query: str,
    top_n_vector: int = 100,
    top_k_final: int = 5,
    metadata_filter: dict[str, str] | None = None,
    use_reranker: bool = True,
) -> list[dict]:
    query_vector = embedder.embed_query(query)

    response = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_n_vector,
        query_filter=_build_filter(metadata_filter),
        with_payload=True,
    )

    candidates = [
        {
            "text": h.payload["text"],
            "article": h.payload.get("article"),
            "source_file": h.payload.get("source_file"),
            "vector_score": h.score,
        }
        for h in response.points
    ]

    if not use_reranker:
        return sorted(candidates, key=lambda c: c["vector_score"], reverse=True)[:top_k_final]

    return reranker.rerank(query, candidates, top_k=top_k_final)


if __name__ == "__main__":
    import argparse

    from ingestion.build_index import build_index
    from ingestion.embedders import get_embedder
    from retrieval.rerankers import get_reranker

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--backend", default="hash")
    parser.add_argument("--rerank-backend", default="hash")
    parser.add_argument("--qdrant", default="memory")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    client, backend_name = build_index(raw_dir, args.backend, args.qdrant)
    embedder = get_embedder(args.backend)
    reranker = get_reranker(args.rerank_backend)

    results = two_stage_search(
        client, embedder, reranker, args.query,
        top_k_final=args.top_k, use_reranker=not args.no_rerank,
    )
    print(f"\nЗапрос: {args.query!r}  (backend={backend_name}, rerank={'off' if args.no_rerank else reranker.name})\n")
    for i, r in enumerate(results, 1):
        score = r.get("rerank_score", r.get("vector_score"))
        print(f"{i}. [{r['article']}] score={score:.3f}")
        print("   " + r["text"][:180].replace("\n", " ") + "...\n")

"""
Строит индекс в Qdrant: чанки → эмбеддинги → upsert с метаданными.

Метаданные (article, chapter, law, source_file) сохраняются в payload
каждой точки Qdrant — это то, по чему retrieval/search.py фильтрует
ДО векторного поиска (дешевле, чем фильтровать после: незачем считать
косинусное расстояние с точками, которые заведомо не подходят по
статье/главе).

Запуск:
    python -m ingestion.build_index --backend hash --qdrant memory
    python -m ingestion.build_index --backend bge-m3 --qdrant http://localhost:6333
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from ingestion.chunk import load_and_chunk_corpus
from ingestion.embedders import get_embedder

COLLECTION = "rag_knowledge_base"


def build_index(
    raw_dir: Path,
    backend: str,
    qdrant_location: str,
    chunk_size: int = 900,
    overlap_size: int = 150,
) -> tuple[QdrantClient, str]:
    embedder = get_embedder(backend)
    chunks = load_and_chunk_corpus(raw_dir, chunk_size=chunk_size, overlap_size=overlap_size)
    if not chunks:
        raise SystemExit(f"В {raw_dir} не найдено ни одного .md документа")

    print(f"[build_index] backend={embedder.name} dim={embedder.dim} chunks={len(chunks)}")

    if qdrant_location == "memory":
        client = QdrantClient(":memory:")
    else:
        client = QdrantClient(url=qdrant_location)

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
    )

    texts = [c.text for c in chunks]
    vectors = embedder.embed(texts)

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={"text": chunks[i].text, **chunks[i].metadata},
        )
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"[build_index] загружено {len(points)} точек в коллекцию '{COLLECTION}'")
    return client, embedder.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=os.getenv("EMBEDDING_BACKEND", "hash"))
    parser.add_argument("--qdrant", default=os.getenv("QDRANT_URL", "memory"))
    parser.add_argument("--raw-dir", default=str(Path(__file__).parent.parent / "data" / "raw"))
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    build_index(Path(args.raw_dir), args.backend, args.qdrant, args.chunk_size, args.overlap)

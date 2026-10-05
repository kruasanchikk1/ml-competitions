"""
Side-by-side сравнение: с реранкингом vs без, на наборе тестовых вопросов
из eval/questions.json.

Это прямая имитация того, что просят в вакансии промпт-инженера:
"side-by-side сравнение ответов моделей на точность/релевантность".
Здесь сравнивается не LLM-ответ, а сам retrieval-этап (что RAG-пайплайн
подкладывает модели в контекст) — потому что именно тут решается,
получит ли модель на входе правильный фрагмент закона или нет.

Метрика: hit@1 — среди top-1 результата встречается ли хотя бы одно
ожидаемое ключевое слово/фраза (`expected_keywords`) из этой статьи.
Грубая метрика (замена нормальному human-eval или сравнению с LLM-judge),
но достаточно, чтобы численно показать эффект реранкинга.

Запуск:
    python -m eval.run_sidebyside --backend hash --rerank-backend hash
    python -m eval.run_sidebyside --backend bge-m3 --rerank-backend bge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.build_index import build_index
from ingestion.embedders import get_embedder
from retrieval.rerankers import get_reranker
from retrieval.search import two_stage_search


def hit(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def run(backend: str, rerank_backend: str, qdrant_location: str) -> None:
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    questions = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))

    client, backend_name = build_index(raw_dir, backend, qdrant_location)
    embedder = get_embedder(backend)
    reranker = get_reranker(rerank_backend)

    rows = []
    for q in questions:
        no_rerank = two_stage_search(
            client, embedder, reranker, q["question"], top_k_final=1, use_reranker=False
        )
        with_rerank = two_stage_search(
            client, embedder, reranker, q["question"], top_k_final=1, use_reranker=True
        )

        no_rerank_hit = bool(no_rerank) and hit(no_rerank[0]["text"], q["expected_keywords"])
        with_rerank_hit = bool(with_rerank) and hit(with_rerank[0]["text"], q["expected_keywords"])

        rows.append({
            "id": q["id"],
            "question": q["question"][:60] + ("..." if len(q["question"]) > 60 else ""),
            "expected_article": q["expected_article"],
            "top1_no_rerank_article": no_rerank[0]["article"] if no_rerank else None,
            "hit_no_rerank": no_rerank_hit,
            "top1_with_rerank_article": with_rerank[0]["article"] if with_rerank else None,
            "hit_with_rerank": with_rerank_hit,
        })

    acc_no_rerank = sum(r["hit_no_rerank"] for r in rows) / len(rows)
    acc_with_rerank = sum(r["hit_with_rerank"] for r in rows) / len(rows)

    try:
        from tabulate import tabulate
        print(tabulate(rows, headers="keys", tablefmt="github"))
    except ImportError:
        for r in rows:
            print(r)

    print(f"\nBackend: эмбеддер={backend_name}, реранкер={reranker.name}")
    print(f"Accuracy@1 без реранкинга:  {acc_no_rerank:.0%}  ({sum(r['hit_no_rerank'] for r in rows)}/{len(rows)})")
    print(f"Accuracy@1 с реранкингом:   {acc_with_rerank:.0%}  ({sum(r['hit_with_rerank'] for r in rows)}/{len(rows)})")

    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(
        json.dumps(
            {
                "backend": backend_name,
                "reranker": reranker.name,
                "accuracy_no_rerank": acc_no_rerank,
                "accuracy_with_rerank": acc_with_rerank,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nРезультаты сохранены в {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="hash")
    parser.add_argument("--rerank-backend", default="hash")
    parser.add_argument("--qdrant", default="memory")
    args = parser.parse_args()
    run(args.backend, args.rerank_backend, args.qdrant)

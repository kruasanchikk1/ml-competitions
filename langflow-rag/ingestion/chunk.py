"""
Чанкинг документов базы знаний с overlap (внахлёст), а не наивной нарезкой
по фиксированной длине.

Стратегия:
  1. Каждый .md-файл в data/raw парсится как YAML frontmatter + тело.
     Frontmatter (закон/глава/статья/источник) становится метаданными
     каждого чанка — на них потом фильтруем ДО векторного поиска.
  2. Тело делится на "логические единицы" — абзацы и пронумерованные части
     статьи (см. _split_into_units), а не режется вслепую по N символов:
     так определение или часть статьи не разрывается посередине.
  3. Единицы жадно упаковываются в чанки размером до `chunk_size` символов.
  4. Каждый следующий чанк начинается с "хвоста" предыдущего
     (`overlap_size` символов) — это и есть overlap: соседние чанки имеют
     общий контекст, поэтому смысл не теряется на границе разреза.

Почему это важно объяснить на собеседовании: наивный фикс-сайз чанкинг
рвёт предложения и определения ровно там, где неудобно — например,
посреди перечисления в статье 3. Overlap + packing по абзацам снижает
вероятность того, что релевantный фрагмент "распадётся" на два чанка,
ни один из которых по отдельности не пройдёт порог релевантности при
поиске.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_markdown_with_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    frontmatter_raw, body = match.groups()
    metadata = yaml.safe_load(frontmatter_raw) or {}
    return metadata, body.strip()


def _split_into_units(body: str) -> list[str]:
    """Делит тело документа на абзацы/пункты, не разрывая их середину."""
    # Разделяем по пустой строке (абзацы markdown), убираем пустые.
    units = [u.strip() for u in re.split(r"\n\s*\n", body) if u.strip()]
    return units


def chunk_document(
    metadata: dict[str, Any],
    body: str,
    chunk_size: int = 900,
    overlap_size: int = 150,
    source_file: str | None = None,
) -> list[Chunk]:
    units = _split_into_units(body)
    chunks: list[Chunk] = []
    current = ""

    def flush(carry_overlap: bool) -> None:
        nonlocal current
        text = current.strip()
        if not text:
            return
        chunk_meta = {
            **metadata,
            "source_file": source_file,
            "chunk_index": len(chunks),
        }
        chunks.append(Chunk(id=str(uuid.uuid4()), text=text, metadata=chunk_meta))
        if carry_overlap:
            current = text[-overlap_size:] + "\n\n"
        else:
            current = ""

    for unit in units:
        # Если один абзац сам по себе длиннее chunk_size (например,
        # статья 3 с одним гигантским перечислением) — не мучаем regex,
        # режем его по предложениям, чтобы не потерять содержимое.
        if len(unit) > chunk_size:
            sentences = re.split(r"(?<=[.;])\s+", unit)
            for sent in sentences:
                if len(current) + len(sent) + 1 > chunk_size and current.strip():
                    flush(carry_overlap=True)
                current += sent + " "
            continue

        if len(current) + len(unit) + 2 > chunk_size and current.strip():
            flush(carry_overlap=True)
        current += unit + "\n\n"

    flush(carry_overlap=False)
    return chunks


def load_and_chunk_corpus(
    raw_dir: Path, chunk_size: int = 900, overlap_size: int = 150
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(raw_dir.glob("*.md")):
        metadata, body = parse_markdown_with_frontmatter(path)
        chunks = chunk_document(
            metadata,
            body,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            source_file=path.name,
        )
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    import sys

    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "data" / "raw"
    chunks = load_and_chunk_corpus(raw_dir)
    print(f"Собрано {len(chunks)} чанков из {len(list(raw_dir.glob('*.md')))} документов\n")
    for c in chunks[:3]:
        print(f"--- chunk {c.metadata['chunk_index']} ({c.metadata.get('article')}) ---")
        print(c.text[:200].replace("\n", " ") + "...")
        print()

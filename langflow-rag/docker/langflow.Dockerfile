FROM langflowai/langflow:latest

# CPU-only torch ставим ОТДЕЛЬНЫМ шагом первым, через индекс PyTorch (не
# PyPI): иначе `pip install sentence-transformers` тянет обычный (GPU)
# torch, а с ним весь CUDA-тулкит — несколько гигабайт лишних закачек,
# из-за которых на нестабильном соединении сборка обрывается по таймауту.
# GPU в контейнере Langflow нет, эмбеддинг одного вопроса при поиске CPU
# считает мгновенно.
RUN /app/.venv/bin/pip install --no-cache-dir --timeout 100 --retries 10 \
    torch --index-url https://download.pytorch.org/whl/cpu

# sentence-transformers — для Custom Component с локальной BGE-M3.
RUN /app/.venv/bin/pip install --no-cache-dir --timeout 100 --retries 10 \
    sentence-transformers

# Компонент Qdrant в образе langflowai/langflow НЕ встроен по умолчанию —
# это отдельный бандл (см. docs.langflow.org/bundles-qdrant), без явной
# установки он просто не появляется в поиске компонентов, без какой-либо
# ошибки в UI (та же природа проблемы, что и с OpenRouter-бандлом,
# см. docs/LANGFLOW_SETUP.md).
RUN /app/.venv/bin/pip install --no-cache-dir --timeout 100 --retries 10 \
    "lfx-bundles[qdrant]"

# Проверяем сразу при сборке образа.
RUN /app/.venv/bin/python -c "import torch, sentence_transformers; print('sentence-transformers OK:', sentence_transformers.__version__)"
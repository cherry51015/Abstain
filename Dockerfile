# Abstain — AI Risk Decision Engine for Chargebacks

FROM python:3.12-slim

# build-essential: faiss-cpu / sentence-transformers occasionally need to
# compile small extensions on slim base images depending on platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY dataset/ ./dataset/

# Pre-download the embedding model at BUILD time, not on the first API
# request — otherwise your first live demo request stalls on a cold
# Hugging Face download. If the build environment can't reach the internet,
# this warns and continues rather than failing the build; index_builder.py's
# BM25-only fallback (see retrieval/index_builder.py) handles it at runtime
# the same way — degrade, don't break, consistently at both layers.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    || echo "WARNING: could not pre-download embedding model at build time — dense retrieval will fall back to BM25-only at runtime."

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
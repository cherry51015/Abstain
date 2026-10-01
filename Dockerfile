# Abstain API. Small on purpose: no torch/FAISS/LangChain at runtime, so it fits a 512 MB instance.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY alembic/ alembic/
COPY alembic.ini .
COPY data/reason_codes.json data/merchants.json data/
COPY models/ models/

RUN useradd --create-home --uid 10001 abstain && mkdir -p .cache && chown abstain .cache
USER abstain

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health')"

# Migrate, then serve. PORT is injected by most PaaS hosts (Render, Railway, Fly).
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]

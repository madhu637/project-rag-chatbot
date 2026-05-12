# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps for PyMuPDF, FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK punkt to avoid first-run delay
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

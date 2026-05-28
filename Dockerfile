FROM python:3.11-slim

WORKDIR /app

# torch CPU wheel is ~200 MB — install first so it caches as its own layer.
# Pinned to match local training environment.
RUN pip install --no-cache-dir torch==2.12.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Remaining inference deps (torch-geometric, fastapi, uvicorn, pydantic, requests)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code + model weights (model.pt is 42 KB)
COPY src/ ./src/
COPY data/model.pt ./data/model.pt

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

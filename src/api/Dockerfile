# Imagen de la API (FastAPI + modelo champion).
# Contexto: src/   ->   docker build -f src/api/Dockerfile -t tee-api-image:v1 src
FROM python:3.12-slim

# LightGBM necesita OpenMP (libgomp1), que no viene en la variante slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Dependencias primero para reutilizar la caché si sólo cambia el código.
COPY api/requirements.txt /code/api/requirements.txt
RUN python -m pip install --no-cache-dir -r /code/api/requirements.txt

COPY entrenamiento/ /code/entrenamiento/
COPY api/ /code/api/

ENV PYTHONPATH=/code

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

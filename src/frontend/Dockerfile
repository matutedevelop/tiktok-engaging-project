# Imagen del frontend (Streamlit). Sólo es cliente HTTP de la API: no carga el modelo
# ni necesita credenciales de Databricks.
# Contexto: src/   ->   docker build -f src/frontend/Dockerfile -t tee-ui-image:v1 src
FROM python:3.12-slim

WORKDIR /code

COPY frontend/requirements.txt /code/frontend/requirements.txt
RUN python -m pip install --no-cache-dir -r /code/frontend/requirements.txt

COPY frontend/ /code/frontend/

# app.py importa `frontend.cliente`; /code debe estar en la ruta de importación
ENV PYTHONPATH=/code \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

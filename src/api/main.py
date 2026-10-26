"""FastAPI que sirve el `Pipeline` champion de Unity Catalog.

Arranque desde la raíz del repositorio:

    uv run fastapi dev src/api/main.py --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

import mlflow
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mlflow import MlflowClient

from api.esquemas import EstadoServicio, RespuestaPrediccion, SolicitudPrediccion
from entrenamiento.configuracion import ROOT, nombre_modelo_registrado

VERSION_API = "v1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("tee.api")

# Se llena una sola vez al iniciar; cada solicitud reutiliza el mismo modelo
estado: dict = {"modelo": None}


def cargar_modelo() -> dict:
    """Resuelve el alias a una versión concreta y carga esa versión.

    Con `RUTA_MODELO_LOCAL` carga un MLflow Model local (contingencia sin Databricks).
    """
    # Las variables del entorno real tienen prioridad sobre el .env
    load_dotenv(ROOT / ".env", override=False)

    ruta_local = os.getenv("RUTA_MODELO_LOCAL", "")
    if ruta_local:
        return {
            "modelo": mlflow.pyfunc.load_model(ruta_local),
            "nombre": "tee-contingencia-local",
            "alias": "sin-alias",
            "version": "local",
            "familia": None,
        }

    tracking_uri = os.getenv("TRACKING_URI", "databricks")
    registry_uri = os.getenv("REGISTRY_URI", "databricks-uc")
    nombre = os.getenv("MODELO_REGISTRADO") or nombre_modelo_registrado()
    alias = os.getenv("ALIAS_MODELO", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    version = MlflowClient(registry_uri=registry_uri).get_model_version_by_alias(nombre, alias)
    # Se carga la versión resuelta: si el alias se mueve después, la respuesta sigue siendo cierta
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{version.version}")
    return {
        "modelo": modelo,
        "nombre": nombre,
        "alias": alias,
        "version": str(version.version),
        "familia": version.tags.get("family"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        estado.update(cargar_modelo())
        logger.info(
            "Modelo cargado: %s@%s -> version %s (%s)",
            estado["nombre"],
            estado["alias"],
            estado["version"],
            estado["familia"],
        )
    except Exception as error:  # noqa: BLE001
        # La API arranca de todos modos: /health lo reporta y las predicciones responden 503
        estado["modelo"] = None
        logger.error("No se pudo cargar el modelo: %s", type(error).__name__)
    yield


app = FastAPI(
    title="TEE — TikTok Engage Estimator",
    description="Estima las reproducciones de un video de TikTok antes de publicarlo.",
    version=VERSION_API,
    lifespan=lifespan,
)


@app.get("/health", response_model=EstadoServicio)
def health():
    """200 si el modelo está cargado; 503 si el proceso responde pero no puede predecir."""
    if estado["modelo"] is None:
        cuerpo = EstadoServicio(status="degradado", modelo_cargado=False, version_api=VERSION_API)
        return JSONResponse(status_code=503, content=cuerpo.model_dump())
    return EstadoServicio(
        status="ok",
        modelo_cargado=True,
        modelo=estado["nombre"],
        alias=estado["alias"],
        version=estado["version"],
        familia=estado["familia"],
        version_api=VERSION_API,
    )


@app.post("/api/v1/predicciones", response_model=RespuestaPrediccion)
def crear_prediccion(solicitud: SolicitudPrediccion) -> RespuestaPrediccion:
    if estado["modelo"] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible; consulta /health.")
    log_pred = float(estado["modelo"].predict(solicitud.a_dataframe())[0])
    respuesta = RespuestaPrediccion(
        reproducciones_estimadas=max(0, round(float(np.expm1(log_pred)))),
        log_reproducciones=round(log_pred, 4),
        modelo=estado["nombre"],
        alias=estado["alias"],
        version=estado["version"],
        familia=estado["familia"],
    )
    logger.info(
        "Prediccion version %s: %s reproducciones",
        respuesta.version,
        respuesta.reproducciones_estimadas,
    )
    return respuesta

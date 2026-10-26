"""Llamadas a la API y traducción de sus respuestas; no depende de Streamlit."""

import math
import os
import re
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv(override=False)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/predicciones")
API_HEALTH_URL = API_URL.split("/api/")[0] + "/health"
TIMEOUT_S = 10.0

# Test RMSLE del champion al verificar (versión 3, lightgbm); ver notebooks/06-api.ipynb
RMSLE_CHAMPION = 2.057


@dataclass
class Resultado:
    ok: bool
    codigo: int | None
    cuerpo: dict | None
    mensaje: str | None = None


def parsear_hashtags(texto: str) -> list[str]:
    """'#amazonfinds, affordablefashion' -> ['amazonfinds', 'affordablefashion']."""
    return [h.lstrip("#").lower() for h in re.split(r"[\s,]+", texto) if h.lstrip("#")]


def describir_errores_validacion(cuerpo: dict) -> str:
    detalle = cuerpo.get("detail")
    if not isinstance(detalle, list):
        return str(detalle)
    lineas = []
    for error in detalle:
        campo = ".".join(str(p) for p in error.get("loc", [])[1:]) or "solicitud"
        lineas.append(f"- **{campo}**: {error.get('msg')}")
    return "\n".join(lineas)


def _json(respuesta: httpx.Response) -> dict | None:
    try:
        return respuesta.json()
    except ValueError:
        return None


def consultar_salud() -> Resultado:
    try:
        respuesta = httpx.get(API_HEALTH_URL, timeout=3.0)
    except httpx.RequestError:
        return Resultado(False, None, None, "API no disponible")
    cuerpo = _json(respuesta)
    if respuesta.status_code == 200:
        return Resultado(True, 200, cuerpo)
    return Resultado(False, respuesta.status_code, cuerpo, "API activa sin modelo cargado")


def predecir(entrada: dict) -> Resultado:
    """Sólo una respuesta 200 trae predicción; cualquier otro caso devuelve un mensaje."""
    try:
        respuesta = httpx.post(API_URL, json=entrada, timeout=TIMEOUT_S)
    except httpx.RequestError:
        return Resultado(
            False,
            None,
            None,
            f"No fue posible conectarse con la API en `{API_URL}`. Verifica que esté en ejecución.",
        )
    cuerpo = _json(respuesta)
    codigo = respuesta.status_code
    if codigo == 200:
        return Resultado(True, codigo, cuerpo)
    if codigo == 422:
        mensaje = "Los datos no cumplen el contrato de la API:\n\n" + describir_errores_validacion(
            cuerpo or {}
        )
    elif codigo == 503:
        mensaje = "La API está activa, pero el modelo no está disponible. Revisa el log de la API."
    else:
        mensaje = f"La API respondió {codigo}. Revisa que `API_URL` apunte a `/api/v1/predicciones`."
    return Resultado(False, codigo, cuerpo, mensaje)


def rango_tipico(log_reproducciones: float) -> tuple[int, int]:
    """Rango de un error típico (± RMSLE) alrededor de la predicción, en reproducciones."""
    return (
        max(0, round(math.expm1(log_reproducciones - RMSLE_CHAMPION))),
        round(math.expm1(log_reproducciones + RMSLE_CHAMPION)),
    )


def orden_de_magnitud(reproducciones: int) -> str:
    exponente = int(math.log10(max(reproducciones, 1)))
    return "10" + str(exponente).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))

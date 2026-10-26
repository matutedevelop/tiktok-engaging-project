import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 06 — API de predicción

    API en FastAPI que sirve el modelo `champion` del Registry. Recibe los datos de un video antes de
    publicarlo y devuelve las reproducciones estimadas junto con la versión del modelo que respondió.

    | Archivo | Contenido |
    |---|---|
    | [`src/api/esquemas.py`](../src/api/esquemas.py) | contrato de entrada y salida (Pydantic) |
    | [`src/api/main.py`](../src/api/main.py) | carga del modelo y endpoints |
    | [`.env.example`](../.env.example) | variables de configuración |

    El servidor se inicia desde la terminal (sección 5); las celdas de pruebas HTTP requieren que esté
    encendido.
    """)
    return


@app.cell
def _():
    import json
    import logging
    import os

    import httpx
    import mlflow
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from fastapi.testclient import TestClient
    from mlflow import MlflowClient
    from pydantic import ValidationError

    from api import main as api_main
    from api.esquemas import EJEMPLO_VALIDO, RespuestaPrediccion, SolicitudPrediccion
    from entrenamiento.configuracion import COLUMNAS_ENTRADA, ROOT, nombre_modelo_registrado

    load_dotenv(ROOT / ".env", override=False)
    for _nombre in ("httpx", "httpx2", "mlflow"):
        logging.getLogger(_nombre).setLevel(logging.WARNING)

    MODELO_REGISTRADO = os.getenv("MODELO_REGISTRADO") or nombre_modelo_registrado()
    ALIAS_MODELO = os.getenv("ALIAS_MODELO", "champion")
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/predicciones")
    API_BASE = API_URL.split("/api/")[0]
    print(f"Modelo: {MODELO_REGISTRADO}@{ALIAS_MODELO}\nAPI: {API_BASE}")
    return (
        ALIAS_MODELO,
        API_BASE,
        COLUMNAS_ENTRADA,
        EJEMPLO_VALIDO,
        MODELO_REGISTRADO,
        MlflowClient,
        ROOT,
        RespuestaPrediccion,
        SolicitudPrediccion,
        TestClient,
        ValidationError,
        api_main,
        httpx,
        json,
        mlflow,
        np,
        os,
        pd,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Recorrido de una solicitud

    1. El cliente envía un JSON a `POST /api/v1/predicciones`.
    2. Pydantic valida el cuerpo; si algo no cumple el contrato responde **422** sin llamar al modelo.
    3. La solicitud se convierte en un `DataFrame` de una fila con las 20 columnas crudas del modelo.
    4. El `Pipeline` registrado aplica `preparar_features`, el preprocesamiento ajustado y el estimador.
    5. El modelo predice $\log(1 + \text{play\_count})$; la API devuelve `expm1` de ese valor junto con
       el modelo, el alias y la versión.

    El modelo se carga **una sola vez** al iniciar la API; ninguna solicitud reentrena ni lee datos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Contrato de entrada

    Los campos son exactamente `COLUMNAS_ENTRADA`, las columnas que recibe el modelo registrado. Los
    rangos y categorías salen de `data/interim`:

    - `duration`, `music_duration`: 0 a 3,600 s. `vq_score`: 0 a 100.
    - `poi_category`: una de las 12 categorías observadas. `item_comment_status`: `"0"` o `"3"`.
    - Los identificadores van como texto con el formato de `data/interim` (p. ej. `"5096337.0"`); un ID
      no visto en entrenamiento es válido y el modelo lo trata como categoría desconocida.
    - `challenges` se recibe como lista de hashtags y la API la serializa como JSON, igual que en el
      dataset.
    - Un campo no reconocido es un error (`extra="forbid"`).
    """)
    return


@app.cell
def _(EJEMPLO_VALIDO, SolicitudPrediccion, pd):
    esquema = SolicitudPrediccion.model_json_schema()
    _limites = {"minimum": "≥", "maximum": "≤", "maxLength": "long ≤", "maxItems": "elems ≤"}


    def _tipo(p: dict) -> str:
        if "enum" in p:
            return f"categoría ({len(p['enum'])})"
        if p.get("type") == "array":
            return "lista de texto"
        return p.get("type", "")


    pd.DataFrame(
        [
            {
                "campo": campo,
                "tipo": _tipo(p),
                "restricción": ", ".join(f"{s} {p[k]}" for k, s in _limites.items() if k in p),
                "ejemplo": EJEMPLO_VALIDO[campo],
            }
            for campo, p in esquema["properties"].items()
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El contrato coincide con la firma del modelo registrado: mismos campos, mismo orden y mismos tipos.
    """)
    return


@app.cell
def _(
    ALIAS_MODELO,
    COLUMNAS_ENTRADA,
    EJEMPLO_VALIDO,
    MODELO_REGISTRADO,
    MlflowClient,
    SolicitudPrediccion,
    mlflow,
    os,
    pd,
):
    cliente = MlflowClient(registry_uri=os.getenv("REGISTRY_URI", "databricks-uc"))
    mlflow.set_tracking_uri(os.getenv("TRACKING_URI", "databricks"))
    mlflow.set_registry_uri(os.getenv("REGISTRY_URI", "databricks-uc"))

    version_champion = cliente.get_model_version_by_alias(MODELO_REGISTRADO, ALIAS_MODELO)
    firma = mlflow.models.get_model_info(
        f"models:/{MODELO_REGISTRADO}/{version_champion.version}"
    ).signature
    fila = SolicitudPrediccion(**EJEMPLO_VALIDO).a_dataframe()

    comparacion = pd.DataFrame(
        {
            "firma_modelo": {c.name: c.type.to_numpy().name for c in firma.inputs.inputs},
            "dataframe_api": fila.dtypes.astype(str),
        }
    ).loc[COLUMNAS_ENTRADA]
    assert list(SolicitudPrediccion.model_fields) == COLUMNAS_ENTRADA == firma.inputs.input_names()
    print(
        f"@{ALIAS_MODELO} -> versión {version_champion.version} "
        f"({version_champion.tags.get('family')}, test RMSLE = {version_champion.tags.get('test_rmsle')})"
    )
    comparacion
    return fila, version_champion


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Validación de casos inválidos con Pydantic, sin servidor. Cada error es lo que la API devuelve como
    422 en `detail`.
    """)
    return


@app.cell
def _(EJEMPLO_VALIDO, SolicitudPrediccion, ValidationError, pd):
    CASOS_INVALIDOS = {'vq_score fuera de rango': {**EJEMPLO_VALIDO, 'vq_score': 150}, 'duración negativa': {**EJEMPLO_VALIDO, 'duration': -5}, 'categoría desconocida': {**EJEMPLO_VALIDO, 'poi_category': 'Playa'}, 'tipo incorrecto': {**EJEMPLO_VALIDO, 'duration': 'corto'}, 'campo omitido': {k: v for k, v in EJEMPLO_VALIDO.items() if k != 'duration'}, 'campo no reconocido': {**EJEMPLO_VALIDO, 'es_anuncio': True}, 'challenges como texto': {**EJEMPLO_VALIDO, 'challenges': 'amazonfinds'}}
    _filas = []
    for _caso, _cuerpo in CASOS_INVALIDOS.items():
        try:
            SolicitudPrediccion(**_cuerpo)
            _filas.append({'caso': _caso, 'error': 'ninguno'})
        except ValidationError as error:
            e = error.errors()[0]
            _filas.append({'caso': _caso, 'campo': e['loc'][0], 'error': e['msg']})
    pd.DataFrame(_filas)
    return (CASOS_INVALIDOS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Contrato de salida

    | Campo | Significado |
    |---|---|
    | `reproducciones_estimadas` | estimación de `play_count` (`expm1` de la salida del modelo) |
    | `log_reproducciones` | salida directa del modelo, en escala $\log(1 + \text{play\_count})$ |
    | `modelo`, `alias`, `version` | modelo registrado, alias y versión concreta que respondió |
    | `familia` | familia del estimador (`ridge`, `random_forest` o `lightgbm`) |
    """)
    return


@app.cell
def _(RespuestaPrediccion, pd):
    pd.DataFrame(
        [
            {"campo": c, "descripción": p.get("description", "")}
            for c, p in RespuestaPrediccion.model_json_schema()["properties"].items()
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Carga del modelo

    Al iniciar, `cargar_modelo()` resuelve el alias a una versión concreta con
    `get_model_version_by_alias` y carga `models:/<modelo>/<versión>`. Así la versión reportada siempre
    es la que se cargó; si `champion` cambia, hay que reiniciar la API.

    El modelo está registrado como `workspace.tiktok-engaging-estimator.pcd_otono_2026_tiktok_engaging_estimator`:
    el equipo usa un esquema propio en Unity Catalog en lugar de `default`. El nombre se configura con
    `MODELO_REGISTRADO`, así que la API no depende de esa decisión.

    **Si el modelo no puede cargarse** (token inválido, nombre o alias inexistente, sin red), la API
    arranca en estado degradado: `/health` y las predicciones responden **503** y nunca se devuelve una
    predicción inventada. Las entradas inválidas siguen respondiendo 422. La celda siguiente lo
    comprueba en memoria apuntando a un modelo que no existe.
    """)
    return


@app.cell
def _(EJEMPLO_VALIDO, ROOT, TestClient, api_main, os, pd):
    _anterior = os.environ.get("RUTA_MODELO_LOCAL")
    os.environ["RUTA_MODELO_LOCAL"] = str(ROOT / "artifacts" / "no-existe")
    try:
        with TestClient(api_main.app) as c:
            degradado = [
                ("GET /health", c.get("/health")),
                ("POST válida", c.post("/api/v1/predicciones", json=EJEMPLO_VALIDO)),
                ("POST inválida", c.post("/api/v1/predicciones", json={**EJEMPLO_VALIDO, "vq_score": 150})),
            ]
    finally:
        if _anterior is None:
            os.environ.pop("RUTA_MODELO_LOCAL")
        else:
            os.environ["RUTA_MODELO_LOCAL"] = _anterior

    pd.DataFrame(
        [{"solicitud": s, "código": r.status_code, "cuerpo": r.json()} for s, r in degradado]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Configuración y arranque

    Variables en `.env` (ver [`.env.example`](../.env.example)):

    | Variable | Valor |
    |---|---|
    | `DATABRICKS_HOST`, `DATABRICKS_TOKEN` | credenciales de Databricks |
    | `MODELO_REGISTRADO` | `workspace.tiktok-engaging-estimator.pcd_otono_2026_tiktok_engaging_estimator` |
    | `ALIAS_MODELO` | `champion` |
    | `RUTA_MODELO_LOCAL` | opcional: MLflow Model local para probar sin Databricks |

    Desde la raíz del repositorio:

    ```bash
    uv sync --locked
    uv run fastapi dev src/api/main.py --port 8000
    ```

    Al iniciar, el log muestra `Modelo cargado: <modelo>@champion -> version N (<familia>)`.

    - Documentación interactiva: <http://127.0.0.1:8000/docs>
    - Disponibilidad: <http://127.0.0.1:8000/health> (200 con la versión cargada, 503 si no hay modelo)

    Ejemplo con `curl` (en PowerShell usar `curl.exe`):

    ```bash
    curl -X POST http://127.0.0.1:8000/api/v1/predicciones \
      -H "Content-Type: application/json" \
      -d '{"duet_info_duet_from_id": "0.0", "duet_enabled": true, "is_ad": false, "item_mute": false,
           "stitch_enabled": true, "music_original": true, "music_duration": 44.0, "vq_score": 74.62,
           "duration": 28, "desc": "Amazon!!!! #amazonfinds #affordablefashion",
           "challenges": ["amazonfinds", "affordablefashion"], "poi_category": "Place and Address",
           "item_comment_status": "0", "city_code": "5096337.0", "poi_tt_type_name_medium": "Places",
           "poi_tt_type_name_tiny": "Province", "music_id": "7.099057143395667e+18",
           "music_author_name": "So Future", "poi_id": "22535865205217746",
           "diversification_id": "10029.0"}'
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Pruebas contra la API en ejecución
    """)
    return


@app.cell
def _(API_BASE, httpx, json):
    def solicitar(metodo: str, ruta: str, cuerpo: dict | None = None) -> httpx.Response:
        return httpx.request(metodo, f"{API_BASE}{ruta}", json=cuerpo, timeout=30.0)


    health = solicitar("GET", "/health")
    print(health.status_code, json.dumps(health.json(), indent=2, ensure_ascii=False))
    return (solicitar,)


@app.cell
def _(CASOS_INVALIDOS, EJEMPLO_VALIDO, pd, solicitar):
    CASOS = [('válido', 'POST', '/api/v1/predicciones', EJEMPLO_VALIDO, 200), *[(caso, 'POST', '/api/v1/predicciones', cuerpo, 422) for caso, cuerpo in CASOS_INVALIDOS.items()], ('ruta inexistente', 'POST', '/api/v1/prediccion', EJEMPLO_VALIDO, 404), ('método incorrecto', 'GET', '/api/v1/predicciones', None, 405)]
    _filas = []
    for _caso, metodo, ruta, _cuerpo, esperado in CASOS:
        r = solicitar(metodo, ruta, _cuerpo)
        detalle = r.json() if r.status_code == 200 else r.json()['detail']
        if isinstance(detalle, list):
            detalle = detalle[0]['msg']
        _filas.append({'caso': _caso, 'solicitud': f'{metodo} {ruta}', 'esperado': esperado, 'obtenido': r.status_code, 'respuesta': detalle})
    pruebas = pd.DataFrame(_filas)
    assert (pruebas['esperado'] == pruebas['obtenido']).all()
    pruebas
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La predicción de la API coincide con cargar la misma versión directamente desde MLflow, así que la
    API sirve el `Pipeline` registrado sin cambios.
    """)
    return


@app.cell
def _(
    EJEMPLO_VALIDO,
    MODELO_REGISTRADO,
    fila,
    mlflow,
    np,
    pd,
    solicitar,
    version_champion,
):
    respuesta_api = solicitar("POST", "/api/v1/predicciones", EJEMPLO_VALIDO).json()
    modelo_directo = mlflow.pyfunc.load_model(f"models:/{MODELO_REGISTRADO}/{version_champion.version}")
    pred_directa = float(modelo_directo.predict(fila)[0])

    assert respuesta_api["version"] == str(version_champion.version)
    assert np.isclose(respuesta_api["log_reproducciones"], pred_directa, atol=1e-3)
    pd.DataFrame(
        {
            "API": [respuesta_api["version"], respuesta_api["log_reproducciones"]],
            "MLflow directo": [str(version_champion.version), round(pred_directa, 4)],
        },
        index=["versión", "log_reproducciones"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Resultados

    - `@champion` apuntaba a la **versión 3** (`lightgbm`, test RMSLE = 2.057) al verificar.
    - `/health` → 200 con la versión 3. El ejemplo válido devolvió **10,905 reproducciones**
      (`log_reproducciones` = 9.297); el video real tuvo 44,035.
    - Las 7 entradas inválidas → 422 sin predicción; ruta inexistente → 404; `GET` → 405; sin modelo → 503.

    Con RMSLE ≈ 2.06 el error típico es un factor de $e^{2.06} \approx 7.8$, así que la predicción
    indica el **orden de magnitud** del alcance (aquí, 10⁴), no el número exacto. El modelo se entrenó
    con contenido *trending* con POI en EE. UU.; fuera de ese dominio la predicción no es confiable.

    | Evidencia | Archivo |
    |---|---|
    | Entrada válida | [`evidencia/api-prediccion-valida.png`](../evidencia/api-prediccion-valida.png) |
    | Entrada inválida | [`evidencia/api-entrada-invalida.png`](../evidencia/api-entrada-invalida.png) |
    """)
    return


if __name__ == "__main__":
    app.run()

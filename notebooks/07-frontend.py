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
    # 07 — Interfaz con Streamlit

    Formulario en Streamlit donde una persona describe un video antes de publicarlo y obtiene la
    estimación de sus reproducciones. La interfaz es solo un **cliente HTTP** de la API de
    [`06-api`](06-api.ipynb): no carga el modelo ni llama a `predict`.

    | Archivo | Contenido |
    |---|---|
    | [`src/frontend/app.py`](../src/frontend/app.py) | formulario y presentación del resultado |
    | [`src/frontend/cliente.py`](../src/frontend/cliente.py) | llamadas a la API y traducción de respuestas y errores |

    `cliente.py` no depende de Streamlit, así que este notebook usa las mismas funciones que la
    interfaz. Las celdas HTTP requieren la API encendida.
    """)
    return


@app.cell
def _():
    import logging

    import pandas as pd

    from api.esquemas import EJEMPLO_VALIDO, SolicitudPrediccion
    from frontend import cliente

    logging.getLogger("httpx").setLevel(logging.WARNING)
    print(f"API: {cliente.API_URL}\nHealth: {cliente.API_HEALTH_URL}")
    return EJEMPLO_VALIDO, SolicitudPrediccion, cliente, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Recorrido

    1. La persona llena el formulario y presiona **Estimar reproducciones**. Los widgets están en
       un `st.form`, así que la API se llama solo al enviar, no con cada cambio.
    2. La interfaz arma un diccionario con las mismas llaves que `SolicitudPrediccion` y lo envía con
       `httpx.post(API_URL, json=entrada)`.
    3. Con un **200** muestra las reproducciones, el modelo, el alias, la versión y una interpretación.
       Con cualquier otro resultado muestra un mensaje de error y **ningún número**.
    4. Un desplegable muestra la solicitud enviada y la respuesta recibida.

    En la barra lateral se consulta `GET /health` para mostrar si la API está disponible y qué
    versión del modelo tiene cargada.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Formulario

    El formulario se divide en cuatro pestañas. Los límites de los widgets son los mismos del
    contrato de la API, y los valores por defecto corresponden al ejemplo válido de `/docs`.

    | Pestaña | Campos | Widget |
    |---|---|---|
    | Video | `desc`, `challenges`, `duration` (0–3600 s), `vq_score` (0–100) | texto, número |
    | Configuración | `duet_enabled`, `stitch_enabled`, `item_mute`, `is_ad`, `item_comment_status`, `duet_info_duet_from_id` | toggle, selectbox, texto |
    | Música | `music_original`, `music_author_name`, `music_id`, `music_duration` | toggle, texto, número |
    | Ubicación | `poi_category` (12 categorías), `poi_tt_type_name_medium`, `poi_tt_type_name_tiny`, `poi_id`, `city_code`, `diversification_id` | selectbox, texto |

    - Los hashtags se escriben separados por comas o espacios, y `parsear_hashtags` los convierte en la
      lista que espera la API.
    - Los IDs son texto libre con el formato del dataset (p. ej. `"5096337.0"`). Un ID desconocido no
      es un error.
    - Los números se convierten explícitamente con `int(...)` y `float(...)` antes de enviarlos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El payload que arma la interfaz cumple el contrato de la API. Se valida con el mismo modelo
    Pydantic, sin servidor:
    """)
    return


@app.cell
def _(EJEMPLO_VALIDO, SolicitudPrediccion, cliente, pd):
    entrada = {
        **EJEMPLO_VALIDO,
        "challenges": cliente.parsear_hashtags("#amazonfinds, affordablefashion"),
        "duration": int(28.0),
        "music_duration": float(44),
    }
    SolicitudPrediccion(**entrada)
    pd.Series(entrada, name="valor").to_frame()
    return (entrada,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Llamada a la API y manejo de errores

    `predecir(entrada)` hace la llamada y devuelve un `Resultado` con `ok`, el código, el cuerpo y un
    mensaje. La interfaz solo muestra la predicción cuando `ok` es verdadero.

    | Situación | Código | Qué ve la persona |
    |---|---|---|
    | Solicitud válida | 200 | reproducciones estimadas, versión del modelo e interpretación |
    | Dato fuera del contrato (p. ej. un ID vacío) | 422 | "Los datos no cumplen el contrato de la API", con una línea por campo |
    | API activa sin modelo cargado | 503 | "La API está activa, pero el modelo no está disponible" |
    | API apagada o `API_URL` incorrecta | sin respuesta | "No fue posible conectarse con la API en `<API_URL>`" |
    | Ruta equivocada | 404 / 405 | "La API respondió <código>. Revisa que `API_URL` apunte a `/api/v1/predicciones`" |
    """)
    return


@app.cell
def _(cliente, entrada, pd):
    casos = {'válido': entrada, 'city_code vacío': {**entrada, 'city_code': ''}, 'vq_score fuera de rango': {**entrada, 'vq_score': 150.0}}
    filas = []
    for caso, cuerpo in casos.items():
        _r = cliente.predecir(cuerpo)
        filas.append({'caso': caso, 'código': _r.codigo, 'ok': _r.ok, 'se muestra': f"{_r.cuerpo['reproducciones_estimadas']:,} reproducciones" if _r.ok else _r.mensaje})
    _url = cliente.API_URL
    cliente.API_URL = 'http://127.0.0.1:9/api/v1/predicciones'
    try:
        _r = cliente.predecir(entrada)
    finally:
        cliente.API_URL = _url
    filas.append({'caso': 'API inaccesible', 'código': _r.codigo, 'ok': _r.ok, 'se muestra': _r.mensaje})
    pruebas = pd.DataFrame(filas).astype({'código': 'Int64'})
    assert pruebas['ok'].tolist() == [True, False, False, False]
    pd.set_option('display.max_colwidth', None)
    pruebas
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Presentación del resultado

    Además de las reproducciones estimadas, la interfaz muestra:

    - **modelo · alias · versión · familia**, tal como los devuelve la API;
    - el **orden de magnitud** de la estimación;
    - un **rango típico**, que resulta de aplicar ± RMSLE en escala log. Con el RMSLE de test del
      champion (2.057), el valor real suele estar dentro de un factor de ~7.8× hacia arriba o hacia
      abajo;
    - las **limitaciones**: el modelo se entrenó con contenido *trending* con ubicación en EE. UU.
      (primavera 2025) y no conoce seguidores ni historial del creador.
    """)
    return


@app.cell
def _(cliente, entrada, pd):
    salud = cliente.consultar_salud()
    _r = cliente.predecir(entrada).cuerpo
    minimo, maximo = cliente.rango_tipico(_r['log_reproducciones'])
    pd.Series({'estado de la API': 'disponible' if salud.ok else salud.mensaje, 'reproducciones estimadas': f"{_r['reproducciones_estimadas']:,}", 'orden de magnitud': cliente.orden_de_magnitud(_r['reproducciones_estimadas']), 'rango típico': f'{minimo:,} – {maximo:,}', 'modelo': _r['modelo'], 'alias · versión · familia': f"{_r['alias']} · {_r['version']} · {_r['familia']}"}, name='interfaz').to_frame()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Configuración y arranque

    La interfaz solo necesita `API_URL` en `.env` (por defecto
    `http://127.0.0.1:8000/api/v1/predicciones`); no usa credenciales de Databricks.

    Primero se enciende la API y después la interfaz, cada una en su terminal, desde la raíz del
    repositorio:

    ```bash
    # Terminal 1: API
    uv run fastapi dev src/api/main.py --port 8000

    # Terminal 2: interfaz
    uv run streamlit run src/frontend/app.py --server.port 8501
    ```

    La interfaz queda en <http://localhost:8501>. Cada envío del formulario deja en la terminal de la
    API las líneas `Prediccion version N: ...` y `"POST /api/v1/predicciones HTTP/1.1" 200`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Resultados

    - Con la API sirviendo el champion (**versión 3, lightgbm**), el ejemplo por defecto devuelve
      **10,905 reproducciones**: orden de magnitud 10⁴ y rango típico de 1,393 a 85,305.
    - Un dato fuera del contrato produce un 422, y la interfaz lo traduce a un mensaje por campo.
    - Con la API apagada, la interfaz muestra el error de conexión y la barra lateral indica
      "API no disponible".
    - En ningún caso de error se muestra una predicción.

    | Evidencia | Archivo |
    |---|---|
    | Formulario y predicción con versión e interpretación | [`evidencia/frontend-prediccion.png`](../evidencia/frontend-prediccion.png) |
    | Solicitud enviada y respuesta de la API para esa misma predicción | [`evidencia/frontend-solicitud-respuesta.png`](../evidencia/frontend-solicitud-respuesta.png) |
    | Error de validación (422) | [`evidencia/frontend-error-validacion.png`](../evidencia/frontend-error-validacion.png) |
    | API apagada | [`evidencia/frontend-error-conexion.png`](../evidencia/frontend-error-conexion.png) |
    """)
    return


if __name__ == "__main__":
    app.run()

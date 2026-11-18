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
    # 08 — Contenedores con Docker y Docker Compose

    La API de [`06-api`](06-api.ipynb) y la interfaz de [`07-frontend`](07-frontend.ipynb) se
    empaquetan en dos imágenes de Docker y se levantan juntas con Docker Compose. Así se ejecutan
    igual en la computadora local y en la instancia EC2 de [`09-despliegue`](09-despliegue.ipynb).

    | Archivo | Contenido |
    |---|---|
    | [`src/api/Dockerfile`](../src/api/Dockerfile) | imagen de la API (FastAPI + modelo `champion`) |
    | [`src/frontend/Dockerfile`](../src/frontend/Dockerfile) | imagen de la interfaz (Streamlit, cliente HTTP) |
    | [`src/.dockerignore`](../src/.dockerignore) | qué no entra al contexto de construcción |
    | [`src/api/requirements.txt`](../src/api/requirements.txt), [`src/frontend/requirements.txt`](../src/frontend/requirements.txt) | dependencias de cada imagen, exportadas de `uv.lock` |
    | [`docker-compose.yaml`](../docker-compose.yaml) | los dos servicios, su configuración y su red |

    Las celdas de código solo leen estos archivos del repositorio; no construyen imágenes ni
    necesitan Docker.
    """)
    return


@app.cell
def _():
    from entrenamiento.configuracion import ROOT

    def mostrar(ruta: str) -> None:
        print(f"── {ruta} ──\n")
        print((ROOT / ruta).read_text(encoding="utf-8"))

    return ROOT, mostrar


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imágenes

    Las dos imágenes parten de `python:3.12-slim`, la misma versión de Python del proyecto
    (`.python-version`). Cada `Dockerfile` copia primero su `requirements.txt` e instala las
    dependencias, y después copia el código: si solo cambia el código, Docker reutiliza la capa
    de dependencias, que es la más lenta (≈ 95 s para la API y ≈ 30 s para la interfaz).

    ### API
    """)
    return


@app.cell
def _(mostrar):
    mostrar("src/api/Dockerfile")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **`libgomp1`.** LightGBM usa OpenMP y la variante `slim` no lo trae; sin esta librería el
      modelo `champion` no se puede cargar.
    - **Se copian `api/` y `entrenamiento/`.** El `Pipeline` registrado usa `preparar_features` y
      `FrequencyEncoder` de `entrenamiento.features`; para deserializarlo la API necesita ese
      módulo importable. `PYTHONPATH=/code` reemplaza al `uv sync` del ambiente local, que
      instalaba `src/` como paquete.
    - **No se copia el modelo.** La imagen no trae ningún artefacto: al iniciar, la API resuelve
      `@champion` en Unity Catalog y descarga esa versión. Reasignar el alias no requiere
      reconstruir la imagen, solo reiniciar el contenedor.

    ### Interfaz
    """)
    return


@app.cell
def _(mostrar):
    mostrar("src/frontend/Dockerfile")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La interfaz es solo un cliente HTTP: copia únicamente `frontend/`, no instala MLflow,
    LightGBM ni scikit-learn y no recibe credenciales. Por eso su imagen pesa menos de dos
    tercios de la de la API.

    | Imagen | Contenido de `/code` | Tamaño |
    |---|---|---:|
    | `tee-api-image:v1` | `api/`, `entrenamiento/` | 1.32 GB |
    | `tee-ui-image:v1` | `frontend/` | 780 MB |

    ### Dependencias por imagen

    Los `requirements.txt` no se escriben a mano: se exportan de `uv.lock` con `uv export --locked`,
    podando (`--prune`) lo que cada imagen no necesita (Prefect, Optuna, marimo, PyTorch,
    `datasets`, etc.). Las versiones son exactamente las del ambiente donde se entrenó el modelo,
    lo que evita diferencias de scikit-learn o LightGBM al deserializarlo. El comando exacto queda
    en la primera línea de cada archivo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Contexto de construcción y `.dockerignore`

    Las dos imágenes se construyen con contexto `src/`, no con la raíz del repositorio. Así quedan
    fuera de entrada `data/` (≈ 2 GB de Parquet), `.venv/`, los notebooks y las evidencias, y el
    contexto que se envía a Docker es de unos cuantos KB. Dentro de `src/`, el `.dockerignore`
    descarta además:
    """)
    return


@app.cell
def _(mostrar):
    mostrar("src/.dockerignore")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lo más importante es que **ningún `.env` entra a una imagen**. Se comprobó listando las
    imágenes construidas:

    ```bash
    docker run --rm --entrypoint sh tee-api-image:v1 -c 'ls -a /code; find / -name ".env*" -not -path "/proc/*"'
    # /code: api  entrenamiento        (y ningún .env en todo el sistema de archivos)
    docker run --rm --entrypoint sh tee-ui-image:v1 -c 'ls -a /code; find /code -name ".env*"'
    # /code: frontend                  (ningún .env)
    ```

    Una imagen se puede compartir o subir a un registro; si tuviera el token de Databricks,
    cualquiera que la descargue lo tendría.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Configuración externa

    La configuración entra al contenedor **al ejecutarlo**, con `env_file`, nunca al construirlo.
    La misma imagen sirve en local y en EC2; solo cambia el `.env` de cada máquina.

    | Archivo (ignorado por Git) | Lo lee | Variables |
    |---|---|---|
    | `.env` (raíz) | servicio `api` | `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `TRACKING_URI`, `REGISTRY_URI`, `MODELO_REGISTRADO`, `ALIAS_MODELO`, `RUTA_MODELO_LOCAL` |
    | `src/frontend/.env` | servicio `ui` | `API_URL` |

    Los dos tienen su `.env.example` versionado, con los nombres y sin valores reales. La
    interfaz no recibe el `.env` de la raíz: no necesita el token y así no puede filtrarlo.

    En el código, `load_dotenv(..., override=False)` hace que las variables que ya puso Docker
    tengan prioridad. Como la imagen no trae `.env`, dentro del contenedor la única fuente es
    `env_file`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Docker Compose y red host
    """)
    return


@app.cell
def _(mostrar):
    mostrar("docker-compose.yaml")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **Un proyecto, dos servicios.** `name: tee` fija los nombres; `ui` depende de `api`
      (`depends_on`), así que Compose arranca primero la API.
    - **`command` explícito.** Repite el `CMD` de cada imagen, pero con el `--host` que corresponde
      al despliegue (ver abajo).
    - **Sin `ports` ni `networks`.** Con `network_mode: host` no hay mapeo de puertos: cada proceso
      escucha directamente en la red de la máquina.

    ### Qué significa `localhost` aquí

    Con la red por defecto de Docker (*bridge*), cada contenedor tiene su propia red, y
    `127.0.0.1` dentro de la interfaz sería **el contenedor de la interfaz**, no la API. Con
    `network_mode: host`, los dos contenedores comparten la red del anfitrión, así que
    `127.0.0.1` dentro de cualquiera de ellos es **la máquina anfitriona**. Por eso:

    - `API_URL=http://127.0.0.1:8000/api/v1/predicciones` funciona igual dentro de Docker que con
      `uv run`, sin cambiar el código del frontend.
    - La API escucha en `--host 127.0.0.1`: solo procesos de la misma máquina (la interfaz, o un
      `curl` por SSH) la alcanzan. Aunque alguien abriera el puerto 8000 en el firewall, la API no
      aceptaría conexiones de fuera.
    - La interfaz escucha en `--server.address=0.0.0.0`: acepta conexiones de otras computadoras,
      y el grupo de seguridad de EC2 decide cuáles (ver [`09-despliegue`](09-despliegue.ipynb)).

    ```
    navegador ──:8501──▶ [ui  0.0.0.0:8501] ──127.0.0.1:8000──▶ [api 127.0.0.1:8000] ──▶ Databricks (@champion)
                         └──────────── misma red: la del anfitrión ────────────┘
    ```

    La API originalmente escuchaba en `0.0.0.0` y se limitó a `127.0.0.1` antes de desplegar
    (commit `fix(tarea-08): limita la api a localhost en compose`).
    """)
    return


@app.cell
def _(ROOT):
    import re

    _compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
    _hosts = dict(
        api=re.search(r'"api\.main:app", "--host", "([^"]+)"', _compose).group(1),
        ui=re.search(r"--server\.address=([^\"]+)", _compose).group(1),
    )
    assert _hosts == {"api": "127.0.0.1", "ui": "0.0.0.0"}, _hosts
    assert len(re.findall(r"^\s+network_mode: host$", _compose, flags=re.M)) == 2
    assert not re.search(r"^\s+ports:", _compose, flags=re.M)
    print("api escucha en", _hosts["api"], "| ui escucha en", _hosts["ui"], "| ambos en red host, sin ports")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Ejecución local

    Desde la raíz del repositorio, con Docker Desktop encendido, `.env` lleno y
    `cp src/frontend/.env.example src/frontend/.env`:

    ```bash
    uv sync --locked
    docker compose config --quiet     # valida el archivo; no imprime nada si está bien
    docker compose up --build         # construye las dos imágenes y deja los logs en la terminal
    # en otra terminal, después de usar http://localhost:8501
    docker compose ps
    docker compose down
    ```

    En Windows, `network_mode: host` requiere Docker Desktop 4.34 o posterior con *host
    networking* habilitado; se probó con Docker Desktop 4.97 sin configuración adicional.

    ### Resultado

    1. **Builds.** Las dos imágenes terminan en `Built` y los contenedores en `Created`.
    2. **Arranque.** La interfaz queda lista en segundos. La API descarga el modelo de Unity Catalog
       y registra
       `Modelo cargado: workspace.tiktok-engaging-estimator.pcd_otono_2026_tiktok_engaging_estimator@champion -> version 3 (lightgbm)`,
       seguido de `Uvicorn running on http://127.0.0.1:8000`.
    3. **Predicción.** Con el ejemplo por defecto, el formulario en `http://localhost:8501` muestra
       **10,905 reproducciones** (versión 3, lightgbm), el mismo resultado que en
       [`07-frontend`](07-frontend.ipynb) sin contenedores. En el log de la API aparecen
       `Prediccion version 3: 10905 reproducciones` y `"POST /api/v1/predicciones HTTP/1.1" 200 OK`,
       con origen `127.0.0.1`: la petición viene de la interfaz por la red host.
    4. **Servicios.** `docker compose ps` muestra `tee-api-container` y `tee-ui-container` en `Up`.
    5. **Cierre.** `docker compose down` detiene y elimina los dos contenedores.

    Al iniciar, MLflow imprime una advertencia larga de *dependency mismatches* (`prefect`,
    `optuna`, `torch`, …). El modelo se registró con el ambiente completo del proyecto y la imagen
    de la API instala solo lo necesario para servirlo. No afecta: la predicción es idéntica a la
    del ambiente completo.

    | Evidencia | Archivo |
    |---|---|
    | Builds de las dos imágenes | [`evidencia/ambiente-builds.png`](../evidencia/ambiente-builds.png) |
    | Servicios arriba (`docker compose ps`) | [`evidencia/integracion-local-servicios.png`](../evidencia/integracion-local-servicios.png) |
    | Log de la API: modelo cargado, `127.0.0.1` y predicción | [`evidencia/integracion-local-logs-api.png`](../evidencia/integracion-local-logs-api.png) |
    | Predicción en la interfaz con versión | [`evidencia/integracion-local-prediccion.png`](../evidencia/integracion-local-prediccion.png) |

    ![Builds](../evidencia/ambiente-builds.png)
    ![Servicios](../evidencia/integracion-local-servicios.png)
    ![Logs de la API](../evidencia/integracion-local-logs-api.png)
    ![Predicción local](../evidencia/integracion-local-prediccion.png)
    """)
    return


if __name__ == "__main__":
    app.run()

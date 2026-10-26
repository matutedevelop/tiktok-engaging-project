import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", css_file="custom.css")

with app.setup:
    import inspect

    import marimo as mo
    import mlflow
    import numpy as np
    import pandas as pd
    from mlflow import MlflowClient

    from entrenamiento import flujo, hpo
    from entrenamiento.configuracion import (
        EXPERIMENTO,
        FAMILIAS,
        N_FILAS,
        RUTA_DATOS,
        RUTA_PARAMETROS,
        configurar_mlflow,
        nombre_modelo_registrado,
    )

    # Ejecutar el flow desde el notebook es opcional: lo normal es correrlo desde la
    # terminal (sección 5). Con False el notebook sólo consulta Prefect y MLflow.
    EJECUTAR_FLOW = False
    HACER_TUNING = False
    IDENTIFICADOR = "notebook"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Pipeline de entrenamiento con Prefect

    **Propósito.** En `03-experimentos` el recorrido de datos a modelo vive en celdas que
    dependen de ejecutarse en el orden correcto. Aquí ese recorrido se convierte en un
    **workflow explícito**: un flow de Prefect que carga y valida los datos, prepara features,
    ejecuta (o reutiliza) el HPO de las tres familias, entrena y evalúa sus mejores modelos y los
    registra en MLflow (Databricks) y Unity Catalog con el alias `champion`.

    El código vive en `src/entrenamiento/`, no en este notebook:

    | Módulo | Responsabilidad |
    |---|---|
    | `configuracion.py` | features, rutas, semilla, familias, nombres de experimento y modelo, conexión a MLflow |
    | `features.py` | preparación sin estado (`preparar_features`), transformers propios y `construir_pipeline` |
    | `hpo.py` | espacios de búsqueda, estudio de Optuna y runs `hpo-<familia>` → `trial-NNN` |
    | `flujo.py` | tasks, flow `tee-entrenamiento` y CLI |

    Este notebook explica las etapas y decisiones, muestra cómo ejecutar el flow y comprueba el
    resultado consultando Prefect y MLflow.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Dos significados de *pipeline*

    | Objeto | Alcance | En TEE |
    |---|---|---|
    | `sklearn.pipeline.Pipeline` | transformaciones aprendidas + estimador | frecuencias, TF-IDF/SVD, one-hot, target encoding + Ridge / RF / LightGBM |
    | Workflow de Prefect | ciclo completo de datos a Registry | carga → validación → HPO → entrenamiento → test → registro → alias |

    El `Pipeline` de scikit-learn garantiza que entrenamiento e inferencia apliquen las mismas
    transformaciones; el flow garantiza que el proceso completo no dependa de pasos manuales.
    El modelo que se registra además antepone `preparar_features` al `Pipeline` entrenado, así que
    recibe las **columnas crudas** de `data/interim` y la API no tiene que reimplementar la
    preparación.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Etapas del flow
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.mermaid(
        """
        flowchart TD
            A[configurar-mlflow] --> B[cargar-datos]
            B --> C[validar-datos]
            C --> D[preparar-features]
            D --> E[dividir-datos<br/>train / validación / test]
            E -->|train + validación| F[parametros-familia<br/>HPO o JSON]
            F --> G[entrenar-familia<br/>train + validación]
            G --> H[evaluar-familia<br/>test, una vez]
            H --> I[registrar-familia<br/>run final + versión UC]
            I -->|x3 familias| J[asignar-aliases]
            J --> K[verificar-champion]
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | # | Task | Entrada | Salida | Criterio de éxito / falla |
    |---|---|---|---|---|
    | 1 | `configurar-mlflow` | `.env` | tracking/registry URI y experimento | falla si faltan `DATABRICKS_HOST` o `DATABRICKS_TOKEN` |
    | 2 | `cargar-datos` | `data/interim/sample_data.parquet`, `n_filas` | submuestra de 300k filas (`random_state=42`) | falla si no existe el Parquet (hay que correr `02`) |
    | 3 | `validar-datos` | submuestra cruda | la misma tabla | falla si faltan columnas, hay menos de 1,000 filas o `play_count` es nulo/negativo; nulos en otras columnas sólo generan *warning* |
    | 4 | `preparar-features` | tabla cruda | 27 features + `log_play_count` | transformaciones fila por fila, sin estadísticas del conjunto |
    | 5 | `dividir-datos` | features | train 80 %, validación 10 %, test 10 % | estratificado por deciles de $y$, igual que en `03` |
    | 6 | `parametros-{familia}` | familia, flag, **train y validación** | mejores parámetros, RMSLE de validación, `hpo_run_id` | con HPO crea `hpo-<familia>` y sus `trial-NNN`; sin HPO lee `artifacts/05-mejores-parametros.json` |
    | 7 | `entrenar-{familia}` | parámetros, train + validación | `Pipeline` ajustado y tiempo | el preprocesamiento se ajusta dentro del `Pipeline` |
    | 8 | `evaluar-{familia}` | `Pipeline` ajustado, **test** | RMSLE, $R^2$, MAE log, Spearman | primera y única vez que se consulta el test |
    | 9 | `registrar-{familia}` | modelo, métricas, ejemplo | run `<familia>-final` anidado en `hpo-<familia>`, nueva versión en UC | modelo con firma, ejemplo de entrada y tags de familia, run y métricas |
    | 10 | `asignar-aliases` | las tres versiones | ranking | menor RMSLE de test → `champion`, luego `challenger` y `candidate` |
    | 11 | `verificar-champion` | nombre del modelo, ejemplo | versión cargada y predicciones | carga `models:/<modelo>@champion` igual que la API |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Decisiones

    - **Fronteras de task.** Cada task es una etapa que se puede nombrar, verificar y
      diagnosticar por separado. `construir_pipeline`, `proponer_parametros` o `evaluar` son
      funciones puras pequeñas y no necesitan ser tasks. Las tasks por familia usan
      `task_run_name` (`entrenar-lightgbm`, …) para identificarlas en la interfaz.
    - **Sin reintentos.** Todas las entradas son locales y un schema inválido no se corrige
      repitiendo la misma función. Tampoco se reintenta `registrar-{familia}`: repetirla después de
      una falla parcial podría crear versiones duplicadas en el Registry. Si Databricks falla, se
      vuelve a ejecutar el flow sin HPO.
    - **Sin caché.** Las tasks reciben DataFrames grandes; calcular la llave de caché a partir de
      ellos sería caro y no queremos reutilizar resultados entre corridas (`cache_policy=NO_CACHE`).
    - **HPO activable.** `--hacer-tuning` ejecuta los tres estudios y guarda el resultado en
      `artifacts/05-mejores-parametros.json`; `--sin-tuning` reutiliza ese archivo para depurar
      el registro sin repetir los trials. En ambos casos se reentrena, se evalúa y se registra. El
      JSON guarda el `hpo_run_id`, así que los runs finales de una corrida sin HPO se cuelgan del
      mismo `hpo-<familia>` que encontró los parámetros.
    - **Separación de datos.** El test no aparece en la firma de las funciones de HPO (se comprueba
      en la sección 6). Los parámetros se eligen con validación; el test sólo ordena los tres
      modelos finales, como en la Clase 13.
    - **Mismo preprocesamiento en entrenamiento e inferencia.** El modelo registrado es
      `preparar_features → ColumnTransformer → estimador`. Los transformers propios viven en
      `entrenamiento.features`, un módulo instalado con el proyecto, para que el pickle se pueda
      cargar desde la API.
    - **Alias por resultado.** El ranking se calcula en cada corrida; no se fija de antemano qué
      familia gana.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## 4. Configuración

    | Concepto | Valor |
    |---|---|
    | Experimento | `{EXPERIMENTO}` |
    | Modelo registrado | `{nombre_modelo_registrado()}` |
    | Datos | `data/interim/{RUTA_DATOS.name}` (salida de `02-preparacion-datos`) |
    | Submuestra | {N_FILAS:,} filas |
    | Parámetros persistidos | `artifacts/{RUTA_PARAMETROS.name}` (ignorado por git) |
    | Familias | {", ".join(f"`{f}`" for f in FAMILIAS)} |

    Las credenciales se leen de `.env` en la raíz (ignorado por git), como en la Clase 12:

    ```dotenv
    DATABRICKS_HOST=https://<identificador-del-workspace>.cloud.databricks.com
    DATABRICKS_TOKEN=<token>
    ```

    `TRACKING_URI` y `REGISTRY_URI` son opcionales: por defecto valen `databricks` y
    `databricks-uc`. Sólo se cambian para probar contra un MLflow local.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Ejecución

    **Terminal 1** (se queda ocupada con el servidor; interfaz en <http://127.0.0.1:4200>):

    ```bash
    uv run prefect server start
    ```

    **Terminal 2**, desde la raíz del repositorio:

    ```bash
    uv run prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

    # Primera corrida: HPO de las tres familias + registro
    uv run python -m entrenamiento.flujo --hacer-tuning --identificador primera-corrida

    # Corridas posteriores: reutiliza los parámetros del JSON
    uv run python -m entrenamiento.flujo --sin-tuning --identificador sin-hpo
    ```

    Opciones: `--n-filas` cambia el tamaño de la submuestra y `--n-trials` fija el número de trials
    de todas las familias (útil para una prueba rápida, p. ej. `--n-filas 5000 --n-trials 2`).
    El nombre del flow run es `tee-<identificador>` y aparece igual en Prefect y en el tag
    `prefect_run` de los runs finales de MLflow, para relacionar ambas vistas.

    El flow también puede ejecutarse desde la celda siguiente con `EJECUTAR_FLOW = True`.
    """)
    return


@app.cell
def _():
    if EJECUTAR_FLOW:
        resultado_flow = flujo.flujo_entrenamiento(
            identificador=IDENTIFICADOR, hacer_tuning=HACER_TUNING
        )
        _salida = resultado_flow["ranking"]
    else:
        resultado_flow = None
        _salida = mo.md(
            "`EJECUTAR_FLOW = False`: se consultan los resultados de la última corrida."
        )
    _salida
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Comprobación: el test no entra al HPO

    Las funciones que eligen hiperparámetros sólo reciben train y validación. Si alguien agregara
    el test a su firma, esta celda fallaría.
    """)
    return


@app.cell
def _():
    _firmas = {
        "hpo.ajustar_familia": hpo.ajustar_familia,
        "flujo.obtener_parametros": flujo.obtener_parametros.fn,
    }
    _filas = []
    for _nombre, _funcion in _firmas.items():
        _parametros = list(inspect.signature(_funcion).parameters)
        assert not any("test" in p for p in _parametros), _nombre
        _filas.append({"función": _nombre, "parámetros": ", ".join(_parametros)})
    pd.DataFrame(_filas)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Última ejecución en Prefect

    Se consulta la API local de Prefect: el flow run más reciente de `tee-entrenamiento` y el
    estado y duración de cada una de sus tasks. Requiere el servidor de la sección 5 encendido.
    """)
    return


@app.cell
def _():
    from prefect.client.orchestration import get_client
    from prefect.client.schemas.filters import (
        FlowFilter,
        FlowFilterName,
        FlowRunFilter,
        FlowRunFilterId,
    )
    from prefect.client.schemas.sorting import FlowRunSort, TaskRunSort

    try:
        with get_client(sync_client=True) as _cliente:
            _flow_runs = _cliente.read_flow_runs(
                flow_filter=FlowFilter(name=FlowFilterName(any_=["tee-entrenamiento"])),
                sort=FlowRunSort.START_TIME_DESC,
                limit=1,
            )
            ultimo_flow_run = _flow_runs[0] if _flow_runs else None
            _task_runs = (
                _cliente.read_task_runs(
                    flow_run_filter=FlowRunFilter(
                        id=FlowRunFilterId(any_=[ultimo_flow_run.id])
                    ),
                    sort=TaskRunSort.EXPECTED_START_TIME_ASC,
                )
                if ultimo_flow_run
                else []
            )
        tasks_prefect = pd.DataFrame(
            [
                {
                    "task": _t.name,
                    "estado": _t.state_name,
                    "duración_s": round(_t.total_run_time.total_seconds(), 1),
                }
                for _t in _task_runs
            ]
        )
        _salida = (
            mo.vstack(
                [
                    mo.md(
                        f"Flow run **`{ultimo_flow_run.name}`**: "
                        f"`{ultimo_flow_run.state_name}`, "
                        f"{ultimo_flow_run.total_run_time.total_seconds():.0f} s"
                    ),
                    tasks_prefect,
                ]
            )
            if ultimo_flow_run
            else mo.md("Todavía no hay ejecuciones de `tee-entrenamiento`.")
        )
    except Exception as _error:
        _salida = mo.md(f"No se pudo consultar Prefect: `{type(_error).__name__}`")
    _salida
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8. Resultado en MLflow y Unity Catalog

    Se leen los runs finales más recientes de cada familia y las versiones del modelo
    registrado. Con esto se reconstruye el ranking sin depender de la sesión que ejecutó el flow.
    """)
    return


@app.cell
def _():
    try:
        configurar_mlflow()
        _runs = mlflow.search_runs(
            experiment_names=[EXPERIMENTO],
            filter_string="tags.role = 'model-final'",
            order_by=["attributes.start_time DESC"],
        )
        runs_finales = (
            _runs.drop_duplicates("tags.family")
            .loc[
                :,
                [
                    "tags.family",
                    "tags.mlflow.runName",
                    "tags.prefect_run",
                    "metrics.test_rmsle",
                    "metrics.test_r2",
                    "metrics.test_spearman",
                    "metrics.training_time_seconds",
                    "run_id",
                ],
            ]
            .sort_values("metrics.test_rmsle")
            .reset_index(drop=True)
        )
        _salida = runs_finales
    except Exception as _error:
        runs_finales = None
        _salida = mo.md(f"No se pudo consultar MLflow: `{type(_error).__name__}: {_error}`")
    _salida
    return (runs_finales,)


@app.cell
def _(runs_finales):
    _salida = mo.md("Sin conexión a MLflow.")
    if runs_finales is not None:
        _nombre = nombre_modelo_registrado()
        _cliente = MlflowClient()
        _alias = {}
        for _a in ("champion", "challenger", "candidate"):
            try:
                _alias[_cliente.get_model_version_by_alias(_nombre, _a).version] = _a
            except mlflow.exceptions.MlflowException:
                continue  # el alias todavía no existe
        versiones = pd.DataFrame(
            [
                {
                    "versión": int(_v.version),
                    "alias": _alias.get(_v.version, ""),
                    "familia": _v.tags.get("family"),
                    "test_rmsle": _v.tags.get("test_rmsle"),
                    "run_final": _v.tags.get("final_run_id"),
                    "prefect_run": _v.tags.get("prefect_run"),
                }
                for _v in _cliente.search_model_versions(f"name='{_nombre}'")
            ]
        ).sort_values("versión", ascending=False)
        _salida = mo.vstack([mo.md(f"Modelo **`{_nombre}`**"), versiones])
    _salida
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9. Comprobación: cargar `@champion`

    La API cargará el modelo por alias. Aquí se hace lo mismo con cinco filas crudas de
    `data/interim` (sin `play_count`) y se convierte la predicción de $\log(1+\text{plays})$ a
    reproducciones.
    """)
    return


@app.cell
def _(runs_finales):
    _salida = mo.md("Sin conexión a MLflow.")
    if runs_finales is not None:
        from entrenamiento.configuracion import COLUMNAS_ENTRADA
        from entrenamiento.features import normalizar_entrada

        _nombre = nombre_modelo_registrado()
        _version = MlflowClient().get_model_version_by_alias(_nombre, "champion")
        _modelo = mlflow.pyfunc.load_model(f"models:/{_nombre}@champion")
        _ejemplo = normalizar_entrada(
            pd.read_parquet(RUTA_DATOS, columns=COLUMNAS_ENTRADA + ["play_count"]).head(5)
        )
        _pred = _modelo.predict(_ejemplo[COLUMNAS_ENTRADA])
        _salida = mo.vstack(
            [
                mo.md(
                    f"`@champion` → versión **{_version.version}** "
                    f"({_version.tags.get('family')})"
                ),
                pd.DataFrame(
                    {
                        "play_count_real": _ejemplo["play_count"].to_numpy(),
                        "log_play_count_pred": _pred,
                        "play_count_pred": np.expm1(_pred).round(),
                    }
                ),
            ]
        )
    _salida
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 10. Diagnóstico de fallas

    | Síntoma | Causa probable | Siguiente verificación |
    |---|---|---|
    | `RuntimeError: Failed to reach API at http://127.0.0.1:4200/api/` | el servidor de Prefect no está encendido | `uv run prefect server start` en otra terminal |
    | el flow no aparece en la interfaz | `PREFECT_API_URL` apunta a otro lugar | `uv run prefect config view` |
    | `configurar-mlflow` falla | faltan variables en `.env` o el token venció | revisa los nombres `DATABRICKS_HOST` y `DATABRICKS_TOKEN` sin imprimir sus valores |
    | `cargar-datos` falla con `FileNotFoundError` | no existe `data/interim/sample_data.parquet` | `uv run python notebooks/02-preparacion-datos.py sample` |
    | `validar-datos` falla | cambió la salida de `02` | compara las columnas del mensaje con `COLUMNAS_LECTURA` |
    | `parametros-*` falla sin HPO | no existe el JSON o le falta la familia | ejecuta una vez con `--hacer-tuning` |
    | `registrar-*` falla con un `run_id` inexistente | el JSON viene de otro servidor de MLflow (p. ej. una prueba local) | borra `artifacts/05-mejores-parametros.json` y corre con `--hacer-tuning` |
    | Registry rechaza el nombre | catálogo o esquema incorrectos | debe ser `workspace.default.<modelo>` (`MLFLOW_CATALOG`, `MLFLOW_SCHEMA`) |
    | el modelo cargado no encuentra `FrequencyEncoder` | el proyecto no está instalado en el ambiente que carga | `uv sync --locked` instala `src/` como paquete |
    | todos los trials quedan al mismo nivel | se perdió `nested=True` | revisa `hpo.ajustar_familia` |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 11. Verificación local

    Antes de usar Databricks, el flow se probó contra un MLflow local (SQLite) con
    `--n-filas 5000 --n-trials 2`, en dos corridas:

    - **con HPO:** las 11 tasks terminaron en `Completed`; se crearon `hpo-ridge`,
      `hpo-random-forest` y `hpo-lightgbm`, cada uno con sus `trial-000`/`trial-001` y su
      `<familia>-final` anidados; se registraron tres versiones del mismo modelo, se asignaron
      los tres aliases y `@champion` predijo sobre columnas crudas;
    - **sin HPO:** las tasks `parametros-*` reutilizaron el JSON, no se creó ningún estudio nuevo,
      los runs finales se colgaron de los `hpo-*` originales y las métricas de test fueron
      idénticas a las de la primera corrida (las semillas están fijas).

    ### Interpretación

    > **Pendiente de redactar con la corrida en Databricks:** duración de cada etapa (el HPO
    > domina el tiempo), familia que quedó como `champion` y su versión, consistencia con
    > `03-experimentos`, y limitaciones (ejecución secuencial de las familias, servidor de Prefect
    > local, ranking por test con un solo split).

    ### Evidencia

    Capturas en `evidencia/`: flow run exitoso con sus tasks en Prefect, experimento con los
    tres `hpo-*` y el modelo registrado con sus versiones y el alias `champion`.
    """)
    return


if __name__ == "__main__":
    app.run()

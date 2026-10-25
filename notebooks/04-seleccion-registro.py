import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", css_file="custom.css")

with app.setup:
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from time import perf_counter

    from _03_experimentos import (
        construir_pipeline,
        preparar_features,
        config_mlflow,
        mlflow_data,
        fit_evaluate,
    )
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
    import mlflow
    import pandas as pd
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    load_dotenv()

    SEED = 42
    N_FILAS = 300_000
    HACER_TUNING = True

    ROOT = Path(__file__).parent.parent
    RUTA_DATOS_SAMPLE = ROOT / "data" / "interim" / "sample_data.parquet"
    RUTA_DATOS_COMPLETE = ROOT / "data" / "interim" / "data.parquet"

    # Mismo conjunto de variables que dejó 02-preparacion-datos.py
    BOOLEAN_FEATURES = [
        "duet_enabled",
        "is_ad",
        "item_mute",
        "stitch_enabled",
        "music_original",
    ]
    NUMERICAL_FEATURES = ["music_duration", "vq_score", "duration"]
    TEXT_FEATURES = ["desc", "challenges"]
    OHE_FEATURES = ["poi_category", "item_comment_status"]
    TE_FEATURES = [
        "city_code",
        "poi_tt_type_name_medium",
        "poi_tt_type_name_tiny",
        "music_id",
        "music_author_name",
        "poi_id",
        "diversification_id",
    ]
    FREQ_FEATURES = ["music_id", "music_author_name", "poi_id", "city_code"]
    TARGET_RAW = "play_count"
    TARGET = "log_play_count"

    DERIVED_FEATURES = [
        "desc_len",
        "desc_n_palabras",
        "desc_n_hashtags",
        "desc_n_menciones",
        "n_challenges",
        "es_duet",
        "no_vq",
        "dif_duracion_musica",
    ]
    NUM_PIPELINE_FEATURES = (
        NUMERICAL_FEATURES + BOOLEAN_FEATURES + DERIVED_FEATURES
    )
    FEATURES = (
        NUM_PIPELINE_FEATURES + OHE_FEATURES + TE_FEATURES + TEXT_FEATURES
    )

    COLUMNAS_LECTURA = (
        [TARGET_RAW, "duet_info_duet_from_id"]
        + BOOLEAN_FEATURES
        + NUMERICAL_FEATURES
        + TEXT_FEATURES
        + OHE_FEATURES
        + TE_FEATURES
    )
    # ***
    FAMILIAS = ("ridge", "random_forest", "lightgbm")

    EXPERIMENT_NAME = (
        "/Shared/pcd/otono-2026-proyecto/tiktok-engaging-estimator"
    )
    REGISTRY_URI = os.getenv("REGISTRY_URI")
    TRACKING_URI = os.getenv("TRACKING_URI")
    CATALOG = os.getenv("MLFLOW_CATALOG")
    SCHEMA = os.getenv("MLFLOW_SCHEMA")


@app.function
def get_best_family_run_id(
    family: str, *, experiment_name: str = EXPERIMENT_NAME
):
    """
    Busca la ejecución con el RMSLE más bajo para una 'family' específica de tags.
    """
    client = MlflowClient()

    # 1. Obtener el ID del experimento a partir de su nombre
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(
            f"No se encontró el experimento con el nombre: '{experiment_name}'"
        )

    experiment_id = experiment.experiment_id

    # 2. Construir el filtro para la etiqueta 'family'
    # Nota: Las cadenas dentro del filtro de MLflow utilizan comillas simples.
    # tags.current is provitional
    filter_string = f"tags.family = '{family}' and tags.current = 'true'"

    # 3. Buscar ejecuciones aplicando el filtro y ordenando por 'rmsle ASC' (el más bajo primero)
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["metrics.rmsle ASC"],
        max_results=1,
    )

    if runs_df.empty:
        print(
            f"No se encontraron ejecuciones con la etiqueta family='{family}'."
        )
        return None, None

    best_run = runs_df.iloc[0]
    best_run_id = best_run.run_id
    best_rmsle = best_run["metrics.rmsle"]
    best_r2 = best_run["metrics.r2"]
    best_run_name = client.get_run(best_run_id).info.run_name

    print(f"best model found for family: '{family}':")
    print(f" - Run ID: {best_run_id}")
    print(f" - Run NAME: {best_run_name}")
    print(f" - RMSLE: {best_rmsle}")
    print(f" - r2: {best_r2}")

    return best_run_id


@app.function
def get_run_model_params(run_id: int) -> dict:
    client = MlflowClient()

    params = client.get_run(run_id).data.params

    parsed_params = {}
    for k, v in params.items():
        try:
            # Intenta convertir a int primero
            parsed_params[k] = int(v)
        except ValueError:
            try:
                # Si falla, intenta convertir a float
                parsed_params[k] = float(v)
            except ValueError:
                # Si no es numérico, déjalo como string (ej. nombres de hiperparámetros textuales)
                parsed_params[k] = v
    return parsed_params


@app.cell
def _():
    run_id = get_best_family_run_id(FAMILIAS[1])
    get_run_model_params(run_id)
    return


@app.function(hide_code=True)
def build_best_model(family: str) -> Pipeline:
    run_id = get_best_family_run_id(family)
    model_params = get_run_model_params(run_id)
    pipeline = construir_pipeline(family, model_params)

    return {"family": family, "pipeline": pipeline, "params": model_params}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2 Particiones

    en el notebook anterior se realizo HPO sobre un sample (300,000 rows) de los datos y se entreno una unica vez  con los parametros encontrados y se evaluo.

    en este caso se ejecutara un run de estos mejores modelos  con `sample_data` (2,000,000 rows). De momento queda descatado usar el dataset completo por temas de computo
    """)
    return


@app.cell
def _():
    df = pd.read_parquet(RUTA_DATOS_COMPLETE)
    df_prepared = preparar_features(df)
    df_prepared
    return (df_prepared,)


@app.cell
def _(df_prepared):
    _estratos = pd.qcut(
        df_prepared[TARGET], q=10, labels=False, duplicates="drop"
    )

    train, test = train_test_split(
        df_prepared, test_size=0.20, stratify=_estratos, random_state=SEED
    )

    (train.shape, test.shape)
    return test, train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 3 Entrenar y logear modelo
    """)
    return


@app.function
def log_model(
    family: str,
    *,
    pipeline: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    metrics: dict,
    params: dict,
):

    # log model

    train_dataset, test_dataset = mlflow_data(
        [
            {
                "data": train,
                "name": "train-prod",
                "source": str(RUTA_DATOS_COMPLETE),
            },
            {
                "data": test,
                "name": "test-prod",
                "source": str(RUTA_DATOS_COMPLETE),
            },
        ]
    )

    df_head = test[FEATURES].head(5)
    signature = infer_signature(
        df_head, pipeline.predict(df_head[FEATURES])
    )

    with mlflow.start_run(run_name=f"{family}-estimator"):
        mlflow.log_input(train_dataset, context="production-train")
        mlflow.log_input(test_dataset, context="production-test")

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "family": family,
                "project": "TEE",
                "course": "proyecto-ciencia-datos",
                "term": "otono-2026",
                "role": "production-train",
            }
        )

        # TODO: do this framework agnostic
        mlflow_model = mlflow.sklearn.log_model(
            pipeline,
            name=f"{family}-estimator",
            signature=signature,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
        version = mlflow.register_model(
            model_uri=f"models:/{mlflow_model.model_id}",
            name=f"{CATALOG}.{SCHEMA}.{family}-estimator",
        )


@app.function
def main_flow(family: str, train: pd.DataFrame, test: pd.DataFrame):

    config_mlflow()

    print(f"[INFO]: building {family}")

    best_model = build_best_model(family)
    pipeline = best_model["pipeline"]
    params = best_model["params"]
    metrics = fit_evaluate(pipeline, train, test)

    print(f"[INFO]: {family}-r2:{metrics['r2']}")
    print(f"[INFO]: {family}-rmsle:{metrics['rmsle']}")

    # log model
    log_model(
        family,
        pipeline=pipeline,
        train=train,
        test=test,
        metrics=metrics,
        params=params,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Main run
    """)
    return


@app.cell
def _(test, train):
    for _family in FAMILIAS:
        main_flow(_family, train, test)
    return


if __name__ == "__main__":
    app.run()

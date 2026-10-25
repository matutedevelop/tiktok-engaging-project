import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", css_file="custom.css")

with app.setup:
    import json
    from collections import Counter
    from pathlib import Path
    from time import perf_counter
    import os
    from dotenv import load_dotenv
    from copy import deepcopy

    import marimo as mo
    import numpy as np
    import optuna
    import pandas as pd
    from lightgbm import LGBMRegressor
    from optuna.samplers import GridSampler, TPESampler
    from scipy.stats import spearmanr
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import (
        mean_absolute_error,
        r2_score,
        root_mean_squared_error,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        OneHotEncoder,
        StandardScaler,
        TargetEncoder,
    )
    import mlflow
    from mlflow.models import infer_signature

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    load_dotenv()

    SEED = 42
    N_FILAS = 300_000
    HACER_TUNING = True

    ROOT = Path(__file__).parent.parent
    RUTA_DATOS = ROOT / "data" / "interim" / "sample_data.parquet"
    RUTA_PARAMETROS = ROOT / "artifacts" / "03-mejores-parametros.json"
    RUTA_RESULTADOS = ROOT / "artifacts" / "03-resultados-test.json"

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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Experimentos: tres familias con HPO

    En este notebook entrenamos y comparamos tres familias de modelos de regresión para estimar
    el alcance de un video de TikTok con información disponible **antes** de publicarlo:

    1. **Lineal regularizado** — `Ridge`.
    2. **Bagging de árboles** — `RandomForestRegressor`.
    3. **Gradient boosting** — `LGBMRegressor` (LightGBM).

    Las tres familias comparten los mismos datos, las mismas particiones, el mismo
    preprocesamiento y la misma métrica. Cada una tiene su propia búsqueda de hiperparámetros con
    Optuna; sólo cambia el estimador final dentro del `Pipeline`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Target y métrica

    - **Target:** $y = \ln(1 + \text{play\_count})$. El EDA mostró que `play_count` tiene una
      cola muy pesada; en escala log los errores relativos pesan igual para un video de $10^3$ que
      para uno de $10^6$ reproducciones.
    - **Métrica principal:** RMSE sobre $y$, es decir **RMSLE** de `play_count`. Se **minimiza**.
      Es la métrica que optimizan las funciones objetivo de Optuna.
    - **Métricas secundarias** (sólo se reportan, no se optimizan): $R^2$ en escala log
      (fracción de la varianza de $y$ explicada), MAE en escala log y correlación de Spearman,
      que mide si el modelo *ordena* bien los videos.

    Un RMSLE de 1 significa que, en promedio cuadrático, la predicción se equivoca por un factor
    de $e \approx 2.7$ en reproducciones.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## 2. Datos

    Partimos de `data/interim/sample_data.parquet`, la salida de `02-preparacion-datos.py`
    (`drop_columns → fix_dtypes → numerical_imputation → cat_imputation → drop_nulls`). Ese
    archivo ya tiene los tipos correctos y no contiene nulos.

    Sobre él sólo aplicamos:

    1. lectura de las columnas a priori + target;
    2. submuestra fija de **{N_FILAS:,}** filas (`random_state={SEED}`) para que la búsqueda de
       Random Forest sea viable en una laptop.

    **Limitación conocida:** `02` imputa las numéricas con la mediana/media de toda la muestra
    antes de dividirla. Como los nulos de `duration`, `music_duration` y `vq_score` son
    marginales (ver la salida de `numerical_imputation` en `02`), el efecto sobre la evaluación
    es despreciable; todo el preprocesamiento que sí aprende de los datos en este notebook
    (encoders, escalado, imputación) se ajusta únicamente con train.
    """)
    return


@app.cell
def _():
    df_interim = pd.read_parquet(RUTA_DATOS, columns=COLUMNAS_LECTURA)
    df_muestra = df_interim.sample(
        n=min(N_FILAS, len(df_interim)), random_state=SEED
    ).reset_index(drop=True)
    print(f"filas en data/interim: {len(df_interim):,}")
    print(f"filas en la submuestra: {len(df_muestra):,}")
    del df_interim
    return (df_muestra,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Features

    Partimos de las variables que dejó `02-preparacion-datos.py` y construimos features
    **fila por fila** (sin estadísticas del conjunto, así que no hay riesgo de fuga):

    | Grupo | Columnas | Tratamiento en el `Pipeline` |
    |---|---|---|
    | Numéricas | `duration`, `music_duration`, `vq_score` | imputación por mediana + estandarización |
    | Booleanas | `duet_enabled`, `is_ad`, `item_mute`, `stitch_enabled`, `music_original` | 0/1 + estandarización |
    | Texto → conteos | `desc_len`, `desc_n_palabras`, `desc_n_hashtags`, `desc_n_menciones`, `n_challenges` | igual que numéricas |
    | Banderas | `es_duet` (`duet_info_duet_from_id` distinto de 0), `no_vq` (`vq_score == 0`), `dif_duracion_musica` | igual que numéricas |
    | Nominales baja cardinalidad | `poi_category`, `item_comment_status` | `OneHotEncoder` (máx. 20 categorías, resto en *infrequent*) |
    | Nominales alta cardinalidad | `city_code`, `poi_tt_type_name_medium`, `poi_tt_type_name_tiny`, `music_id`, `music_author_name`, `poi_id`, `diversification_id` | `TargetEncoder` con *cross-fitting* interno de 5 folds |
    | Frecuencia | `music_id`, `music_author_name`, `poi_id`, `city_code` | `FrequencyEncoder`: $\log(1 + \text{veces en train})$ + estandarización |
    | Popularidad de hashtags | `challenges` | `HashtagFrequency`: media y máximo de $\log(1 + \text{frecuencia en train})$ de los hashtags del video y fracción de hashtags no vistos |
    | Texto de hashtags | `challenges` | TF-IDF sobre los hashtags (`min_df=5`, 20k términos) → `TruncatedSVD` a 50 componentes |
    | Texto de la descripción | `desc` | TF-IDF de palabras (`min_df=5`, 20k términos) → `TruncatedSVD` a 30 componentes |

    Decisiones:

    - **Frecuencia como señal de tendencia.** El `TargetEncoder` casi no aporta para ids que
      aparecen pocas veces (se encoge a la media global). Contar cuántos videos usan una canción,
      un autor de audio, un lugar o un hashtag sí distingue lo que es tendencia de lo que no, y
      llega también a las categorías raras. No usa el target y los conteos salen sólo de train.
    - **Texto.** `desc` y `challenges` se vectorizan con TF-IDF y se reducen con SVD (LSA) para
      que las tres familias reciban una matriz densa de tamaño manejable. TF-IDF y SVD se
      ajustan en cada `fit`, es decir, sólo con train. `min_df=5` descarta términos que casi
      no aparecen y no se pueden aprender. También conservamos los conteos simples.
    - `address` y `poi_name` se omiten porque describen el mismo lugar que `poi_id`; `music_title`
      se omite en favor de `music_id`.
    - `duet_info_duet_from_id` es el id del video original (0 si no es dueto); sólo usamos si es
      dueto (`es_duet`).
    - `TargetEncoder` usa *cross-fitting* durante `fit`, así que la codificación de cada fila de
      train no usa su propio target. Categorías no vistas en train reciben la media global.
    - El mismo preprocesamiento se usa en las tres familias, aunque LightGBM podría manejar
      categóricas de forma nativa, para que la comparación dependa sólo del estimador.
    """)
    return


@app.function
def parsear_challenges(valor) -> list[str]:
    """Convierte la lista JSON de `challenges` en una lista de hashtags en minúsculas."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return []
    texto = str(valor).strip()
    if texto in ("", "[]"):
        return []
    try:
        lista = json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        lista = texto.strip("[]").replace('"', "").split(",")
    if not isinstance(lista, list):
        lista = [lista]
    return [str(h).strip().lower() for h in lista if str(h).strip()]


@app.function
def contar_challenges(valor) -> int:
    return len(parsear_challenges(valor))


@app.function
# ***
# TODO: turn this function into a match case to allow to have diferent preparations
def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transformaciones sin estado: no aprenden nada de los datos."""
    _df = pd.DataFrame(index=df.index)

    for _c in BOOLEAN_FEATURES:
        _df[_c] = df[_c].astype(float)

    for _c in NUMERICAL_FEATURES:
        _df[_c] = pd.to_numeric(df[_c], errors="coerce")

    _desc = df["desc"].astype("string").fillna("")
    _df["desc_len"] = _desc.str.len().astype(float)
    _df["desc_n_palabras"] = (
        _desc.str.split().str.len().fillna(0).astype(float)
    )
    _df["desc_n_hashtags"] = _desc.str.count(r"#\w+").astype(float)
    _df["desc_n_menciones"] = _desc.str.count(r"@\w+").astype(float)
    _df["n_challenges"] = (
        df["challenges"].map(contar_challenges).astype(float)
    )
    _duet_from = pd.to_numeric(
        df["duet_info_duet_from_id"].astype(str), errors="coerce"
    )
    _df["es_duet"] = (_duet_from.fillna(0) != 0).astype(float)
    _df["no_vq"] = (_df["vq_score"] == 0).astype(float)
    _df["dif_duracion_musica"] = (
        _df["duration"] - _df["music_duration"]
    ).abs()

    for _c in OHE_FEATURES + TE_FEATURES:
        _df[_c] = df[_c].astype(str)

    # Texto crudo: lo vectorizan TF-IDF y la frecuencia de hashtags dentro del Pipeline
    _df["desc"] = _desc.astype(str)
    _df["challenges"] = (
        df["challenges"].astype("string").fillna("[]").astype(str)
    )

    _df[TARGET] = np.log1p(df[TARGET_RAW].astype(float))
    return _df[FEATURES + [TARGET]]


@app.cell
def _(df_muestra):
    datos = preparar_features(df_muestra)
    datos.head()
    return (datos,)


@app.cell
def _(datos):
    datos[FEATURES].describe(include="all").T
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Particiones

    División **80 / 10 / 10**, estratificada por deciles de $y$ para que las tres particiones
    tengan la misma distribución del target, con `random_state=42`:

    | Partición | Uso |
    |---|---|
    | **train** (80 %) | ajustar cada trial de Optuna (incluido el preprocesamiento) |
    | **validación** (10 %) | evaluar cada trial; es lo único que ve la función objetivo |
    | **test** (10 %) | evaluación final, **una sola vez**, después de fijar hiperparámetros |

    Una vez elegidos los hiperparámetros, cada familia se reentrena con train + validación y se
    evalúa en test. El test nunca aparece en la firma de las funciones de HPO.
    """)
    return


@app.cell
def _(datos):
    _estratos = pd.qcut(datos[TARGET], q=10, labels=False, duplicates="drop")
    desarrollo, test = train_test_split(
        datos, test_size=0.10, stratify=_estratos, random_state=SEED
    )
    _estratos_dev = pd.qcut(
        desarrollo[TARGET], q=10, labels=False, duplicates="drop"
    )
    train, validation = train_test_split(
        desarrollo, test_size=1 / 9, stratify=_estratos_dev, random_state=SEED
    )

    pd.DataFrame(
        {
            "filas": [len(train), len(validation), len(test)],
            "media_y": [
                train[TARGET].mean(),
                validation[TARGET].mean(),
                test[TARGET].mean(),
            ],
            "std_y": [
                train[TARGET].std(),
                validation[TARGET].std(),
                test[TARGET].std(),
            ],
        },
        index=["train", "validation", "test"],
    )
    return desarrollo, test, train, validation


@app.function
def evaluate_num_pred(y_real, y_pred) -> dict:
    # TODO create  a cat_evaluate
    return {
        "rmsle": root_mean_squared_error(y_real, y_pred),
        "r2": r2_score(y_real, y_pred),
        "mae_log": mean_absolute_error(y_real, y_pred),
        "spearman": spearmanr(y_real, y_pred).statistic,
    }


@app.function
def evaluate_cat_pred(y_real, y_pred) -> dict:
    return NotImplemented


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Baselines triviales

    Antes de cualquier modelo medimos el piso: predecir la media, la mediana o la mediana por
    `poi_category`, todas calculadas **sólo con train**. Un modelo que no supere estos valores
    no aporta nada. Por construcción, la media tiene $R^2 \approx 0$.
    """)
    return


@app.cell
def _(test, train, validation):
    _media = train[TARGET].mean()
    _mediana = train[TARGET].median()
    _mediana_cat = train.groupby("poi_category")[TARGET].median()

    def _pred_cat(df):
        return df["poi_category"].map(_mediana_cat).fillna(_mediana)

    _filas = []
    for _nombre, _pred in [
        ("media", lambda df: np.full(len(df), _media)),
        ("mediana", lambda df: np.full(len(df), _mediana)),
        ("mediana_por_poi_category", _pred_cat),
    ]:
        for _part, _df in [("validacion", validation), ("test", test)]:
            _m = evaluate_num_pred(_df[TARGET], _pred(_df))
            _filas.append({"baseline": _nombre, "particion": _part, **_m})

    baselines = pd.DataFrame(_filas)
    baselines
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Pipeline compartido

    Una sola función construye el `Pipeline` de las tres familias. El `ColumnTransformer` es
    idéntico; sólo cambia el estimador final. Todo lo que aprende de los datos (medianas,
    medias y desviaciones, categorías del one-hot, codificación por target, conteos de
    frecuencia, vocabulario TF-IDF y componentes SVD) vive dentro del `Pipeline`, así que en
    cada `fit` se ajusta únicamente con las filas de entrenamiento.

    `FrequencyEncoder` y `HashtagFrequency` son transformers propios, compatibles con
    scikit-learn (`fit` aprende los conteos y `transform` los aplica).
    """)
    return


@app.class_definition
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Reemplaza cada categoría por log(1 + veces que aparece en train).

    No usa el target. Categorías no vistas en train reciben 0.
    """

    def fit(self, X, y=None):
        _X = pd.DataFrame(X)
        self.conteos_ = [_X[c].value_counts() for c in _X.columns]
        self.n_features_in_ = _X.shape[1]
        return self

    def transform(self, X):
        _X = pd.DataFrame(X)
        _salida = [
            np.log1p(_X[c].map(_conteo).fillna(0).to_numpy(dtype=float))
            for c, _conteo in zip(_X.columns, self.conteos_)
        ]
        return np.column_stack(_salida)


@app.class_definition
class HashtagFrequency(BaseEstimator, TransformerMixin):
    """Popularidad de los hashtags de cada video según cuántas veces aparecen en train.

    Devuelve, por video: media y máximo de log(1 + frecuencia) de sus hashtags y la
    fracción de sus hashtags que no aparecen en train. No usa el target.
    """

    def fit(self, X, y=None):
        _conteo = Counter()
        for _valor in np.asarray(X).ravel():
            _conteo.update(parsear_challenges(_valor))
        self.conteos_ = _conteo
        return self

    def transform(self, X):
        _filas = []
        for _valor in np.asarray(X).ravel():
            _hashtags = parsear_challenges(_valor)
            if not _hashtags:
                _filas.append((0.0, 0.0, 0.0))
                continue
            _frec = np.log1p([self.conteos_.get(h, 0) for h in _hashtags])
            _filas.append(
                (_frec.mean(), _frec.max(), float(np.mean(_frec == 0)))
            )
        return np.asarray(_filas, dtype=float)


@app.function
def construir_pipeline(familia: str, parametros: dict) -> Pipeline:
    # TODO: turn this function into a match case to allow having diferent preprocessing pipelines
    preprocesamiento = ColumnTransformer(
        [
            (
                "frecuencias",
                Pipeline(
                    [
                        ("contar", FrequencyEncoder()),
                        ("escalar", StandardScaler()),
                    ]
                ),
                FREQ_FEATURES,
            ),
            (
                "popularidad_hashtags",
                Pipeline(
                    [
                        ("contar", HashtagFrequency()),
                        ("escalar", StandardScaler()),
                    ]
                ),
                "challenges",
            ),
            (
                "texto_challenges",
                Pipeline(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer=parsear_challenges,
                                min_df=5,
                                max_features=20_000,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "svd",
                            TruncatedSVD(
                                n_components=50, random_state=SEED
                            ),
                        ),
                        ("escalar", StandardScaler()),
                    ]
                ),
                "challenges",
            ),
            (
                "texto_desc",
                Pipeline(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(
                                min_df=5,
                                max_features=20_000,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "svd",
                            TruncatedSVD(
                                n_components=30, random_state=SEED
                            ),
                        ),
                        ("escalar", StandardScaler()),
                    ]
                ),
                "desc",
            ),
            (
                "numericas",
                Pipeline(
                    [
                        # ("imputar", SimpleImputer(strategy="median")), # Esto no es necesario desde que los datos fueron previamente imputados
                        ("escalar", StandardScaler()),
                    ]
                ),
                NUM_PIPELINE_FEATURES,
            ),
            (
                "baja_cardinalidad",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=20,
                    sparse_output=False,
                ),
                OHE_FEATURES,
            ),
            (
                "alta_cardinalidad",
                TargetEncoder(
                    target_type="continuous",
                    cv=5,
                    shuffle=True,
                    random_state=SEED,
                ),
                TE_FEATURES,
            ),
        ],
        sparse_threshold=0,
    )
    # ***
    estimadores = {
        "ridge": Ridge,
        "random_forest": RandomForestRegressor,
        "lightgbm": LGBMRegressor,
    }
    if familia not in estimadores:
        raise ValueError(f"Familia no reconocida: {familia}")
    return Pipeline(
        [
            ("preprocessing", preprocesamiento),
            ("model", estimadores[familia](**parametros)),
        ]
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Espacios de búsqueda y presupuesto

    | Familia | Hiperparámetro | Rango / valores | Estrategia | Trials |
    |---|---|---|---|---:|
    | Ridge | `alpha` | 13 valores en `logspace(-1, 5)` | búsqueda exhaustiva (`GridSampler`) | 13 |
    | Random Forest | `n_estimators` | 50–200, paso 50 | TPE | 12 |
    | | `max_depth` | 8–24 | | |
    | | `min_samples_leaf` | 2–100 (log) | | |
    | | `max_features` | `sqrt`, 0.3, 0.5 | | |
    | | `max_samples` | 0.3–1.0 | | |
    | LightGBM | `n_estimators` | 200–1000, paso 100 | TPE | 25 |
    | | `learning_rate` | 0.02–0.2 (log) | | |
    | | `num_leaves` | 15–255 (log) | | |
    | | `min_child_samples` | 20–300 (log) | | |
    | | `subsample` | 0.5–1.0 (`subsample_freq=1`) | | |
    | | `colsample_bytree` | 0.5–1.0 | | |
    | | `reg_lambda` | 1e-4–10 (log) | | |

    Justificación:

    - **Ridge** tiene un solo hiperparámetro relevante y entrenar es barato, así que recorremos
      la rejilla completa en escala log en lugar de muestrear. En una primera corrida con
      `logspace(-3, 3)` el mejor valor fue `alpha=1000`, el extremo de la rejilla, por lo que
      la desplazamos a `logspace(-1, 5)` para que el óptimo quede dentro del rango.
    - **Random Forest** es la familia más cara. Limitamos `n_estimators` a 200 y usamos
      `max_samples` (fracción del bootstrap) para que cada árbol vea sólo una parte de las filas.
      En la primera corrida los mejores valores quedaron en los bordes (`min_samples_leaf=5`,
      `max_samples≈0.79`), así que ampliamos esos rangos a 2–100 y 0.3–1.0.
    - **LightGBM** es rápido con 240k filas, por lo que le damos más trials. `num_leaves` y
      `min_child_samples` controlan la complejidad de cada árbol; `subsample`,
      `colsample_bytree` y `reg_lambda` regularizan en un target ruidoso. `n_estimators` se busca
      junto con `learning_rate` porque ambos se compensan; no usamos *early stopping* porque
      requeriría pasar un `eval_set` que el `Pipeline` no transforma.
    - En las búsquedas con TPE fijamos `seed=42` y `n_startup_trials=5`: los primeros 5 trials son
      aleatorios y generan la evidencia inicial antes de que TPE modele qué regiones son mejores.
    - Las semillas de los estimadores (`random_state=42`) también se fijan.
    """)
    return


@app.function
def proponer_parametros(trial: optuna.Trial, familia: str) -> dict:
    # ***
    if familia == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-1, 1e5, log=True)}
    if familia == "random_forest":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 200, step=50
            ),
            "max_depth": trial.suggest_int("max_depth", 8, 24),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 2, 100, log=True
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", 0.3, 0.5]
            ),
            "max_samples": trial.suggest_float("max_samples", 0.3, 1.0),
            "random_state": SEED,
            "n_jobs": -1,
        }
    if familia == "lightgbm":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 200, 1000, step=100
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.2, log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", 15, 255, log=True
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 20, 300, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-4, 10, log=True
            ),
            "subsample_freq": 1,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
    raise ValueError(f"Familia no reconocida: {familia}")


@app.function
def completar_parametros(familia: str, mejores: dict) -> dict:
    """Agrega los parámetros fijos que no forman parte del espacio de búsqueda."""
    # ***
    fijos = {
        "ridge": {},
        "random_forest": {"random_state": SEED, "n_jobs": -1},
        "lightgbm": {
            "subsample_freq": 1,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        },
    }
    return {**mejores, **fijos[familia]}


@app.function
def crear_estudio(familia: str) -> tuple[optuna.Study, int, str]:
    # ***
    if familia == "ridge":
        rejilla = {"alpha": np.logspace(-1, 5, 13).tolist()}
        sampler = GridSampler(rejilla, seed=SEED)
        sampler_name = "optuna-grid"
        return (
            optuna.create_study(direction="minimize", sampler=sampler),
            13,
            sampler_name,
        )

    n_trials = {"random_forest": 12, "lightgbm": 25}[familia]
    sampler = TPESampler(seed=SEED, n_startup_trials=5)
    sampler_name = "optuna-TPE"
    return (
        optuna.create_study(direction="minimize", sampler=sampler),
        n_trials,
        sampler_name,
    )


@app.function
def fit_evaluate(
    pipeline: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    is_regresion: bool = True,
) -> dict:
    """if test is not provided it will evaluate results on training set"""
    # TODO: do this framework agnostic
    evaluate = evaluate_num_pred if is_regresion else evaluate_cat_pred

    evaluation_data = train if test is None else test

    start = perf_counter()
    pipeline.fit(train[FEATURES], train[TARGET])
    training_time = perf_counter() - start
    y_pred = pipeline.predict(evaluation_data[FEATURES])
    results = evaluate(evaluation_data[TARGET], y_pred)

    return results | {"training_time": training_time}


@app.function
def config_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(REGISTRY_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


@app.function
def mlflow_data(
    dataframes_info: dict[str, pd.DataFrame], target: str = TARGET
):

    _datasets = []

    for _df_info in dataframes_info:
        _dataset = mlflow.data.from_pandas(
            _df_info["data"],
            name=_df_info["name"],
            targets=target,
            source=_df_info["source"],
        )
        _datasets.append(_dataset)
    return _datasets


@app.function
def ajustar_familia(
    familia: str, train: pd.DataFrame, validation: pd.DataFrame
) -> dict:
    """HPO de una familia. train and log"""
    # TODO: Handle HPO flag
    # ***
    study, n_trials, sampler = crear_estudio(familia)
    dataset_train, dataset_validation = mlflow_data(
        [
            {
                "data": train,
                "name": "train",
                "source": str(RUTA_DATOS),
            },
            {
                "data": validation,
                "name": "validation",
                "source": str(RUTA_DATOS),
            },
        ]
    )

    # create fun to optimize
    def objective(trial: optuna.Trial) -> float:
        parametros = proponer_parametros(trial, familia)
        pipeline = construir_pipeline(familia, parametros)

        metrics = fit_evaluate(pipeline, train, validation)

        with mlflow.start_run(
            run_name=f"trial-{trial.number:03d}", nested=True
        ) as parent:
            mlflow.log_params(pipeline.named_steps["model"].get_params())
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "project": "TEE",
                    "course": "proyecto-ciencia-datos",
                    "term": "otono-2026",
                    "role": "tuning-trial",
                    "family": familia,
                }
            )

        return metrics["rmsle"]

    # main study
    with mlflow.start_run(run_name=f"tuning-{familia}") as parent:
        mlflow.log_input(dataset_train, context="training")
        mlflow.log_input(dataset_validation, context="validation")
        mlflow.log_params({"sampler": sampler, "n_trials": n_trials})
        mlflow.set_tags(
            {
                "project": "TEE",
                "course": "proyecto-ciencia-datos",
                "term": "otono-2026",
                "role": "tuning-study",
                "family": familia,
            }
        )

        study.optimize(objective, n_trials=n_trials)

        # log results

        mlflow.log_params(
            {f"best_{k}": v for k, v in study.best_params.items()}
        )
        mlflow.log_metric("best_rmsle", study.best_value)

        # return unfitted pipeline with best params
        final_pipeline = construir_pipeline(familia, study.best_params)

    return {
        "family": familia,
        "params": study.best_params,
        "pipeline": final_pipeline,
        "n_trials": len(study.trials),
    }


@app.function
def run_best_model(
    # This is kinda redundant and more useful to notebook 5
    model_spec: dict,
    dev_data: pd.DataFrame,
    test: pd.DataFrame,
    model_eval: dict,
):
    """The pipeline is trained once with the complete of training data (train + val) and logged as a single run in mlflow"""

    pipeline = model_spec["pipeline"]

    df_head = test.head(5)
    signature = infer_signature(df_head, pipeline.predict(df_head))

    dataset_dev, dataset_test = mlflow_data(
        [
            {
                "data": dev_data,
                "name": "development",
                "source": str(RUTA_DATOS),
            },
            {
                "data": test,
                "name": "test",
                "source": str(RUTA_DATOS),
            },
        ]
    )

    with mlflow.start_run(
        run_name=f"{model_spec['family']}-final"
    ) as final:
        mlflow.log_input(dataset_dev)
        mlflow.log_input(dataset_test)
        mlflow.log_params(model_spec["params"])

        mlflow.log_metrics(model_eval)
        mlflow.set_tags(
            {
                "project": "TEE",
                "course": "proyecto-ciencia-datos",
                "term": "otono-2026",
                "role": "evaluation-run",
                "family": model_spec["family"],
            }
        )

        # TODO: do this framework agnostic
    # mlflow.sklearn.log_model(
    #     pipeline,
    #     name="model",
    #     input_example=df_head,
    #     signature=signature,
    #     serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
    # )


@app.function
def run_experiment(
    familia: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
):
    dev_data = pd.concat([train, validation]).reset_index(drop=True)
    config_mlflow()
    best_model_spec = ajustar_familia(familia, train, validation)
    best_model_test_results = fit_evaluate(
        best_model_spec["pipeline"], dev_data, test
    )
    run_best_model(
        best_model_spec, dev_data, test, best_model_test_results
    )

    return best_model_spec | best_model_test_results


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## 8. Búsqueda de hiperparámetros y experiment tracking

    `HACER_TUNING = {HACER_TUNING}`. Con `True` se ejecutan los tres estudios y se guardan los
    mejores parámetros (con su RMSLE y $R^2$ de validación) en
    `artifacts/03-mejores-parametros.json` (ignorado por git). Con `False` se reutiliza ese
    archivo para no repetir los trials al volver a correr el notebook.
    """)
    return


@app.cell
def _(desarrollo, test, validation):
    estudios = {}
    if HACER_TUNING:
        for _familia in FAMILIAS:
            _inicio = perf_counter()
            estudios[_familia] = run_experiment(
                _familia, desarrollo, validation, test
            )
            print(
                f"[{_familia}] {estudios[_familia]['n_trials']} trials, "
                f"mejor RMSLE validación = {estudios[_familia]['rmsle']:.4f}, "
                f"mejor r2 validación = {estudios[_familia]['r2']:.4f}, "
                f"tiempo = {perf_counter() - _inicio:.0f} s"
            )
        # _mejores = {
        #    _f: {
        #        "params": _e.best_trial.params,
        #        "best_trial": _e.best_trial.number,
        #        "rmsle_validacion": _e.best_value,
        #        "r2_validacion": _e.best_trial.user_attrs["r2_validacion"],
        #        "n_trials": len(_e.trials),
        #    }
        #    for _f, _e in estudios.items()
        # }
        # RUTA_PARAMETROS.parent.mkdir(parents=True, exist_ok=True)
        # RUTA_PARAMETROS.write_text(
        #    json.dumps(_mejores, indent=2), encoding="utf-8"
        # )
    # resumen_hpo = json.loads(RUTA_PARAMETROS.read_text(encoding="utf-8"))
    return (estudios,)


@app.cell
def _(estudios):
    _estudios = deepcopy(estudios)
    list(
        map(lambda d: d.pop("pipeline"), _estudios.values())
    )  # quitar el pipeline de sklearn de la tabla

    pd.DataFrame(
        {k: v for k, v in _estudios.items()},
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9. Modelos finales y evaluación en test

    Para cada familia: se reconstruye el `Pipeline` con sus mejores parámetros, se entrena con
    **train + validación** y se evalúa **una vez** en test. El resultado de test no regresa a la
    búsqueda. Las métricas finales se guardan en `artifacts/03-resultados-test.json`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Interpretación

    > **Pendiente de redactar con los resultados de la corrida:** comparar las tres familias
    > (diferencia de RMSLE en factor multiplicativo de reproducciones, $R^2$, Spearman y tiempo),
    > justificar si la mejora del modelo seleccionado compensa su costo frente al segundo lugar,
    > y señalar dos limitaciones (submuestra de 300k filas, presupuesto pequeño de trials,
    > texto reducido a 80 componentes LSA, imputación de `02` con toda la muestra).
    """)
    return


if __name__ == "__main__":
    app.run()

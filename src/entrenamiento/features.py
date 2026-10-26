"""Preparación sin estado y `Pipeline` de scikit-learn compartido por las familias.

Los transformers viven aquí (y no en un notebook) para que el pickle registrado en
MLflow los encuentre como `entrenamiento.features.*` al cargarse desde la API.
"""

import json
from collections import Counter

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)

from entrenamiento.configuracion import (
    BOOLEAN_FEATURES,
    FEATURES,
    FREQ_FEATURES,
    NUM_PIPELINE_FEATURES,
    NUMERICAL_FEATURES,
    OHE_FEATURES,
    SEED,
    TARGET,
    TARGET_RAW,
    TE_FEATURES,
)


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


def contar_challenges(valor) -> int:
    return len(parsear_challenges(valor))


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transformaciones sin estado: no aprenden nada de los datos.

    Si `df` trae `play_count` también calcula el target `log_play_count`.
    """
    salida = pd.DataFrame(index=df.index)

    for c in BOOLEAN_FEATURES:
        salida[c] = df[c].astype(float)

    for c in NUMERICAL_FEATURES:
        salida[c] = pd.to_numeric(df[c], errors="coerce")

    desc = df["desc"].astype("string").fillna("")
    salida["desc_len"] = desc.str.len().astype(float)
    salida["desc_n_palabras"] = desc.str.split().str.len().fillna(0).astype(float)
    salida["desc_n_hashtags"] = desc.str.count(r"#\w+").astype(float)
    salida["desc_n_menciones"] = desc.str.count(r"@\w+").astype(float)
    salida["n_challenges"] = df["challenges"].map(contar_challenges).astype(float)
    duet_from = pd.to_numeric(df["duet_info_duet_from_id"].astype(str), errors="coerce")
    salida["es_duet"] = (duet_from.fillna(0) != 0).astype(float)
    salida["no_vq"] = (salida["vq_score"] == 0).astype(float)
    salida["dif_duracion_musica"] = (salida["duration"] - salida["music_duration"]).abs()

    for c in OHE_FEATURES + TE_FEATURES:
        salida[c] = df[c].astype(str)

    # Texto crudo: lo vectorizan TF-IDF y la frecuencia de hashtags dentro del Pipeline
    salida["desc"] = desc.astype(str)
    salida["challenges"] = df["challenges"].astype("string").fillna("[]").astype(str)

    if TARGET_RAW not in df.columns:
        return salida[FEATURES]
    salida[TARGET] = np.log1p(df[TARGET_RAW].astype(float))
    return salida[FEATURES + [TARGET]]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Reemplaza cada categoría por log(1 + veces que aparece en train).

    No usa el target. Categorías no vistas en train reciben 0.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.conteos_ = [X[c].value_counts() for c in X.columns]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        salida = [
            np.log1p(X[c].map(conteo).fillna(0).to_numpy(dtype=float))
            for c, conteo in zip(X.columns, self.conteos_)
        ]
        return np.column_stack(salida)


class HashtagFrequency(BaseEstimator, TransformerMixin):
    """Popularidad de los hashtags de cada video según cuántas veces aparecen en train.

    Devuelve, por video: media y máximo de log(1 + frecuencia) de sus hashtags y la
    fracción de sus hashtags que no aparecen en train. No usa el target.
    """

    def fit(self, X, y=None):
        conteo = Counter()
        for valor in np.asarray(X).ravel():
            conteo.update(parsear_challenges(valor))
        self.conteos_ = conteo
        return self

    def transform(self, X):
        filas = []
        for valor in np.asarray(X).ravel():
            hashtags = parsear_challenges(valor)
            if not hashtags:
                filas.append((0.0, 0.0, 0.0))
                continue
            frec = np.log1p([self.conteos_.get(h, 0) for h in hashtags])
            filas.append((frec.mean(), frec.max(), float(np.mean(frec == 0))))
        return np.asarray(filas, dtype=float)


def _texto(n_componentes: int, **tfidf) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=5, max_features=20_000, sublinear_tf=True, **tfidf)),
            ("svd", TruncatedSVD(n_components=n_componentes, random_state=SEED)),
            ("escalar", StandardScaler()),
        ]
    )


def construir_preprocesamiento() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "frecuencias",
                Pipeline([("contar", FrequencyEncoder()), ("escalar", StandardScaler())]),
                FREQ_FEATURES,
            ),
            (
                "popularidad_hashtags",
                Pipeline([("contar", HashtagFrequency()), ("escalar", StandardScaler())]),
                "challenges",
            ),
            ("texto_challenges", _texto(50, analyzer=parsear_challenges), "challenges"),
            ("texto_desc", _texto(30), "desc"),
            # 02 ya imputó las numéricas, así que sólo se escalan
            ("numericas", StandardScaler(), NUM_PIPELINE_FEATURES),
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
                    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
                ),
                TE_FEATURES,
            ),
        ],
        sparse_threshold=0,
    )


ESTIMADORES = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "lightgbm": LGBMRegressor,
}


def construir_pipeline(familia: str, parametros: dict) -> Pipeline:
    if familia not in ESTIMADORES:
        raise ValueError(f"Familia no reconocida: {familia}")
    return Pipeline(
        [
            ("preprocessing", construir_preprocesamiento()),
            ("model", ESTIMADORES[familia](**parametros)),
        ]
    )


def pipeline_de_inferencia(entrenado: Pipeline) -> Pipeline:
    """Antepone `preparar_features` a un pipeline ya entrenado, sin volver a ajustarlo.

    El modelo registrado recibe las columnas crudas de `data/interim` (`COLUMNAS_ENTRADA`)
    y aplica exactamente la misma preparación que se usó al entrenar.
    """
    return Pipeline([("preparar", FunctionTransformer(preparar_features)), *entrenado.steps])


def normalizar_entrada(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte `category`/`string` a `str` para que MLflow pueda inferir la firma."""
    tipos = {
        c: str
        for c in df.columns
        if isinstance(df[c].dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(df[c])
    }
    return df.astype(tipos)

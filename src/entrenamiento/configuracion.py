"""Constantes compartidas por el entrenamiento, el flow y la inferencia."""

import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

SEED = 42
N_FILAS = 300_000

ROOT = Path(__file__).resolve().parents[2]
RUTA_DATOS = ROOT / "data" / "interim" / "sample_data.parquet"
RUTA_PARAMETROS = ROOT / "artifacts" / "05-mejores-parametros.json"

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
NUM_PIPELINE_FEATURES = NUMERICAL_FEATURES + BOOLEAN_FEATURES + DERIVED_FEATURES
FEATURES = NUM_PIPELINE_FEATURES + OHE_FEATURES + TE_FEATURES + TEXT_FEATURES

TARGET_RAW = "play_count"
TARGET = "log_play_count"
COLUMNAS_LECTURA = (
    [TARGET_RAW, "duet_info_duet_from_id"]
    + BOOLEAN_FEATURES
    + NUMERICAL_FEATURES
    + TEXT_FEATURES
    + OHE_FEATURES
    + TE_FEATURES
)
# Columnas que recibe el modelo registrado: las de lectura sin el target
COLUMNAS_ENTRADA = [c for c in COLUMNAS_LECTURA if c != TARGET_RAW]

FAMILIAS = ("ridge", "random_forest", "lightgbm")
ALIASES = ("champion", "challenger", "candidate")

PROYECTO = "tiktok-engaging-estimator"
EXPERIMENTO = f"/Shared/pcd-otono-2026-proyecto-{PROYECTO}"
TAGS_PROYECTO = {
    "project": "TEE",
    "course": "proyecto-ciencia-datos",
    "term": "otono-2026",
}


def nombre_familia(familia: str) -> str:
    """`random_forest` -> `random-forest`, para los nombres de los runs."""
    return familia.replace("_", "-")


def nombre_modelo_registrado() -> str:
    catalogo = os.getenv("MLFLOW_CATALOG", "workspace")
    esquema = os.getenv("MLFLOW_SCHEMA", "default")
    return f"{catalogo}.{esquema}.pcd_otono_2026_{PROYECTO.replace('-', '_')}"


def configurar_mlflow() -> dict[str, str]:
    """Lee `.env` y apunta MLflow a Databricks (o a lo que digan las variables)."""
    load_dotenv(ROOT / ".env")
    tracking_uri = os.getenv("TRACKING_URI", "databricks")
    registry_uri = os.getenv("REGISTRY_URI", "databricks-uc")
    if tracking_uri == "databricks" and not (
        os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN")
    ):
        raise ValueError("Faltan DATABRICKS_HOST o DATABRICKS_TOKEN en .env")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    experimento = mlflow.set_experiment(EXPERIMENTO)
    return {
        "tracking_uri": tracking_uri,
        "registry_uri": registry_uri,
        "experiment_id": experimento.experiment_id,
        "experiment_name": experimento.name,
    }

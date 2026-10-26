import json

import numpy as np
import pandas as pd


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


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transformaciones sin estado: no aprenden nada de los datos."""
    _df = pd.DataFrame(index=df.index)

    for _c in BOOLEAN_FEATURES:
        _df[_c] = df[_c].astype(float)

    for _c in NUMERICAL_FEATURES:
        _df[_c] = pd.to_numeric(df[_c], errors="coerce")

    _desc = df["desc"].astype("string").fillna("")
    _df["desc_len"] = _desc.str.len().astype(float)
    _df["desc_n_palabras"] = _desc.str.split().str.len().fillna(0).astype(float)
    _df["desc_n_hashtags"] = _desc.str.count(r"#\w+").astype(float)
    _df["desc_n_menciones"] = _desc.str.count(r"@\w+").astype(float)
    _df["n_challenges"] = df["challenges"].map(contar_challenges).astype(float)
    _duet_from = pd.to_numeric(df["duet_info_duet_from_id"].astype(str), errors="coerce")
    _df["es_duet"] = (_duet_from.fillna(0) != 0).astype(float)
    _df["no_vq"] = (_df["vq_score"] == 0).astype(float)
    _df["dif_duracion_musica"] = (_df["duration"] - _df["music_duration"]).abs()

    for _c in OHE_FEATURES + TE_FEATURES:
        _df[_c] = df[_c].astype(str)

    # Texto crudo: lo vectorizan TF-IDF y la frecuencia de hashtags dentro del Pipeline
    _df["desc"] = _desc.astype(str)
    _df["challenges"] = df["challenges"].astype("string").fillna("[]").astype(str)

    _df[TARGET] = np.log1p(df[TARGET_RAW].astype(float))
    return _df[FEATURES + [TARGET]]

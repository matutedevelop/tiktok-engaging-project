import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:

    import marimo as mo
    import pandas as pd
    import numpy as np


@app.cell
def _():
    mo.md("""
    # Preparación del dataset
    """)
    return


@app.cell
def _():

    DROP_COLUMNS = [
        "stitch_display",
        "duet_display",
        "city",
        "poi_tt_type_name_super",
        "poi_tt_type_code",
        "country_code",
    ]

    BOOLEAN_FEATURES = [
        "duet_enabled",
        "is_ad",
        "item_mute",
        "item_control_can_repost",
        "official_item",
        "original_item",
        "share_enabled",
        "stitch_enabled",
        "music_original",
    ]
    NOMINAL_FEATURES = [
        "desc",
        "address",
        "poi_name",
        "city_code",
        "poi_category",
        "poi_tt_type_name_medium",
        "poi_tt_type_name_tiny",
        "challenges",
        "music_id",
        "music_title",
        "duet_info_duet_from_id",
        "music_author_name",
        "poi_id",
        "diversification_id",
        "item_comment_status",
    ]
    NUMERICAL_FEATURES = [
        "music_duration",
        "vq_score",
        "duration",
    ]
    TARGET = "play_count"

    ID_FEATURES = [c for c in NOMINAL_FEATURES if c.endswith("id")]
    KEEP_COLUMNS = BOOLEAN_FEATURES + NOMINAL_FEATURES + NUMERICAL_FEATURES + [TARGET]
    return (
        BOOLEAN_FEATURES,
        DROP_COLUMNS,
        ID_FEATURES,
        KEEP_COLUMNS,
        NOMINAL_FEATURES,
        NUMERICAL_FEATURES,
        TARGET,
    )


@app.cell
def _():
    df_raw = pd.read_parquet("data/raw/sample_data.parquet")
    return (df_raw,)


@app.cell
def _():
    mo.md("""
    ## 1. Selección de variables
    """)
    return


@app.cell
def _(DROP_COLUMNS, KEEP_COLUMNS, df_raw):
    # Dropeo explícito y luego filtro a las columnas que sí queremos
    df_1 = df_raw.drop(columns=[c for c in DROP_COLUMNS if c in df_raw.columns])
    faltantes = [c for c in KEEP_COLUMNS if c not in df_1.columns]
    if faltantes:
        raise KeyError(f"Columnas esperadas que no están en el dataset: {faltantes}")
    df_1 = df_1[KEEP_COLUMNS].copy()
    df_1.shape
    return (df_1,)


@app.cell
def _():
    mo.md("""
    ## 2. Tipos de dato
    """)
    return


@app.cell
def _(
    BOOLEAN_FEATURES,
    ID_FEATURES,
    NOMINAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    df_1,
):
    # no no vamo a engañar, esta preparacion ta batante vaicodea 
    df_2 = df_1.copy()

    # Booleanas: primero a dtype nullable "boolean" para no convertir NaN -> True
    # (astype(bool) directo hace eso). El cast final a bool va al terminar la limpieza.
    _map = {
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False, "t": True, "f": False,
    }
    for _c in BOOLEAN_FEATURES:
        _s = df_2[_c]
        if _s.dtype == object:
            _s = _s.astype(str).str.strip().str.lower().map(_map)
        df_2[_c] = _s.astype("bool")

    # *id -> category (más ligero que object y semánticamente correcto)
    for _c in ID_FEATURES:
        df_2[_c] = df_2[_c].astype("string").astype("category")

    # Resto de nominales -> object (string nullable)
    for _c in [c for c in NOMINAL_FEATURES if c not in ID_FEATURES]:
        df_2[_c] = df_2[_c].astype("string")

    # Numéricas y target -> numérico
    for _c in NUMERICAL_FEATURES + [TARGET]:
        df_2[_c] = pd.to_numeric(df_2[_c], errors="coerce")

    df_2.dtypes
    return (df_2,)


@app.cell
def _():
    mo.md("""
    ## 3. Imputación de numéricas
    """)
    return


@app.cell
def _(NUMERICAL_FEATURES, df_2):
    # Diagnóstico previo: % nulos y asimetría de cada numérica
    resumen_num = df_2[NUMERICAL_FEATURES].agg(["count", "mean", "median", "skew"]).T
    resumen_num["pct_nulos"] = df_2[NUMERICAL_FEATURES].isna().mean() * 100
    resumen_num
    return


@app.cell
def _(NUMERICAL_FEATURES, df_2):
    # Estrategia (Little & Rubin; Hastie et al.): para univariada simple, la media solo
    # es razonable con distribución ~simétrica; con asimetría fuerte (|skew| > 1) la
    # mediana es el estimador robusto. Se decide por columna.
    df_3 = df_2.copy()
    imputacion = {}
    for _c in NUMERICAL_FEATURES:
        _skew = df_3[_c].skew(skipna=True)
        if np.isnan(_skew) or abs(_skew) > 1:
            _valor, _estrategia = df_3[_c].median(), "median"
        else:
            _valor, _estrategia = df_3[_c].mean(), "mean"
        _n = int(df_3[_c].isna().sum())
        df_3[_c] = df_3[_c].fillna(_valor)
        imputacion[_c] = {"estrategia": _estrategia, "valor": _valor, "imputados": _n}
    imputacion
    return (df_3,)


@app.cell
def _():
    mo.md("""
    ## 4. Nulos residuales
    """)
    return


@app.cell
def _(df_3):
    nulos_restantes = df_3.isna().sum()
    nulos_restantes[nulos_restantes > 0]
    return


@app.cell
def _(BOOLEAN_FEATURES, df_3):
    n_antes = len(df_3)
    df_final = df_3.dropna().copy()

    # Ahora sí, cast final a bool nativo (ya sin nulos)
    for _c in BOOLEAN_FEATURES:
        df_final[_c] = df_final[_c].astype(bool)

    # Limpia categorías huérfanas tras el dropna
    for _c in df_final.select_dtypes("category").columns:
        df_final[_c] = df_final[_c].cat.remove_unused_categories()

    print(f"Filas eliminadas: {n_antes - len(df_final)} ({(n_antes - len(df_final)) / n_antes:.2%})")
    df_final.shape
    return (df_final,)


@app.cell
def _(df_final):
    df_final.dtypes
    return


@app.cell
def _(df_final):
    df_final.head()
    return


@app.cell
def _(df_final):
    # Descomenta para guardar el resultado
    # df_final.to_parquet("data_clean.parquet", index=False)
    df_final.describe(include="all").T
    return


if __name__ == "__main__":
    app.run()

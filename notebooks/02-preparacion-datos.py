import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import typer


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
        "item_control_can_repost",
        "official_item",
        "original_item",
        "share_enabled",
    ]

    BOOLEAN_FEATURES = [
        "duet_enabled",
        "is_ad",
        "item_mute",
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
    KEEP_COLUMNS = (
        BOOLEAN_FEATURES + NOMINAL_FEATURES + NUMERICAL_FEATURES + [TARGET]
    )

    ROOT = Path(__file__).parent.parent
    return (
        BOOLEAN_FEATURES,
        DROP_COLUMNS,
        ID_FEATURES,
        KEEP_COLUMNS,
        NOMINAL_FEATURES,
        NUMERICAL_FEATURES,
        ROOT,
        TARGET,
    )


@app.cell
def _(ROOT):
    df_raw = pd.read_parquet(ROOT / "data" / "raw" / "sample_data.parquet")
    return (df_raw,)


@app.cell
def _():
    mo.md("""
    ## 1. Selección de variables
    """)
    return


@app.cell
def _(DROP_COLUMNS, KEEP_COLUMNS, df_raw):
    def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
        _df = df.copy(deep=True)
        _df = _df.drop(
            columns=[c for c in DROP_COLUMNS if c in df_raw.columns]
        )
        faltantes = [c for c in KEEP_COLUMNS if c not in _df.columns]
        if faltantes:
            raise KeyError(
                f"Columnas esperadas que no están en el dataset: {faltantes}"
            )
        _df = _df[KEEP_COLUMNS].copy()
        print(f"[drop_columns] df.shape : {_df.shape}")
        return _df

    return (drop_columns,)


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
):
    def fix_dtypes(df: pd.DataFrame):

        _df = df.copy()

        _map = {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "t": True,
            "f": False,
        }
        for _c in BOOLEAN_FEATURES:
            _s = _df[_c]
            if pd.api.types.is_string_dtype(_s):
                _s = _s.astype(str).str.strip().str.lower().map(_map)
            _df[_c] = _s.astype("bool")

        # *id -> category (más ligero que object y semánticamente correcto)
        for _c in ID_FEATURES:
            _df[_c] = _df[_c].astype("string").astype("category")

        # Resto de nominales -> object (string nullable)
        for _c in [c for c in NOMINAL_FEATURES if c not in ID_FEATURES]:
            _df[_c] = _df[_c].astype("string")

        # Numéricas y target -> numérico
        for _c in NUMERICAL_FEATURES + [TARGET]:
            _df[_c] = pd.to_numeric(_df[_c], errors="coerce")

        _df.dtypes
        return _df

    return (fix_dtypes,)


@app.cell
def _():
    mo.md("""
    ## 3. Imputación de numéricas y de categoricas


    en retrospectiva, consideramos como algo mas prudente el imputar las descripciones vacias con `'NA'`
    """)
    return


@app.function
def cat_imputation(df: pd.DataFrame) -> pd.DataFrame:

    # We impute nulls by mode or by `Null` value based on the proportion of nulls of each variable
    # and the context avaliable for each variable
    _df = df.copy(deep=True)
    _df["desc"] = _df["desc"].fillna("NA")
    _df["poi_tt_type_name_tiny"] = _df["poi_tt_type_name_tiny"].fillna(
        _df["poi_tt_type_name_tiny"].mode()[0]
    )
    _df["diversification_id"] = (
        _df["diversification_id"]
        .astype("string")
        .fillna("Unknown")
        .astype("category")
    )
    return _df


@app.cell
def _(NUMERICAL_FEATURES):
    def numerical_imputation(df: pd.DataFrame) -> pd.DataFrame:

        _df = df.copy()

        for _c in NUMERICAL_FEATURES:
            _skew = _df[_c].skew(skipna=True)
            if np.isnan(_skew) or abs(_skew) > 1:
                _valor, _estrategia = _df[_c].median(), "median"
            else:
                _valor, _estrategia = _df[_c].mean(), "mean"
            _n = int(_df[_c].isna().sum())
            _df[_c] = _df[_c].fillna(_valor)
            print(
                f"[numerical_imputation]: Column: {_c}, Estrategia: {_estrategia}, valor: {_valor}, num_imputados: {_n}"
            )

        return _df

    return (numerical_imputation,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Dropeo de nulos
    """)
    return


@app.function
def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


@app.cell
def _():
    mo.md("""
    ## 5. Nulos residuales
    """)
    return


@app.cell
def _(df_raw, drop_columns, fix_dtypes, numerical_imputation):
    df_final = (
        df_raw.pipe(drop_columns)
        .pipe(fix_dtypes)
        .pipe(numerical_imputation)
        .pipe(cat_imputation)
        .pipe(drop_nulls)
    )
    return (df_final,)


@app.cell
def _(df_final):
    # Descomenta para guardar el resultado
    # df_final.to_parquet("data_clean.parquet", index=False)
    df_final.describe(include="all").T
    return


@app.cell
def _(ROOT, drop_columns, fix_dtypes, numerical_imputation):
    app = typer.Typer()

    @app.command()
    def sample():
        df = pd.read_parquet(ROOT / "data" / "raw" / "sample_data.parquet")
        (
            df.pipe(drop_columns)
            .pipe(fix_dtypes)
            .pipe(numerical_imputation)
            .pipe(cat_imputation)
            .pipe(drop_nulls)
            .to_parquet(ROOT / "data" / "interim" / "sample_data.parquet")
        )

    @app.command()
    def complete():
        df = pd.read_parquet(ROOT / "data" / "raw" / "data.parquet")
        (
            df.pipe(drop_columns)
            .pipe(fix_dtypes)
            .pipe(numerical_imputation)
            .pipe(cat_imputation)
            .pipe(drop_nulls)
            .to_parquet(ROOT / "data" / "interim" / "data.parquet")
        )

    if mo.app_meta().mode == "script":
        app()
    return


if __name__ == "__main__":
    app.run()

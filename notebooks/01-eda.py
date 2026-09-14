import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", css_file="custom.css")

with app.setup:
    from functools import partial

    import gc

    import marimo as mo
    import pandas as pd
    import polars as pl
    import altair as alt
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns


@app.cell
def _():
    sns.set_theme(style="white", palette="RdYlGn", font="BlexMono Nerd Font Mono")
    SEED = 171205


    def ctheme():
        return {
            "config": {
                "theme": "powerbi",
                "background": "#ffffff",
                "font": "Inter, system-ui, sans-serif",
                "autosize": {"type": "fit", "contains": "padding"},
                "title": {
                    "color": "#111111",
                    "font": "Inter, system-ui, sans-serif",
                    "fontSize": 16,
                    "fontWeight": "bold",
                    "anchor": "middle",
                    "offset": 10,
                },
                "view": {
                    "height": 450,
                    "width": 800,
                    "stroke": "transparent",
                },
                "axis": {
                    "labelColor": "#333333",
                    "titleColor": "#111111",
                    "labelFontSize": 12,
                    "titleFontSize": 14,
                    "gridColor": "#e5e5e5",
                    "labelAngle": -45,
                },
            }
        }


    alt.themes.register("ctheme", ctheme)
    alt.themes.enable("ctheme")
    return


@app.cell
def _():
    df = pd.read_parquet("data/raw/sample_data.parquet")
    df
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Data shape y quality
    """)
    return


@app.cell
def _(df):
    print(f"shape: {df.shape}")
    print(df.columns)
    return


@app.cell
def _(df):
    df.dtypes
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Tipos de datos

    Con solo ver los primeros registros del conjunto de datos, así como los nombres de las características y los tipos de datos, podemos observar que es necesario realizar algunas conversiones en algunas variables; por ejemplo, `is_ad` claramente está pensada para ser un valor booleano o `create_time` y similares que deberia de ser un datetime, pero se almacena como cadena.
    """)
    return


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell
def _(df):
    df.isna().mean()
    return


@app.cell
def _(df):
    df.isna().any(axis=1).mean()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Valores faltantes

    A primera vista, la mayoría de las variables tienen un porcentaje marginal de valores faltantes; sin embargo, si solo contamos los registros que no tienen ningún valor faltante, terminamos con menos del 1 % de nuestros datos, por lo que se necesita algún tipo de interpolación. Suponemos que bastará con interpolar las features numéricas, además de considerar que probablemente la mayoría de las features, como `addres, user_avatar_large, url, etc.` seran con alta probabilidad dropeadas a la hora del modelado, ya sea por cuestiones de privacidad de leakeage o por falta de relevancia.
    """)
    return


@app.cell
def _(df):
    print(f"proporcion de duplicados {df['id'].duplicated().mean()}")
    df.loc[df["id"].duplicated()]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Duplicados

    podemos observar que existe una cantidad muy pequeña de valores duplicados (comparados con el id). Por lo que dropearlos es una decision aceptable.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Analisis estadistico de los datos

    A continuacion exploramos distribuciones y un pequeño analisis descriptivo

    ## Preprocesamiento discreto

    Antes de comenzar a analizar los datos haremos un  preprocesamiento muy discreto que consiste casi unicamente en cambiar los tipos de datos a los correctos para poder analizarlos.
    """)
    return


@app.cell
def _():
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
    DATETIME_VARIABLES = [
        # "create_time" # Probablemente no sea usada por cuestiones de leakage
    ]
    NOMINAL_FEATURES = [
        "desc",
        "address",
        "poi_name",
        "city",
        "city_code",
        "country_code",
        "poi_category",
        "poi_tt_type_code",
        "poi_tt_type_name_medium",
        "poi_tt_type_name_super",
        "poi_tt_type_name_tiny",
        "challenges",
        "music_id",
        "music_title",
        "music_album",
        "music_author_name",
        "music_duration",
    ]
    NUMERICAL_FEATURES = [
        "music_duration",
        "music_id",
        "duet_info_duet_from_id",
        "vq_score",
        "playlist_id",
        "diversification_id",
        "country_code",
        "duration",
        "poi_id",
        "stitch_display",
        "duet_display",
        "item_comment_status",
    ]

    POST_STATISTICS_FEATURES = [
        "share_count",
        "play_count",
        "digg_count",
        "comment_count",
        "collect_count",
    ]
    TARGET = "play_count"
    return BOOLEAN_FEATURES, NOMINAL_FEATURES, TARGET


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Se separaron las `NUMERICAL_FEATURES` de las `POST_STATISTICS_FEATURES` debido a que por las razones explicadas en `informe-final.ipynb` las estadisticas de interacciones del post son candidatos o en su defecto material para el target
    """)
    return


@app.cell
def _(BOOLEAN_FEATURES, df):
    _bool_mapping = {"t": True, "f": False}
    boolean_features_df = (
        df[BOOLEAN_FEATURES]
        .copy(deep=True)
        .dropna()
        .map(lambda x: _bool_mapping.get(x))
        .astype("bool")
    )
    boolean_features_df
    return (boolean_features_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analisis de Dummies
    """)
    return


@app.cell
def _(boolean_features_df):
    _df = (
        boolean_features_df.mean()
        .reset_index()
        .rename(columns={"index": "column", 0: "proportion_1s"})
    )


    _c1 = (
        alt.Chart(_df)
        .mark_bar()
        .encode(x="column", y="proportion_1s")
        .properties(title="balance de variables booleanas")
    )
    _c2 = (
        alt.Chart()
        .mark_rule(
            color="salmon",
            strokeDash=[
                4,
                4,
            ],
            size=1,
        )
        .encode(y=alt.datum(0.5))
    )
    _c1 + _c2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    En general podemos ver que las variables doomies tal como estan no estan para nada balanceadas, esto puede ser un problema en el entrenamiento y tiene que ser tomado en cuenta a la hora del preprocesamiento.

    ### comparacion con el target

    a continuacion vamos a comparar las distribuciones del target respecto a cada variable dummie. Como se mostro en las celdas anteriores, se usara para el analisis de forma preliminar `play_count`
    """)
    return


@app.cell
def _(TARGET, boolean_features_df, df):
    boolean_df = pd.concat([boolean_features_df, df[[TARGET]]], axis=1)
    boolean_df
    return (boolean_df,)


@app.cell
def _(BOOLEAN_FEATURES, TARGET, boolean_df):
    _, ax = plt.subplots(
        nrows=len(BOOLEAN_FEATURES),
        ncols=1,
        figsize=(15, 10 * len(BOOLEAN_FEATURES)),
    )
    ax = ax.flatten()
    for i, dummie_col in enumerate(BOOLEAN_FEATURES):
        sns.violinplot(
            boolean_df, x=TARGET, hue=dummie_col, ax=ax[i], log_scale=True
        )
        ax[i].set_title(
            f"{dummie_col}\n false/true ratio: {boolean_df[dummie_col].mean()}"
        )
    plt.show()
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A primera vista los resultados no son alentadores, las distribuciones lucen muy similares, y las que tienen diferencias mas marcadas son en aquellas variables donde las clases estan sumamente desbalanceadas.


    ### Variables Nominales

    A continuacion haremos un analisis somero y superficial de las variables Nominales que pueden tener mucha mas variedad de valores, como por ejemplo `desc`, el analisis sera superficial debido a que al tratarse variables de texto pueden llegar a requerir tecnicas de feature engineering mas sofisticadas. Por lo que lo dejamos para mas delante.
    """)
    return


@app.cell
def _(NOMINAL_FEATURES, df):
    # df[["city", "city_code"]]
    nom_df_ = df[NOMINAL_FEATURES].copy(deep=True)
    nom_df_
    return (nom_df_,)


@app.cell
def _(nom_df_):
    nom_df_.isna().mean()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Ahora recordamos que city es una de las variables con mayor cantidad de valores nulosm no obstante existe la variable `city_code` que apenas tiene  nulos, solo nos queda verificar que en efecto estas dos variables nos dicen la misma informacion
    """)
    return


@app.cell
def _(nom_df_):
    city_crosstab = pd.crosstab(
        nom_df_["city"], nom_df_["city_code"], normalize="index"
    )

    # silly code para poder ordenar todas las columnas en descending no encontre otra forma al menos que le entendiera
    (city_crosstab * -1).transform(np.sort).abs()
    return (city_crosstab,)


@app.cell
def _(city_crosstab):
    # city_crosstab.sum(axis=1)
    # city_crosstab.sum()

    sns.histplot(city_crosstab.sum())
    plt.title("suma de las columnad de crosstab \ncity vs city_code")
    return


@app.cell
def _(nom_df_):
    sns.histplot(
        nom_df_.groupby("city_code")["city"]
        .nunique()
        .sort_values(ascending=False),
    )
    plt.title(
        "histograma del numero de diferentes ciudades\nasociadas a un mismo city_code"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    El resultado es a decir verdad postivo, ambos de estos graficos nos cuentan una historia similar y nos dicen basicamente que a efectos practicos `city` y `city_code` son colineares, evidentemente que los resultados son un tanto diferentes si conmutamos la comparacion, debido a que en una hay mas de la mitad de nulos. Viendo los datos linea por linea evidentemente no es tan sencillo, por ejemplo aquellas `city_code` con mas de una ciudad, que se muestra que existen en ambos graficos, no obstante siendo una proporcion relativamente pequeña lo consideramos  suficiente para dropear `city` y trabajar unicamente con `city_code`
    """)
    return


@app.cell
def _(nom_df_):
    nom_df = nom_df_.drop(columns=["city"])
    nom_df
    return (nom_df,)


@app.cell
def _(nom_df):
    _df = (
        nom_df.nunique()
        .reset_index()
        .rename(columns={"index": "column", 0: "Distinct_count"})
    )
    alt.Chart(_df).mark_bar().encode(x="column", y="Distinct_count").properties(
        title="cadenas diferentes por columna"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    una de las cosas que salta mucho a la vista es la gran diferencia entre `music_title` y `music_id` uno podria llegar a pensar que deberian de ser similares, la hipotesis es que puede haber una buena cantidad de canciones diferentes con el mismo titulo, aun asi vale la pena investigar
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

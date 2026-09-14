# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.24.2",
# ]
# ///

import marimo

__generated_with = "0.24.2"
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
    sns.set_theme(
        style="white", palette="RdYlGn", font="BlexMono Nerd Font Mono"
    )
    SEED = 171205

    def ctheme():
        return {
            "config": {
                "theme": "powerbi",
                "background": "#ffffff",
                # "font": "Inter, system-ui, sans-serif",
                "font": " BlexMono Nerd Font Mono",
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

    # alt.data_transformers.enable("vegafusion")
    alt.themes.register("ctheme", ctheme)
    alt.theme.enable("ctheme")
    return (SEED,)


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
    _df = (
        df.isna()
        .mean()
        .rename_axis("column")
        .reset_index(name="null_proportion")
    )
    alt.Chart(_df).mark_bar().encode(
        x=alt.X(
            "column:N",
            sort=None,
            title="column",
            axis=alt.Axis(labelAngle=-90),
        ),
        y="null_proportion",
    ).properties(title="proporcion de nulos por variable")
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
        "stitch_display",
        "duet_display",
    ]

    POST_STATISTICS_FEATURES = [
        "share_count",
        "play_count",
        "digg_count",
        "comment_count",
        "collect_count",
    ]
    TARGET = "play_count"
    return BOOLEAN_FEATURES, NOMINAL_FEATURES, NUMERICAL_FEATURES, TARGET


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
    for i1, dummie_col in enumerate(BOOLEAN_FEATURES):
        sns.violinplot(
            boolean_df, x=TARGET, hue=dummie_col, ax=ax[i1], log_scale=True
        )
        ax[i1].set_title(
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
    plt.title("suma de las columnas de crosstab \ncity vs city_code")
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
def _():
    return


@app.cell
def _(nom_df_):
    _df = (
        nom_df_.nunique()
        .reset_index()
        .rename(columns={"index": "column", 0: "Distinct_count"})
    )
    alt.Chart(_df).mark_bar().encode(
        x="column", y="Distinct_count"
    ).properties(title="cadenas diferentes por columna")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    una de las cosas que salta mucho a la vista es la gran diferencia entre `music_title` y `music_id` uno podria llegar a pensar que deberian de ser similares, la hipotesis es que puede haber una buena cantidad de canciones diferentes con el mismo titulo, aun asi vale la pena investigar

    veamos cuales son aquellas columnas que tienen menos granularidad en sus filas
    """)
    return


@app.cell
def _(nom_df_):
    nom_df_.nunique().sort_values(ascending=True)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A primera instancia si queremos analizar algo como la media condicional vemos que a lo sumo hay 5 variables utiles, desde `poi_category` hasta `poi_tt_type_code`. es decir 364 clases para 2,000,000 de datos  o 10,000,000 en el dataset completo, todavia puede contarnos algo. Mas de 1,000 entradas diferentes necesitan algun otro tipo de feature engineering par que nos sean de utilidad, y en su defeto `country_code es inutil`.

    Otra cosa que vale la pena mirar, es analizar la colinearidad entre `poi_category, poi_tt_type_name_super` y `poi_tt_type_name_tiny, poi_tt_type_code` debido a que tienen granularidades muy similares
    """)
    return


@app.cell
def _(nom_df_):
    _df = (
        nom_df_.groupby("poi_category")["poi_tt_type_name_super"]
        .nunique()
        .reset_index()
        .rename(
            columns={
                "poi_tt_type_name_super": "distinct_poi_tt_type_name_super_count"
            }
        )
    )
    alt.Chart(_df).mark_bar().encode(
        y="distinct_poi_tt_type_name_super_count", x="poi_category"
    ).properties(title="colinearidad  poi_category vs poi_tt_type_name_super ")
    return


@app.cell
def _(nom_df_):
    _table = pd.crosstab(
        nom_df_["poi_category"],
        nom_df_["poi_tt_type_name_super"],
        normalize=True,
    )

    (_table * -1).transform(np.sort).abs()
    return


@app.cell
def _(nom_df_):
    # poi_tt_type_name_tiny, poi_tt_type_code
    _df = (
        nom_df_.groupby("poi_tt_type_name_tiny")["poi_tt_type_code"]
        .nunique()
        .reset_index()
        .rename(columns={"poi_tt_type_code": "distinct_poi_tt_type_code"})
    )
    alt.Chart(_df).mark_bar().encode(
        y="distinct_poi_tt_type_code", x="poi_tt_type_name_tiny"
    ).properties(title="colinearidad  poi_category vs poi_tt_type_name_super ")
    return


@app.cell
def _(nom_df_):
    _table = pd.crosstab(
        nom_df_["poi_tt_type_name_tiny"],
        nom_df_["poi_tt_type_code"],
        normalize=True,
    )

    (_table * -1).transform(np.sort).abs()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `poi_category, poi_tt_type_name_super` y `poi_tt_type_name_tiny, poi_tt_type_code` son colineares por los que deberiamos de quedarnos solamente con una por cada par
    """)
    return


@app.cell
def _(nom_df_):
    nom_cols_to_drop = [
        "city",
        "poi_tt_type_name_super",
        "poi_tt_type_code",
        "country_code",
    ]
    nom_df = nom_df_.drop(columns=nom_cols_to_drop)
    nom_df
    return (nom_df,)


@app.cell
def _(nom_df):
    nominal_barcount_charts = []

    for nom_col in nom_df:
        # 1. Preparar el DataFrame con orden descendente y calcular la proporción acumulada
        hist_df_ = (
            nom_df[nom_col]
            .value_counts()
            .sort_values(ascending=False)
            .rename_axis("string")
            .reset_index(name="count")
        )

        # Ojiva acumulada (0 a 1)
        hist_df_["cum_prop"] = (
            hist_df_["count"].cumsum() / hist_df_["count"].sum()
        )

        hist_df = hist_df_.head(40)  # Top 40 categorías

        bars = (
            alt.Chart(hist_df)
            .mark_bar(color="#4c78a8")
            .encode(
                x=alt.X(
                    "string:N",
                    sort=None,
                    title=nom_col,
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y("count:Q", title="Conteo"),
            )
        )

        line = (
            alt.Chart(hist_df)
            .mark_line(color="salmon", point=False)
            .encode(
                x=alt.X("string:N", sort=None),
                y=alt.Y(
                    "cum_prop:Q",
                    title="Proporción Acumulada",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"),
                ),
            )
        )

        chart = (
            alt.layer(bars, line)
            .resolve_scale(y="independent")
            .properties(title=f"Conteo y Ojiva de {nom_col}")
        )

        nominal_barcount_charts.append(chart)

    alt.vconcat(*nominal_barcount_charts)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Las interpretaciones de los graficos arrojan los diferentes resultados, aunque `desc` tenga un histograma con una clase dominante, hay demasiada variedad, cosa que ya veiamos desde el grafico anterior `address` es una variable mas manejable, con un codo de pareto mas definido, esto entre otras cosas nos sirve para establecer un threshold y definir una label `other` para los demas valores, otras variables con las que masomenos podriamos hacer algo similar son `city_code` `poi_tt_type_name_medium`, `poi_tt_type_name_tiny` y `diversification_id`



    ### Analisis de variables numericas
    """)
    return


@app.cell
def _(NUMERICAL_FEATURES, SEED, TARGET, df):
    num_df = df[NUMERICAL_FEATURES + [TARGET]]
    num_sample = num_df.sample(10_000, random_state=SEED)

    c = sns.pairplot(num_sample, hue=TARGET, palette="RdYlGn")

    c._legend.remove()

    _norm = plt.Normalize(num_sample[TARGET].min(), num_sample[TARGET].max())
    _sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=_norm)

    c.fig.colorbar(_sm, ax=c.axes, orientation="vertical", label=TARGET)
    c
    return num_df, num_sample


@app.cell
def _(num_df):
    num_df.describe()
    return


@app.cell
def _(NUMERICAL_FEATURES, TARGET, num_df):
    _, ax1 = plt.subplots(
        nrows=len(NUMERICAL_FEATURES),
        ncols=2,
        figsize=(15, 10 * len(NUMERICAL_FEATURES)),
    )

    ax1 = ax1.flatten()

    for i2, num_col in enumerate(NUMERICAL_FEATURES):
        sns.scatterplot(num_df, x=num_col, y=TARGET, ax=ax1[i2 * 2])
        sns.boxplot(num_df, x=num_col, ax=ax1[i2 * 2 + 1], log_scale=True)
        ax1[i2 * 2].set_yscale("log")
    plt.show()
    return


@app.cell
def _(num_df):
    # para chequear que duet y stitch display tienen un solo valor

    alt.data_transformers.enable("vegafusion")

    _cols = ["duet_display", "stitch_display"]

    alt.hconcat(
        *[
            alt.Chart(num_df[[c]])
            .mark_bar()
            .encode(x=alt.X(c, bin=True), y="count()")
            .properties(title=f"histograma de {c}")
            for c in _cols
        ]
    )
    return


@app.cell
def _(num_df):
    num_cols_to_drop = ["stitch_display", "duet_display"]
    num_df_ = num_df.drop(columns=num_cols_to_drop)
    return (num_df_,)


@app.cell
def _(TARGET, num_df, num_sample):
    alt.data_transformers.enable("default")

    _, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    sns.boxplot(num_sample[TARGET], log_scale=True, ax=ax2[0])
    sns.histplot(num_sample[TARGET], log_scale=True, ax=ax2[1])
    ax2[0].set_title("boxplot de play_count")
    ax2[1].set_title(f"histplot de play_count\nskew:{num_df[TARGET].skew()}")
    return


@app.cell
def _(num_df_):
    sns.heatmap(num_df_.corr(), cmap="RdYlGn")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Este pequeño analisis nos indica tambien que otras variables podemos tratar o debemos dropear. Lo Primero y mas evidente es que probablemente busquemos modelar al final $\ln(y)$ o $y^\lambda$ debido a que sino nos quedamos con un target con demasiados outliers. Por otro lado tambien revela que debemos de dropear `duet_display, stitch_display` y que hay oportunidad para discretizar o crear una nueva variable `no_vq` para intentar tratar aquellos donde `vq_score` es 0, por otro lado tenemos una correlacion alta entre `duration` y `music_duration` un tema que hay que cuidar si de casualidad se usa modelos lineales con interpretacion sobre los $w$ de las features, aunque de primeras luce complicado que un modelo lineal ajuste bien
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Conclusion

    Siendo concisos nos encontramos con un dataset relativamente limpio con muchos metadatos y datos de caracter nominal o variables de texto con mucha granularidad. Pocas Features numericas y sin relacion aparente con el target, el target es una variable con alta varianza y desviacion estandar alta y bastante sesgada.


    Lo que sigue a continuacion es  una preparacion de los datos sumamente conservadora, tales como imputacion de nulos dropear variables

    si quisieramos ser mas especificos los pasos a seguir serian los siguientes

    1. Dropear las variables

    ["stitch_display", "duet_display","city","poi_tt_type_name_super","poi_tt_type_code","country_code"]


    y quedarnos solamente con
    ```
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
        "music_album",
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
    ```


    2. cambiar a los tipos de dato correctos

    boolean_features -> astype(bool)
    *id -> object / category

    3. imputacion de datos numericos segun la estrategia recomendada por la literatura

    4. dropear la cantidad marginal de nulos restantes


    Riesgos eticos y sesgos


    Hemos optado en principio en no usar ni variables temporales por cuestiones de un posible leakeage ni variables relacionadas con la informacion del usuario
    """)
    return


if __name__ == "__main__":
    app.run()

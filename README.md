# TEE — TikTok Engage Estimator

*Título provisional.*

Proyecto de ciencia de datos para estimar el número de reproducciones (`play_count`) de un video de TikTok usando únicamente información disponible **antes** de publicarlo.

## Integrantes

- Owen David Loza Quirarte
- Juan Pablo Arroyo Godinez
- Santiago Ayon sanchez

## Problema

Un creador que está por publicar un video no tiene ninguna estimación cuantitativa de cuántas reproducciones alcanzará; solo lo sabe después del hecho. Este proyecto busca responder si el alcance de un TikTok se puede predecir **a priori**, es decir, a partir de decisiones que el creador controla en el momento de publicar: configuración de la publicación (duet, stitch, repost, share, comentarios), música, texto descriptivo y hashtags, punto de interés geográfico, duración y calidad del video.

Deliberadamente **no** se usan como entrada las métricas de engagement posteriores (`digg_count`, `comment_count`, `share_count`, `collect_count`), variables temporales (`create_time`) ni información del autor (`user_id`, avatares, `user_verified`, `user_tt_seller`), por leakage y privacidad.

### Resultado esperado

Un modelo de regresión sobre una transformación logarítmica de `play_count` que, evaluado con métricas robustas a colas pesadas (RMSLE, MAE en escala log, correlación de Spearman), supere a baselines triviales y permita discriminar el orden de magnitud del alcance de un video. Además, una cuantificación de qué tan predecible es el éxito sin señal social ni temprana, y de qué decisiones controlables tienen efecto medible.

## Datos

- **Fuente original:** [The-data-company/TikTok-10M](https://huggingface.co/datasets/The-data-company/TikTok-10M) (Hugging Face). ~6.65 M publicaciones de TikTok en formato Parquet con metadatos de la publicación, estadísticas de engagement, punto de interés, música y video. Contenido *trending* con POI en Estados Unidos, primavera 2025.
- **Diccionario de datos, variables seleccionadas y sesgos:** [`data/README.md`](data/README.md).

Los datos no se versionan en el repositorio; ver instrucciones de descarga más abajo.

## Estructura del repositorio

```
.
├── README.md                  # este archivo
├── pyproject.toml             # dependencias del proyecto
├── uv.lock                    # lockfile de uv
├── custom.css                 # estilos para los notebooks de marimo
├── data/
│   ├── README.md              # diccionario de datos y notas sobre la fuente
│   ├── raw/                   # datos descargados (ignorado por git)
│   │   └── sample_data.parquet
│   └── processed/             # salida del pipeline de preparación (ignorado por git)
├── notebooks/
│   ├── 01-eda.py              # análisis exploratorio (marimo)
│   └── 02-preparacion.py      # pipeline de preparación con pandas (marimo)
└── docs/
    └── propuesta.md           # documento del proyecto: contexto, objetivos, EDA, siguientes pasos
```

## Estado actual

- [x] Análisis exploratorio (`01-eda.py`): calidad de datos, nulos, duplicados, tipos, distribuciones del target y de las features booleanas, nominales y numéricas, colinealidades.
- [x] Definición del conjunto de features a priori: 9 booleanas, 16 nominales, 3 numéricas y target `play_count`.
- [x] Pipeline de preparación (`02-preparacion.py`): selección de columnas, tipificación, imputación de numéricas por media/mediana según asimetría, eliminación de nulos residuales. Factorizado en funciones encadenables con `.pipe()`.
- [ ] Ingeniería de features para variables de alta cardinalidad y texto (`desc`, `challenges`, `music_id`, `poi_id`).
- [ ] Protocolo de evaluación, baselines y modelado.

### Hallazgos principales del EDA

- Si se exige que un registro no tenga ningún nulo queda menos del 1 % de los datos; se imputan solo las numéricas.
- El target tiene cola muy pesada; se modelará `log(play_count)` o una transformación Box-Cox.
- Las booleanas están muy desbalanceadas y sus distribuciones de `play_count` son casi idénticas entre clases.
- `city`/`city_code`, `poi_category`/`poi_tt_type_name_super` y `poi_tt_type_name_tiny`/`poi_tt_type_code` son colineales; se conserva una de cada par. `country_code`, `duet_display` y `stitch_display` son constantes.
- `duration` y `music_duration` están altamente correlacionadas. Las numéricas no muestran relación aparente con el target.

## Siguientes pasos

1. Feature engineering: longitud y conteo de hashtags en `desc`/`challenges`; frecuencia o target encoding para `music_id`, `poi_id`, `poi_category`; agrupación de colas en `other` para `address`, `city_code`, `poi_tt_type_name_*`, `diversification_id`; bandera `no_vq` para `vq_score == 0`.
2. Split estratificado por cuantiles del target y baselines triviales (media, mediana, mediana por categoría).
3. Modelos: regresión lineal regularizada como baseline interpretable, gradient boosting (LightGBM/XGBoost) como modelo principal, y una formulación alternativa como clasificación ordinal por orden de magnitud.
4. Interpretación con importancia por permutación / SHAP.

## Reproducibilidad

### Requisitos

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/)

### Preparar el ambiente

```bash
git clone <url-del-repo>
cd <nombre-del-repo>
uv sync
```

Esto crea `.venv/` e instala las dependencias declaradas en `pyproject.toml` respetando `uv.lock`.

### Obtener los datos

se debe ejecutar el siguiente codigo dentro de `data/raw/`
```python
import gc

import pandas as pd
from datasets import load_dataset


ds = load_dataset("The-data-company/TikTok-10M")

ds["train"].to_parquet("data/raw/data.parquet")

del ds
gc.collect()

df = pd.read_parquet("data/raw/data.parquet")
sample_df = df.sample(frac=0.2, random_state=69)
sample_df.to_parquet("data/raw/sample_data.parquet")


```

> Ajusta el tamaño de la muestra según tu memoria disponible. El EDA se realizó sobre ~2 M registros.

### Ejecutar los notebooks

Ambos notebooks son archivos `.py` de [marimo](https://marimo.io/).
aunque tambien existen sus respectivos exports a jupyter

```bash
# EDA (modo interactivo)
uv run marimo edit notebooks/01-eda.py

# Preparación de datos (modo interactivo)
uv run marimo edit notebooks/02-preparacion.py

# o ejecutar de corrido, sin editor
uv run marimo run notebooks/01-eda.py
uv run marimo run notebooks/02-preparacion.py
```

`02-preparacion.py` lee `data/raw/sample_data.parquet` y, si se descomenta la última celda, escribe `data/processed/data_clean.parquet`.

### Qué no se versiona

`.venv/`, `data/raw/`, `data/processed/`, secretos, credenciales y cualquier artefacto generado. Ver `.gitignor

estos notebooks son para el EDA y probar una version inicial del pipeline, el resultado no es persistente








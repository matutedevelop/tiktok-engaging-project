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
├── .env.example               # nombres de las variables de entorno, sin valores reales
├── .gitignore                 # ignora .env, .venv, datos y artefactos
├── .python-version            # 3.12
├── LICENSE
├── devenv.nix / devenv.yaml / devenv.lock # entorno de desarrollo con devenv
├── data/
│   ├── README.md              # diccionario de datos y notas sobre la fuente
│   ├── raw/                   # datos descargados (ignorado por git)
│   │   └── sample_data.parquet
│   ├── interim/               # salida de 02-preparacion-datos (ignorado por git)
│   └── processed/             # salida del pipeline de preparación (ignorado por git)
├── notebooks/
│   ├── 01-eda.py              # análisis exploratorio (marimo)
│   ├── 02-preparacion.py      # pipeline de preparación con pandas (marimo)
│   ├── 01-eda.ipynb           # export a Jupyter del EDA
│   ├── 02-preparacion-datos.ipynb # export a Jupyter de la preparación
│   ├── 03-experimentos.py     # baselines, 3 familias, HPO con Optuna y comparación (marimo)
│   ├── 04-seleccion-registro.py # selección, registro y aliases en MLflow (marimo)
│   ├── 05-pipeline.py         # flow de Prefect: etapas, ejecución y verificación (marimo)
│   ├── 06-api.ipynb           # contrato, carga del modelo y pruebas de la API
│   └── 07-frontend.ipynb      # formulario, llamada HTTP, resultado y errores
├── evidencia/                 # capturas enlazadas desde este README
│   ├── evidencia-01-eda.png
│   ├── evidencia-02-preparacion-datos.png
│   ├── api-prediccion-valida.png
│   ├── api-entrada-invalida.png
│   ├── frontend-prediccion.png
│   ├── frontend-solicitud-respuesta.png
│   ├── frontend-error-validacion.png
│   └── frontend-error-conexion.png
├── informe/
│   └── informe-final.ipynb    # síntesis del proyecto con enlaces a los notebooks
├── src/
│   ├── entrenamiento/         # features, HPO y flow de Prefect
│   │   ├── configuracion.py   # features, rutas, semilla, experimento y modelo
│   │   ├── features.py        # preparación sin estado y Pipeline de scikit-learn
│   │   ├── hpo.py             # espacios de búsqueda y estudios de Optuna
│   │   └── flujo.py           # tasks y flow tee-entrenamiento
│   ├── api/                   # FastAPI
│   │   ├── esquemas.py        # contrato de entrada y salida (Pydantic)
│   │   └── main.py            # carga del modelo champion y endpoints
│   └── frontend/              # Streamlit
│       ├── app.py             # formulario y presentación del resultado
│       └── cliente.py         # llamadas HTTP a la API y manejo de errores
└── docs/
    ├── mapa-sistema-ml.md     # mapa del sistema de ML
    └── propuesta.md           # documento del proyecto: contexto, objetivos, EDA, siguientes pasos
```

## Estado actual

- [x] Análisis exploratorio (`01-eda.py`): calidad de datos, nulos, duplicados, tipos, distribuciones del target y de las features booleanas, nominales y numéricas, colinealidades.
- [x] Definición del conjunto de features a priori: 9 booleanas, 16 nominales, 3 numéricas y target `play_count`.
- [x] Pipeline de preparación (`02-preparacion.py`): selección de columnas, tipificación, imputación de numéricas por media/mediana según asimetría, eliminación de nulos residuales. Factorizado en funciones encadenables con `.pipe()`.
- [x] Ingeniería de features para variables de alta cardinalidad y texto (`desc`, `challenges`, `music_id`, `poi_id`).
- [x] Protocolo de evaluación, baselines y modelado.
- [x] API de predicción (`src/api/`, `06-api.ipynb`): FastAPI que carga el modelo `champion` del Registry una sola vez al iniciar, valida la entrada con Pydantic y devuelve las reproducciones estimadas junto con la versión del modelo.
- [x] Frontend (`src/frontend/`, `07-frontend.ipynb`): formulario de Streamlit que llama a la API por HTTP, presenta la predicción con su interpretación y maneja errores de conexión y validación.

### Hallazgos principales del EDA

- Si se exige que un registro no tenga ningún nulo queda menos del 1 % de los datos; se imputan solo las numéricas.
- El target tiene cola muy pesada; se modelará `log(play_count)` o una transformación Box-Cox.
- Las booleanas están muy desbalanceadas y sus distribuciones de `play_count` son casi idénticas entre clases.
- `city`/`city_code`, `poi_category`/`poi_tt_type_name_super` y `poi_tt_type_name_tiny`/`poi_tt_type_code` son colineales; se conserva una de cada par. `country_code`, `duet_display` y `stitch_display` son constantes.
- `duration` y `music_duration` están altamente correlacionadas. Las numéricas no muestran relación aparente con el target.

## API y frontend

El modelo se entrena y registra en un proceso aparte; la API solo lo carga y sirve predicciones, y el frontend es un cliente HTTP de la API. Ninguno de los dos reentrena al atender una solicitud.

```
Streamlit (formulario) --POST JSON--> FastAPI (Pydantic) --> Pipeline champion --> reproducciones + versión
```

### Modelo servido

- **Experimento:** `/Shared/pcd-otono-2026-proyecto-tiktok-engaging-estimator`
- **Modelo registrado:** `workspace.tiktok-engaging-estimator.pcd_otono_2026_tiktok_engaging_estimator`. El equipo registra en un esquema propio de Unity Catalog (`tiktok-engaging-estimator`) en lugar de `default`; la API toma el nombre de `MODELO_REGISTRADO`, así que no depende de esa decisión.
- **Versión `champion` al verificar:** versión 3 (`lightgbm`, test RMSLE = 2.057). Las versiones 2 (`random_forest`) y 1 (`ridge`) tienen los aliases `challenger` y `candidate`.

### API (`src/api/`)

- `esquemas.py`: contrato de entrada y salida. Los campos de entrada son exactamente las 20 columnas crudas que recibe el modelo registrado, con rangos y categorías tomados de `data/interim`; un campo faltante, fuera de rango o desconocido responde **422** sin llamar al modelo.
- `main.py`: al iniciar resuelve `champion` a una versión concreta y carga esa versión; cada solicitud reutiliza el mismo `Pipeline`.
- `POST /api/v1/predicciones` devuelve `reproducciones_estimadas` (`expm1` de la salida del modelo), `log_reproducciones`, `modelo`, `alias`, `version` y `familia`.
- `GET /health` responde 200 con la versión cargada. Si el modelo no pudo cargarse, la API arranca de todos modos y tanto `/health` como las predicciones responden **503**; nunca se devuelve una predicción inventada.
- Si se reasigna `champion`, hay que reiniciar la API para que cargue la nueva versión.

Detalle del contrato, carga del modelo y pruebas en [`notebooks/06-api.ipynb`](notebooks/06-api.ipynb).

### Frontend (`src/frontend/`)

- `app.py`: formulario de Streamlit en cuatro pestañas (Video, Configuración, Música, Ubicación) con los mismos límites que la API. Muestra las reproducciones estimadas, el modelo, alias y versión que respondieron, el orden de magnitud y un rango típico según el RMSLE del champion, junto con las limitaciones del modelo.
- `cliente.py`: llamadas HTTP a la API y traducción de errores. Un 422 se muestra campo por campo, un 503 indica que no hay modelo y una API apagada muestra un error de conexión; en ningún caso de error se muestra un número.
- No carga el modelo ni usa credenciales de Databricks; solo necesita `API_URL`.

Detalle del formulario, la llamada HTTP y el manejo de errores en [`notebooks/07-frontend.ipynb`](notebooks/07-frontend.ipynb).

### Evidencia

| Caso | Captura |
|---|---|
| API con entrada válida (predicción y versión) | [`evidencia/api-prediccion-valida.png`](evidencia/api-prediccion-valida.png) |
| API con entrada inválida (422) | [`evidencia/api-entrada-invalida.png`](evidencia/api-entrada-invalida.png) |
| Frontend con predicción | [`evidencia/frontend-prediccion.png`](evidencia/frontend-prediccion.png) |
| Solicitud y respuesta de la API para esa predicción | [`evidencia/frontend-solicitud-respuesta.png`](evidencia/frontend-solicitud-respuesta.png) |
| Frontend con error de validación | [`evidencia/frontend-error-validacion.png`](evidencia/frontend-error-validacion.png) |
| Frontend con la API apagada | [`evidencia/frontend-error-conexion.png`](evidencia/frontend-error-conexion.png) |

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

### Configurar variables de entorno

La API y el frontend leen su configuración de `.env` en la raíz (ignorado por git). Se parte de la plantilla:

```bash
cp .env.example .env
```

| Variable | Uso |
|---|---|
| `DATABRICKS_HOST`, `DATABRICKS_TOKEN` | acceso a Databricks (solo la API) |
| `MODELO_REGISTRADO` | nombre de tres niveles del modelo, sin `models:/` ni alias |
| `ALIAS_MODELO` | alias que carga la API (`champion`) |
| `RUTA_MODELO_LOCAL` | opcional: MLflow Model local para probar sin Databricks |
| `API_URL` | endpoint que llama el frontend (`http://127.0.0.1:8000/api/v1/predicciones`) |

### Levantar la API y el frontend

Primero la API y después el frontend, cada uno en su terminal, desde la raíz del repositorio:

```bash
uv sync --locked

# Terminal 1: API (documentación interactiva en http://127.0.0.1:8000/docs)
uv run fastapi dev src/api/main.py --port 8000

# Terminal 2: frontend (http://localhost:8501)
uv run streamlit run src/frontend/app.py --server.port 8501
```

Al iniciar, el log de la API muestra `Modelo cargado: <modelo>@champion -> version N (<familia>)`, y `http://127.0.0.1:8000/health` debe responder `"modelo_cargado": true`.

### Qué no se versiona

`.venv/`, `data/raw/`, `data/processed/`, secretos, credenciales y cualquier artefacto generado. Ver `.gitignor

estos notebooks son para el EDA y probar una version inicial del pipeline, el resultado no es persistente








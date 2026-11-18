# TEE — TikTok Engage Estimator

Proyecto de ciencia de datos para estimar el número de reproducciones (`play_count`) de un video de TikTok usando únicamente información disponible **antes** de publicarlo. Incluye el modelo, un workflow de entrenamiento con Prefect, el registro en MLflow / Unity Catalog (Databricks), una API en FastAPI y una interfaz en Streamlit, empaquetadas con Docker Compose y desplegadas en una instancia AWS EC2.

## Integrantes

- Owen David Loza Quirarte
- Juan Pablo Arroyo Godínez
- Santiago Ayón Sánchez

## Problema

Un creador que está por publicar un video no tiene ninguna estimación cuantitativa de cuántas reproducciones alcanzará; solo lo sabe después del hecho. Este proyecto busca responder si el alcance de un TikTok se puede predecir **a priori**, a partir de decisiones que el creador controla en el momento de publicar: configuración de la publicación (duet, stitch, comentarios), música, texto descriptivo y hashtags, punto de interés geográfico, duración y calidad del video.

Deliberadamente **no** se usan como entrada las métricas de engagement posteriores (`digg_count`, `comment_count`, `share_count`, `collect_count`), variables temporales (`create_time`) ni información del autor (`user_id`, avatares, `user_verified`, `user_tt_seller`), por leakage y privacidad.

## Resultado

El modelo `champion` es un **LightGBM** sobre $\log(1 + \text{play\_count})$, entrenado con 270,000 filas y evaluado una sola vez en test:

| Métrica (test) | Valor |
|---|---:|
| RMSLE | 2.057 |
| $R^2$ (escala log) | 0.170 |
| MAE (escala log) | 1.642 |
| Spearman | 0.406 |

Comparación de las tres familias en la misma corrida del flow (mismos datos, particiones y preprocesamiento; ranking por RMSLE de test):

| Familia | Versión | Alias | RMSLE validación | RMSLE test | $R^2$ test | Spearman test |
|---|---:|---|---:|---:|---:|---:|
| **LightGBM** | 3 | `champion` | 2.063 | **2.057** | **0.170** | **0.406** |
| Random Forest | 2 | `challenger` | 2.086 | 2.082 | 0.150 | 0.384 |
| Ridge | 1 | `candidate` | 2.115 | 2.110 | 0.127 | 0.354 |

Las variables a priori explican cerca del 17 % de la varianza (el baseline de la media tiene $R^2 \approx 0$). El error típico es de un factor de ~5 a ~8 en reproducciones, así que la predicción sirve para estimar el **orden de magnitud** del alcance, no la cifra exacta. El análisis completo está en [`informe/informe-final.ipynb`](informe/informe-final.ipynb).

## Datos

- **Fuente original:** [The-data-company/TikTok-10M](https://huggingface.co/datasets/The-data-company/TikTok-10M) (Hugging Face). ~6.65 M publicaciones de TikTok en Parquet con metadatos de la publicación, estadísticas de engagement, punto de interés, música y video. Contenido *trending* con POI en Estados Unidos, primavera 2025.
- **Diccionario de datos, variables seleccionadas y sesgos:** [`data/README.md`](data/README.md).

Los datos no se versionan en el repositorio; ver [Ejecución](#ejecución).

## Estructura del repositorio

```
.
├── README.md                  # este archivo
├── pyproject.toml             # dependencias del proyecto
├── uv.lock                    # lockfile de uv
├── .env.example               # nombres de las variables de entorno, sin valores reales
├── .gitignore                 # ignora .env, .venv, datos y artefactos
├── docker-compose.yaml        # servicios api y ui en red host
├── .python-version            # 3.12
├── LICENSE
├── devenv.nix / devenv.yaml / devenv.lock # entorno de desarrollo con devenv
├── data/
│   ├── README.md              # diccionario de datos y notas sobre la fuente
│   ├── raw/                   # datos descargados (ignorado por git)
│   └── interim/               # salida de la preparación (ignorado por git)
├── artifacts/                 # parámetros y resultados generados (ignorado por git)
├── notebooks/                 # cada notebook es un .py de marimo con su export .ipynb
│   ├── custom.css             # estilos para los notebooks de marimo
│   ├── 01-eda                 # análisis exploratorio
│   ├── 02-preparacion-datos   # pipeline de preparación con pandas
│   ├── 03-experimentos        # baselines, 3 familias, HPO con Optuna y comparación
│   ├── _03_experimentos.py    # copia importable de 03 (marimo no importa nombres con guion)
│   ├── 04-seleccion-registro  # criterio de selección, registro, versiones y aliases
│   ├── 05-pipeline            # flow de Prefect: etapas, ejecución y verificación
│   ├── 06-api                 # contrato, carga del modelo y pruebas de la API
│   ├── 07-frontend            # formulario, llamada HTTP, resultado y errores
│   ├── 08-contenedores        # Dockerfiles, contexto, configuración externa, red host y Compose
│   └── 09-despliegue          # instancia EC2, reglas de acceso, despliegue, predicción remota y cierre
├── evidencia/                 # capturas enlazadas desde este README
├── informe/
│   └── informe-final.ipynb    # síntesis del proyecto
├── src/
│   ├── entrenamiento/         # features, HPO y flow de Prefect
│   │   ├── configuracion.py   # features, rutas, semilla, experimento y modelo
│   │   ├── features.py        # preparación sin estado y Pipeline de scikit-learn
│   │   ├── hpo.py             # espacios de búsqueda y estudios de Optuna
│   │   └── flujo.py           # tasks, flow tee-entrenamiento y CLI
│   ├── .dockerignore          # deja fuera .env, .venv y cachés del contexto de construcción
│   ├── api/                   # FastAPI
│   │   ├── Dockerfile         # imagen tee-api-image
│   │   ├── requirements.txt   # dependencias de la imagen, exportadas de uv.lock
│   │   ├── esquemas.py        # contrato de entrada y salida (Pydantic)
│   │   └── main.py            # carga del modelo champion y endpoints
│   ├── frontend/              # Streamlit
│   │   ├── Dockerfile         # imagen tee-ui-image
│   │   ├── requirements.txt   # dependencias de la imagen, exportadas de uv.lock
│   │   ├── .env.example       # API_URL
│   │   ├── app.py             # formulario y presentación del resultado
│   │   └── cliente.py         # llamadas HTTP a la API y manejo de errores
│   └── data_pipelines/        # utilidades de preparación (todavía no las usa el flow)
└── docs/
    └── mapa-sistema-ml.md     # mapa del sistema de ML
```

## Arquitectura

```
            (local)                         (local)               (Databricks)
data/raw ──02──> data/interim ──flow de Prefect──> MLflow + Unity Catalog (@champion)
                                                                │ se descarga al iniciar la API
                       EC2 · Docker Compose (red host)          ▼
navegador ──:8501──> Streamlit 0.0.0.0:8501 ──POST JSON──> FastAPI 127.0.0.1:8000 ──> Pipeline champion ──> reproducciones + versión
```

El modelo se entrena y registra en un proceso aparte. La API solo lo carga y sirve predicciones, y el frontend es un cliente HTTP de la API. Ninguno de los dos reentrena al atender una solicitud. En EC2 solo la interfaz (8501) es accesible desde fuera, y solo para IPs autorizadas; la API escucha en `127.0.0.1` y no se expone.

### Tracking y Registry en Databricks

| Concepto | Valor |
|---|---|
| Tracking URI / Registry URI | `databricks` / `databricks-uc` |
| Experimento de MLflow (Workspace) | `/Shared/pcd-otono-2026-proyecto-tiktok-engaging-estimator` |
| Runs por familia | parent `hpo-<familia>` (`hpo-ridge`, `hpo-random-forest`, `hpo-lightgbm`) con child runs `trial-NNN` y `<familia>-final` |
| Modelo registrado (Unity Catalog) | `workspace.tiktok-engaging-estimator.pcd_otono_2026_tiktok_engaging_estimator` (esquema propio en lugar de `default`) |
| Versión seleccionada | **versión 3 (`lightgbm`) con alias `champion`** |
| Otras versiones | versión 2 (`random_forest`) → `challenger`; versión 1 (`ridge`) → `candidate` |

Todas las familias y corridas usan el mismo experimento. Cada versión del modelo lleva los tags `family`, `final_run_id`, `hpo_run_id`, `test_rmsle`, `test_r2` y `prefect_run`, que la relacionan con su run de procedencia y con la corrida de Prefect que la creó. Los aliases se asignan por RMSLE de test en cada corrida del flow; no se fija de antemano qué familia gana.

La ejecución remota la hizo una sola persona del equipo, con su cuenta de Databricks y su `.env` local; el resto del equipo colaboró en código, análisis, API, interfaz y documentación (detalle en la descripción del PR).

## Ejecución

Todos los comandos se ejecutan desde la **raíz del repositorio**.

### 0. Requisitos

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/)
- Un workspace de Databricks con token personal (para entrenar, registrar y servir el modelo)
- ~15 GB libres en disco para el dataset completo y la caché de Hugging Face
- Para contenedores: Docker con el plugin de Compose (en Windows, Docker Desktop 4.34 o posterior, por `network_mode: host`)
- Para el despliegue: acceso a la consola AWS, una instancia EC2 con Ubuntu 24.04 y su llave `.pem`

### 1. Preparar el ambiente

```bash
git clone https://github.com/matutedevelop/tiktok-engaging-project.git
cd tiktok-engaging-project
uv sync --locked
```

Esto crea `.venv/`, instala las dependencias de `pyproject.toml` respetando `uv.lock` e instala `src/` como paquete, lo que permite importar `entrenamiento`, `api` y `frontend` y cargar el modelo desde la API.

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Llena `.env` (ignorado por git; nunca subas el token):

| Variable | Uso | Quién la usa |
|---|---|---|
| `DATABRICKS_HOST` | `https://<workspace>.cloud.databricks.com` | flow, API |
| `DATABRICKS_TOKEN` | token personal de Databricks | flow, API |
| `TRACKING_URI` / `REGISTRY_URI` | opcionales; por defecto `databricks` y `databricks-uc` | flow, API |
| `MLFLOW_CATALOG` / `MLFLOW_SCHEMA` | catálogo y esquema en Unity Catalog (`workspace` / `tiktok-engaging-estimator`) | flow |
| `MODELO_REGISTRADO` | nombre de tres niveles del modelo, sin `models:/` ni alias | API |
| `ALIAS_MODELO` | alias que carga la API (`champion`) | API |
| `RUTA_MODELO_LOCAL` | opcional: MLflow Model local para probar sin Databricks | API |
| `API_URL` | endpoint que llama el frontend (`http://127.0.0.1:8000/api/v1/predicciones`) | frontend |

### 3. Descargar los datos

Crea `data/raw/get_data.py` con este contenido:

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

Y ejecútalo desde la raíz:

```bash
uv run python data/raw/get_data.py
```

Se generan `data/raw/data.parquet` (~9 GB, dataset completo) y `data/raw/sample_data.parquet` (~1.85 GB, muestra del 20 %, ~2 M filas). `datasets` deja además una copia en la caché de Hugging Face (`~/.cache/huggingface`) que se puede borrar al terminar.

### 4. EDA y preparación de datos

```bash
# Explorar el EDA (opcional)
uv run marimo edit notebooks/01-eda.py

# Preparar la muestra: escribe data/interim/sample_data.parquet
uv run python notebooks/02-preparacion-datos.py sample
```

La preparación selecciona columnas, corrige tipos, imputa y elimina nulos residuales. Su salida en `data/interim/` es la entrada de todos los pasos siguientes.

### 5. Entrenar y registrar el modelo (flow de Prefect)

**Terminal 1**, servidor de Prefect (se queda ocupada; interfaz en <http://127.0.0.1:4200>):

```bash
uv run prefect server start
```

**Terminal 2:**

```bash
uv run prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

# Primera corrida: HPO de las tres familias, entrenamiento, evaluación, registro y aliases
uv run python -m entrenamiento.flujo --hacer-tuning --identificador primera-corrida

# Corridas posteriores: reutiliza los parámetros de artifacts/05-mejores-parametros.json
uv run python -m entrenamiento.flujo --sin-tuning --identificador sin-hpo
```

| Opción | Efecto |
|---|---|
| `--hacer-tuning` / `--sin-tuning` | ejecuta los estudios de Optuna o reutiliza el JSON de parámetros |
| `--identificador <texto>` | nombre del flow run (`tee-<identificador>`), también en el tag `prefect_run` de MLflow |
| `--n-filas <n>` | tamaño de la submuestra (por defecto 300,000) |
| `--n-trials <n>` | fija el número de trials de todas las familias |

Para una prueba rápida antes de la corrida completa: `--hacer-tuning --n-filas 5000 --n-trials 2`.

El flow termina con tres versiones nuevas del modelo en Unity Catalog y los aliases `champion`, `challenger` y `candidate` asignados por RMSLE de test. Para revisar el resultado desde un notebook:

```bash
uv run marimo edit notebooks/05-pipeline.py
```

`notebooks/_03_experimentos.py` y `notebooks/04-seleccion-registro.py` documentan la experimentación y el registro que el flow automatiza; no es necesario correrlos para reproducir el modelo.

### 6. Levantar la API

```bash
uv run fastapi dev src/api/main.py --port 8000
```

Al iniciar, el log muestra `Modelo cargado: <modelo>@champion -> version N (<familia>)`.

- Documentación interactiva: <http://127.0.0.1:8000/docs>
- Disponibilidad: <http://127.0.0.1:8000/health> (200 con la versión cargada, 503 si no hay modelo)

Prueba rápida (en PowerShell usa `curl.exe`):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predicciones \
  -H "Content-Type: application/json" \
  -d '{"duet_info_duet_from_id": "0.0", "duet_enabled": true, "is_ad": false, "item_mute": false,
       "stitch_enabled": true, "music_original": true, "music_duration": 44.0, "vq_score": 74.62,
       "duration": 28, "desc": "Amazon!!!! #amazonfinds #affordablefashion",
       "challenges": ["amazonfinds", "affordablefashion"], "poi_category": "Place and Address",
       "item_comment_status": "0", "city_code": "5096337.0", "poi_tt_type_name_medium": "Places",
       "poi_tt_type_name_tiny": "Province", "music_id": "7.099057143395667e+18",
       "music_author_name": "So Future", "poi_id": "22535865205217746",
       "diversification_id": "10029.0"}'
```

Si se reasigna `champion`, hay que reiniciar la API para que cargue la nueva versión. Detalle del contrato y pruebas en [`notebooks/06-api.ipynb`](notebooks/06-api.ipynb).

### 7. Levantar el frontend

Con la API encendida, en otra terminal:

```bash
uv run streamlit run src/frontend/app.py --server.port 8501
```

La interfaz queda en <http://localhost:8501>. Solo necesita `API_URL`; no usa credenciales de Databricks. Detalle en [`notebooks/07-frontend.ipynb`](notebooks/07-frontend.ipynb).

### 8. Repetir la comprobación

Después de correr el flow (paso 5) y con API y frontend encendidos (pasos 6 y 7):

1. **Prefect** (<http://127.0.0.1:4200>, *Runs*): el flow run `tee-<identificador>` termina en `Completed` y sus 19 task runs también (`configurar-mlflow`, `cargar-datos`, `validar-datos`, `preparar-features`, `dividir-datos`, `parametros-*`, `entrenar-*`, `evaluar-*`, `registrar-*`, `asignar-aliases`, `verificar-champion`).
2. **MLflow en Databricks** (*Experiments* → `/Shared/pcd-otono-2026-proyecto-tiktok-engaging-estimator`): un parent `hpo-<familia>` por familia con sus `trial-NNN` (hiperparámetros y `tuning_rmsle`) y un `<familia>-final` con parámetros, métricas `test_*` y el modelo. Con `--sin-tuning` sólo aparecen runs `*-final` nuevos, sin estudios de HPO.
3. **Unity Catalog** (*Catalog* → `workspace` → esquema → modelo): una versión nueva por familia y los aliases `champion`, `challenger` y `candidate`; los tags de cada versión indican su familia y run.
4. **Notebook** `uv run marimo edit notebooks/05-pipeline.py`: consulta la última corrida en Prefect, el ranking en MLflow, las versiones registradas y carga `@champion` para predecir.
5. **API:** `GET /health` devuelve la misma versión que tiene `champion`; la solicitud de ejemplo del paso 6 responde con una predicción y esa versión, y una entrada inválida responde 422 sin predicción.
6. **Frontend:** el formulario muestra la predicción y la versión que devolvió la API; con la API apagada muestra un error de conexión comprensible.

### 9. API e interfaz con Docker Compose

Las dos piezas se empaquetan en imágenes de Docker (contexto `src/`) y se levantan juntas con [`docker-compose.yaml`](docker-compose.yaml). Detalle en [`notebooks/08-contenedores.ipynb`](notebooks/08-contenedores.ipynb).

| Servicio | Imagen / contenedor | Escucha en | Configuración (`env_file`) |
|---|---|---|---|
| `api` | `tee-api-image:v1` / `tee-api-container` | `127.0.0.1:8000` (solo dentro de la máquina) | `.env` (raíz) |
| `ui` | `tee-ui-image:v1` / `tee-ui-container` | `0.0.0.0:8501` | `src/frontend/.env` (solo `API_URL`) |

Los dos usan `network_mode: host`: comparten la red de la máquina, así que la interfaz llega a la API en `http://127.0.0.1:8000` igual que sin Docker. Ningún `.env` entra a las imágenes (`src/.dockerignore`); la configuración llega al ejecutar.

```bash
cp src/frontend/.env.example src/frontend/.env   # una sola vez; el .env de la raíz es el del paso 2
docker compose config --quiet                    # sin salida si el archivo es válido
docker compose up --build                        # construye y deja los logs en la terminal
```

Salida esperada: `Image tee-api-image:v1 Built`, `Image tee-ui-image:v1 Built` y, en los logs de la API, `Modelo cargado: …@champion -> version 3 (lightgbm)`, `Application startup complete` y `Uvicorn running on http://127.0.0.1:8000`. La interfaz queda en <http://localhost:8501>; el ejemplo por defecto devuelve **10,905 reproducciones**. En otra terminal:

```bash
docker compose ps      # tee-api-container y tee-ui-container en Up
docker compose down    # al terminar
```

Al iniciar, MLflow imprime una lista larga de *dependency mismatches*: la imagen instala solo lo necesario para servir el modelo, no el ambiente completo con el que se registró. No afecta la predicción.

### 10. Despliegue en AWS EC2

Se ejecutan las mismas imágenes y el mismo Compose en una instancia EC2 (Ubuntu 24.04, `us-east-1`). El modelo se sigue cargando desde Databricks. Detalle, evidencia y costos en [`notebooks/09-despliegue.ipynb`](notebooks/09-despliegue.ipynb).

**1. Consola AWS.** Región `us-east-1` → EC2 → *Instances* → la instancia → *Instance state → Start*. Esperar `Running` y los *status checks* y anotar la IPv4 pública, que cambia en cada arranque. En el grupo de seguridad, *Inbound rules*:

| Puerto | Origen |
|---|---|
| 22 (SSH) | *My IP* |
| 8501 | *My IP* y la `IP/32` de cada persona que vaya a probar |
| 8000 | **sin regla**: la API no se expone |

Ninguna regla con `0.0.0.0/0`.

**2. Conexión y Docker** (en Git Bash):

```bash
ssh -i ~/.ssh/<llave>.pem ubuntu@IP_PUBLICA
sudo docker --version && sudo docker compose version   # si falla, instalar Docker Engine desde el repositorio oficial para Ubuntu
```

**3. Código y configuración privada** (en la EC2; el repositorio es público y se clona sin token):

```bash
git clone https://github.com/matutedevelop/tiktok-engaging-project.git
cd tiktok-engaging-project
cp .env.example .env && chmod 600 .env
nano .env                                    # DATABRICKS_HOST y DATABRICKS_TOKEN reales; nunca se imprime ni se sube
cp src/frontend/.env.example src/frontend/.env
git check-ignore .env src/frontend/.env      # imprime las dos rutas
```

**4. Arrancar y comprobar:**

```bash
sudo docker compose up --build               # esperar "Modelo cargado … version 3" y "Application startup complete"
# en otra terminal SSH:
curl -i http://127.0.0.1:8000/health         # 200 con la versión del modelo
curl -i -X POST http://127.0.0.1:8000/api/v1/predicciones -H "Content-Type: application/json" -d '{}'   # 422
sudo docker compose logs --tail 30 api
```

Desde otra computadora autorizada: `http://IP_PUBLICA:8501` → la barra lateral dice *Disponible · modelo versión 3 (lightgbm)* y el ejemplo por defecto devuelve **10,905 reproducciones**; el log de la API registra `Prediccion version 3: 10905 reproducciones`. `http://IP_PUBLICA:8000` no responde.

**5. Cierre:**

```bash
# Ctrl+C en la terminal del up
sudo docker compose down
sudo docker compose ps     # vacío
exit
```

En la consola: *Instance state → **Stop*** (nunca *Terminate*) hasta ver `Stopped`. Detenida, la instancia deja de cobrar el cómputo, pero el volumen EBS se conserva y sigue generando costo.

### Problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `Failed to reach API at http://127.0.0.1:4200/api/` | el servidor de Prefect no está encendido | `uv run prefect server start` en otra terminal |
| el flow falla en `cargar-datos` con `FileNotFoundError` | falta `data/interim/sample_data.parquet` | correr el paso 4 |
| el flow falla en `configurar-mlflow` | faltan `DATABRICKS_HOST` / `DATABRICKS_TOKEN` o el token venció | revisar `.env` |
| `parametros-*` falla con `--sin-tuning` | no existe el JSON de parámetros | correr una vez con `--hacer-tuning` |
| el Registry rechaza el nombre del modelo | catálogo o esquema incorrectos | revisar `MLFLOW_CATALOG` y `MLFLOW_SCHEMA` |
| la API no encuentra `FrequencyEncoder` al cargar el modelo | el proyecto no está instalado en el ambiente | `uv sync --locked` |
| `/health` responde 503 | el modelo no se pudo cargar (token, nombre o alias) | revisar `MODELO_REGISTRADO`, `ALIAS_MODELO` y el log de la API |
| el frontend dice "No fue posible conectarse con la API" | la API está apagada o `API_URL` es incorrecta | levantar la API (paso 6) y revisar `API_URL` |
| `docker compose` falla con `env file … src/frontend/.env not found` | falta el `.env` de la interfaz | `cp src/frontend/.env.example src/frontend/.env` |
| con Compose la interfaz no llega a la API | *host networking* deshabilitado (Docker Desktop) o la API aún carga el modelo | habilitar *host networking*; esperar `Application startup complete` en `docker compose logs api` |
| `ssh` da `Connection timed out` | instancia detenida, IP pública nueva o la regla 22 no tiene la IP actual | revisar estado e IP; volver a elegir *My IP* en la regla 22 |
| `ssh` da `Permission denied (publickey)` | usuario o llave incorrectos | usar `ubuntu@` y la `.pem` de esa instancia |
| `http://IP_PUBLICA:8501` no abre | la regla 8501 no incluye la IP de quien prueba | agregar su `IP/32` (la consulta en <https://checkip.amazonaws.com>) |

## Evidencia

En las capturas remotas se ocultaron correo, avatar, URL del workspace e identificadores personales.

| Caso | Qué muestra | Captura |
|---|---|---|
| Experimento remoto | ruta `/Shared/pcd-otono-2026-proyecto-tiktok-engaging-estimator` con `hpo-ridge`, `hpo-random-forest` y `hpo-lightgbm` | [`evidencia/experimento-remoto.png`](evidencia/experimento-remoto.png) |
| HPO de Ridge | trials de `hpo-ridge` con `alpha`, `tuning_rmsle` y mejor resultado | [`evidencia/hpo-ridge.png`](evidencia/hpo-ridge.png) |
| HPO de Random Forest | trials de `hpo-random-forest` con hiperparámetros, `tuning_rmsle` y mejor resultado | [`evidencia/hpo-random-forest.png`](evidencia/hpo-random-forest.png) |
| HPO de LightGBM | trials de `hpo-lightgbm` con hiperparámetros, `tuning_rmsle` y mejor resultado | [`evidencia/hpo-lightgbm.png`](evidencia/hpo-lightgbm.png) |
| Comparación final | modelos de los tres runs `*-final` del flow con sus métricas de test (`test_mae_log`, `test_r2`); los mejores parámetros de cada familia están en las capturas de HPO | [`evidencia/comparacion-final.png`](evidencia/comparacion-final.png) |
| Model Registry | nombre completo, tres versiones con familia y run, alias `champion` | [`evidencia/model-registry.png`](evidencia/model-registry.png) |
| Prefect | flow run `tee-primera-corrida` en `Completed` con sus 19 task runs | [`evidencia/prefect.png`](evidencia/prefect.png) |
| API: entrada válida | solicitud, respuesta con predicción y versión del modelo | [`evidencia/api-prediccion-valida.png`](evidencia/api-prediccion-valida.png) |
| API: entrada inválida | solicitud y respuesta 422 de validación, sin predicción | [`evidencia/api-entrada-invalida.png`](evidencia/api-entrada-invalida.png) |
| Frontend integrado | formulario con los valores usados y predicción obtenida de la API | [`evidencia/frontend-prediccion.png`](evidencia/frontend-prediccion.png) |
| Frontend integrado | solicitud y respuesta de la API para esa misma predicción | [`evidencia/frontend-solicitud-respuesta.png`](evidencia/frontend-solicitud-respuesta.png) |
| Frontend: error de validación | mensaje comprensible ante una entrada inválida | [`evidencia/frontend-error-validacion.png`](evidencia/frontend-error-validacion.png) |
| Frontend: error de conexión | mensaje comprensible con la API apagada | [`evidencia/frontend-error-conexion.png`](evidencia/frontend-error-conexion.png) |
| Ambiente y builds | `tee-api-image:v1` y `tee-ui-image:v1` en `Built` y contenedores `Created` | [`evidencia/ambiente-builds.png`](evidencia/ambiente-builds.png) |
| Integración local | `docker compose ps` con los dos servicios en `Up` | [`evidencia/integracion-local-servicios.png`](evidencia/integracion-local-servicios.png) |
| Integración local | log de la API: modelo versión 3 cargado, escucha en `127.0.0.1` y predicción | [`evidencia/integracion-local-logs-api.png`](evidencia/integracion-local-logs-api.png) |
| Integración local | predicción en `localhost:8501` con versión del modelo | [`evidencia/integracion-local-prediccion.png`](evidencia/integracion-local-prediccion.png) |
| Instancia y red | instancia `t3.medium` en `Running` con `3/3 checks passed` | [`evidencia/instancia-running.png`](evidencia/instancia-running.png) |
| Instancia y red | reglas de entrada: 22 y 8501 desde una sola IP, sin 8000 | [`evidencia/instancia-reglas-entrada.png`](evidencia/instancia-reglas-entrada.png) |
| Despliegue | build de las dos imágenes en EC2 | [`evidencia/despliegue-ec2-build.png`](evidencia/despliegue-ec2-build.png) |
| Despliegue | API en EC2 con `@champion -> version 3 (lightgbm)` y `Application startup complete` | [`evidencia/despliegue-ec2-modelo-cargado.png`](evidencia/despliegue-ec2-modelo-cargado.png) |
| Predicción remota | interfaz abierta desde otra computadora: 10,905 reproducciones, versión 3 | [`evidencia/despliegue-ec2-prediccion-remota.png`](evidencia/despliegue-ec2-prediccion-remota.png) |
| Entrada inválida en EC2 | `/health` 200 y `POST {}` 422 desde dentro del servidor | [`evidencia/despliegue-ec2-entrada-invalida.png`](evidencia/despliegue-ec2-entrada-invalida.png) |
| Logs | log de la API con la 422 y `Prediccion version 3: 10905 reproducciones` desde `127.0.0.1` | [`evidencia/despliegue-ec2-logs.png`](evidencia/despliegue-ec2-logs.png) |
| Cierre | `docker compose down` y `ps` vacío en EC2 | [`evidencia/cierre-docker-down.png`](evidencia/cierre-docker-down.png) |
| Cierre | diálogo de *Stop*: el volumen EBS se sigue cobrando | [`evidencia/cierre-stop-costos.png`](evidencia/cierre-stop-costos.png) |
| Cierre | instancia en `Stopped` | [`evidencia/cierre-instancia-stopped.png`](evidencia/cierre-instancia-stopped.png) |

En las capturas de AWS y EC2 se ocultaron IPs, identificadores de instancia y de grupo de seguridad, la cuenta y el hostname.

## Qué no se versiona

`.venv/`, `.env`, `data/raw/`, `data/interim/`, `artifacts/`, secretos, credenciales y cualquier artefacto generado. Ver `.gitignore`.

## Trabajo futuro

1. Entrenar el `champion` con la muestra completa o el dataset entero y ampliar el presupuesto de HPO.
2. Sustituir TF-IDF + SVD por embeddings de texto para `desc` y `challenges`.
3. Interpretar el modelo con importancia por permutación o SHAP.
4. Mover la imputación de `02` dentro del `Pipeline`.
5. Endurecer el despliegue: HTTPS con un proxy inverso, IP fija o dominio, arranque automático (`up -d` con `restart: unless-stopped`) y monitoreo.

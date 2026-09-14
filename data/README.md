# Datos del proyecto

## Fuente

https://huggingface.co/datasets/The-data-company/TikTok-10M

## Licencia o condiciones de uso

other
https://huggingface.co/datasets/The-data-company/TikTok-10M/blob/main/README.md

Indica el permiso identificado y enlaza su evidencia.

## Descripción

> pd. Parte del siguiente contenido en especial el diccionario de los datos fue hecho con ayuda de gemini


Cada fila representa una publicación individual (video) en TikTok junto con todas sus métricas de interacción en un momento determinado, los metadatos de su creador, el audio utilizado, su geolocalización (si aplica) y sus configuraciones de privacidad o publicidad.  




### Diccionario de Características del Dataset TikTok-10M

Este documento sirve como referencia oficial del esquema y diccionario de características para **`The-data-company/TikTok-10M`**, el cual contiene 10 millones de publicaciones de TikTok con  metadatos multimodales.

---

### Estructura General de los Datos

El dataset está particionado en formato Parquet y contiene métricas numéricas, indicadores de interacción, perfiles de usuarios, metadatos de audio, geolocalización (POI) y configuraciones algorítmicas.



---

#### Diccionario Detallado de Características

##### 1. Metadatos del Post e Identificadores Principales

| Feature | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `id` | `int64` | Identificador único de la publicación asignado por TikTok. |
| `url` | `string` | URL directa del video en la web (`https://tiktok.com/@/video/<id>`). |
| `create_time` | `int64` | Timestamp Unix (en segundos) que indica cuándo fue publicado el video. |
| `collected_time` | `string` | Marca de tiempo (UTC string) de cuando se extrajo el registro para el dataset. |
| `desc` | `string` | Descripción o pie de foto escrito por el creador (incluye menciones y hashtags en texto). |
| `duration` | `int64` | Duración total del video en segundos. |
| `vq_score` | `float64` | *Video Quality Score*: Calificación estimada sobre la calidad técnica del video. |
| `share_cover` | `string` | Lista en formato JSON/Array con las URLs de las portadas/miniaturas del video. |

---

##### 2. Métricas de Interacción y Rendimiento

| Feature | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `play_count` | `int64` | Número acumulado de reproducciones / vistas del video. |
| `digg_count` | `int64` | Número acumulado de "Me gusta" (likes). |
| `comment_count` | `int64` | Cantidad total de comentarios realizados en el video. |
| `share_count` | `int64` | Cantidad de veces que el video fue compartido por otros usuarios. |
| `collect_count` | `int64` | Cantidad de veces que los usuarios guardaron el video en sus favoritos. |
| `stats_time` | `string` | Marca de tiempo (UTC string) en la que se registraron estas métricas de interacción. |

---

##### 3. Atributos del Usuario y Creador

| Feature | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `user_id` | `int64` | Identificador numérico único de la cuenta del creador. |
| `user_verified` | `string` | Estado de verificación de la cuenta (`'t'` para verdadero, `'f'` para falso). |
| `user_tt_seller` | `string` | Indica si el creador está registrado como vendedor en TikTok Shop (`'t'` / `'f'`). |
| `user_avatar_larger` | `string` | URL a la foto de perfil del usuario en alta resolución. |
| `user_avatar_medium` | `string` | URL a la foto de perfil del usuario en resolución media. |
| `user_avatar_thumb` | `string` | URL a la foto de perfil en formato miniatura. |

---

##### 4. Propiedades de Audio y Música

| Feature | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `music_id` | `float64` | Identificador único de la pista de audio utilizada en la publicación. |
| `music_title` | `string` | Título del sonido o canción. |
| `music_author_name` | `string` | Nombre del artista o creador del sonido. |
| `music_album` | `string` | Nombre del álbum asociado (si aplica). |
| `music_duration` | `float64` | Duración de la pista de audio en segundos. |
| `music_original` | `string` | Indica si el audio es un sonido original del usuario (`'t'`) o música comercial (`'f'`). |
| `music_play_url` | `string` | Enlace directo para reproducir/descargar el archivo de audio (.mp3). |

---

##### 5. Categorización de Contenido y Hashtags

| Feature | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `challenges` | `string` | Lista codificada en JSON con los nombres de hashtags o retos etiquetados en el video. |
| `diversification_id` | `float64` | ID interno del sistema de recomendación para diversificar el feed. |

---

##### 6. Geolocalización y Puntos de Interés (POI)

| Feature | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `poi_id` | `int64` | Identificador único del Punto de Interés (lugar) etiquetado en la publicación. |
| `poi_name` | `string` | Nombre del establecimiento o sitio físico etiquetado. |
| `address` | `string` | Dirección física exacta del lugar etiquetado. |
| `city` | `string` | Nombre de la ciudad de la ubicación. |
| `city_code` | `float64` | Código numérico interno asignado a la ciudad. |
| `country_code` | `int64` | Código numérico correspondiente al país. |
| `poi_category` | `string` | Categoría primaria de la ubicación (ej. *Food and Drink*, *Outdoors and Traveling*). |
| `father_poi_id` | `string` | ID del POI padre para ubicaciones anidadas. |
| `father_poi_name` | `string` | Nombre del POI padre para ubicaciones anidadas. |
| `poi_tt_type_code` | `string` | Código de tipo de POI interno de TikTok. |
| `poi_tt_type_name_super` | `string` | Categoría de clasificación superior (ej. *Company*, *Places*). |
| `poi_tt_type_name_medium` | `string` | Categoría de clasificación intermedia (ej. *Office and Industrial*, *Province*). |
| `poi_tt_type_name_tiny` | `string` | Clasificación de subcategoría específica (ej. *Business Facility*, *Amusement Park*). |

---

##### 7. Ajustes del Post y Publicidad

| Feature | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `is_ad` | `string` | Indicador de contenido publicitario (`'t'` = anuncio patrocinado, `'f'` = post orgánico). |
| `duet_enabled` | `string` | Indica si la opción de hacer Dúos está habilitada por el creador (`'t'` / `'f'`). |
| `duet_display` | `int64` | Configuración de visualización para Dúos. |
| `duet_info_duet_from_id` | `float64` | ID del video original en caso de que este post sea la respuesta de un Dúo. |
| `stitch_enabled` | `string` | Indica si la opción de Pegar (Stitch) está habilitada (`'t'` / `'f'`). |
| `stitch_display` | `int64` | Configuración de visualización para la función Pegar/Stitch. |
| `item_comment_status` | `int64` | Estado de la sección de comentarios (ej. habilitados vs desactivados). |
| `item_mute` | `string` | Indica si el audio del video fue silenciado (`'t'` / `'f'`). |
| `item_control_can_repost` | `string` | Permiso asignado para permitir que otros resuban/revisen el post (`'t'` / `'f'`). |
| `official_item` | `string` | Indicador de contenido oficial o verificado (`'t'` / `'f'`). |
| `original_item` | `string` | Estado de publicación original (`'t'` / `'f'`). |
| `share_enabled` | `string` | Permiso asignado para compartir la publicación (`'t'` / `'f'`). |

---




## Target y features iniciales

Para este conjunto de datos, la variable objetivo (target) principal a predecir es play_count, ya que representa el alcance total y el éxito de visualizaciones de un video en la plataforma. A primera vista, las features con mayor potencial analítico y predictivo para este target son las métricas de interacción inicial como digg_count (likes), share_count y comment_count, el contenido textual y temático extraído de desc y challenges (hashtags), el impacto del creador mediante user_id y su estado user_verified, así como el rendimiento del sonido o música a través de music_id y los datos contextuales o de localización como poi_category y duration

## Obtención de los datos


los datos son obtenibles atraves de HugginFace, para descargarlos y generar la muestra utilizada para el EDA de forma reproducible, Utilizamos el siguiente codigo dentro de /data/raw/get_data.py y ejecutando `uv run python data/raw/get_data.py ` desde la raiz del proyecto

```python
# get_data.py

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










## Archivos locales



- `data/raw/get_data.py`




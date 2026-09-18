# Mapa del sistema de ML — TEE (TikTok Engage Estimator)

## 1. Contexto

**Problema:** 
Para creadores, marcas y agencias, el alcance de un video de TikTok (reproducciones) es la métrica que define si una pieza funcionó. Ese resultado solo se conoce *después* de publicar: qué grabar, con qué música, qué duración, qué hashtags y desde qué lugar se decide sin ninguna estimación cuantitativa. Las herramientas de analítica de TikTok muestran resultados a posteriori y la intuición sobre "qué funciona" es anecdótica.

TEE estima `play_count` usando exclusivamente información disponible antes de publicar. Formalmente: dado un vector $x$ observable al momento de publicar (configuración, música, texto, hashtags, POI, duración, calidad), se busca $\hat{f}(x) \approx \mathbb{E}[\log(1 + \text{play\_count}) \mid x]$, o equivalentemente un estimador del orden de magnitud del alcance.

**Por qué el problema es difícil (antecedentes):** 
La literatura predice bien la popularidad cuando hay señal *temprana* (Szabo & Huberman, 2010) o *social* del autor (Khosla, 2014, donde los factores sociales dominan pero el contenido intrínseco aporta señal no trivial). Para video corto, Chen(2016) y Ling(2022) confirman que texto, hashtags y música sí informan. TEE elimina deliberadamente la señal temprana y el dataset no expone seguidores ni historial del autor, así que el modelo trabaja en el escenario más adverso: casi solo factores intrínsecos y de configuración.

**Usuario / sistema consumidor:** 
** Una persona creadora o equipo de contenido que, al configurar la publicación, quiere conocer el orden de magnitud esperado del alcance (10³ vs. 10⁶) y qué palancas moverían la estimación.

**Decisión que apoya la predicción.** 
Ajustar decisiones controlables antes de publicar: música, duración, hashtags/descripción, POI etiquetado y configuración de propagación (duet, stitch, repost, share, comentarios). Objetivo específico 5 del proyecto: identificar qué decisiones controlables tienen efecto medible.

**Hipótesis de trabajo.** 
Las variables a priori explican una fracción modesta pero estadísticamente significativa de la varianza de `log(play_count)`: suficiente para discriminar órdenes de magnitud, no para valores puntuales.

- *Supuesto:* el orden de magnitud es útil para el creador aunque la estimación puntual sea imprecisa.
- *Supuesto:* el usuario acepta una estimación condicionada a contenido *trending* con POI en EE. UU. (primavera 2025); fuera de ese dominio la predicción no es confiable.
- *Pregunta pendiente:* ¿el consumidor final es un formulario interactivo (un video a la vez) o un reporte por lote sobre varios videos planeados?
- *Pregunta pendiente:* ¿la salida debe incluir explicaciones por feature (qué palanca mover) o solo la estimación?

## 2. Nivel de datos

### Evidencia del EDA (`notebooks/01-eda.py`, muestra de ~2 M registros)

- **Fuente:** 
[The-data-company/TikTok-10M](https://huggingface.co/datasets/The-data-company/TikTok-10M)
  (Hugging Face), ≈ 10 M publicaciones en Parquet (~9 GB). Licencia reportada como "other"; declarado construido con datos públicos para investigación. Contenido *trending* con POI en Estados Unidos, primavera 2025. Diccionario completo en [`data/README.md`](../data/README.md).
- **Unidad de observación:** 
una fila = una publicación (video) con sus métricas de interacción en un momento dado, metadatos de creador, audio, POI y configuración.
- **Target:** 
`play_count`, con varianza y sesgo altos (cola pesada), visible en boxplot e histograma en escala log.
- **Tipos incorrectos en la fuente:** banderas como `is_ad` vienen como cadenas `'t'`/`'f'`; campos temporales como cadena; `music_id`, `city_code` y `diversification_id` como `float64`.
- **Nulos:** individualmente marginales por columna, pero exigir registros completos deja <1 % de los datos. Imputando solo numéricas y eliminando residuales se descarta ≈15 % de la muestra.
- **Duplicados de `id`:** 
proporción muy pequeña.
- **Booleanas:** 
las nueve están muy desbalanceadas (ninguna cerca del 50 %) y sus distribuciones de `play_count` (violin plots en log) son casi idénticas entre clases; las diferencias aparecen justo en las más desbalanceadas, lo que las hace poco confiables.
- **Colinealidades:** 
`city`/`city_code` (`city` tiene >50 % nulos), `poi_category`/`poi_tt_type_name_super`, `poi_tt_type_name_tiny`/`poi_tt_type_code`; `duration`/`music_duration` altamente correlacionadas.
- **Constantes:** 
`country_code`, `duet_display`, `stitch_display`.
- **Numéricas vs. target:** 
sin relación aparente en scatter/pairplot.
- **vq_score:** 
masa de valores en 0.
- **Cardinalidad:** 
solo ~5 nominales tienen ≤364 clases (estimables con medias condicionales); el resto supera 1 000 valores distintos. `desc` tiene una clase dominante pero demasiada variedad; `address` muestra un codo de Pareto claro que permite agrupar la cola en `other`.
- **`music_title` vs. `music_id`:** 
cardinalidades muy distintas; hipótesis: muchas canciones comparten título.

### Decisión inicial

- **Volumen de trabajo:** 
todo el pipeline opera sobre `data/raw/sample_data.parquet` (20 % del dataset, `random_state=69`, ~1.8 GB, ~2 M filas antes de limpiar, ~1.7 M después), generado por `data/raw/get_data.py`. Escalar al dataset completo queda condicionado a memoria disponible y a que el modelo demuestre señal sobre la muestra.
- **Features a priori (28):**
  - Booleanas (9): `duet_enabled`, `is_ad`, `item_mute`, `item_control_can_repost`, `official_item`, `original_item`, `share_enabled`, `stitch_enabled`, `music_original`.
    Justificación: configuración explícita del creador que afecta la propagación.
  - Nominales (16): `desc`, `address`, `poi_name`, `city_code`, `poi_category`, `poi_tt_type_name_medium`, `poi_tt_type_name_tiny`, `challenges`, `music_id`, `music_title`, `music_album`, `duet_info_duet_from_id`, `music_author_name`, `poi_id`, `diversification_id`, `item_comment_status`.
    Justificación: texto, hashtags, música y ubicación son las modalidades que
    la literatura identifica como informativas y que se fijan antes de publicar.
  - Numéricas (3): `duration`, `music_duration`, `vq_score`.
    Justificación: características de producción controlables.
- **Excluidas por leakage:** 
`digg_count`, `comment_count`, `share_count`, `collect_count` (material del target, no features), `create_time`, `stats_time`, `collected_time`.
- **Excluidas por privacidad / ausencia de señal social deliberada:**
`user_id`, `user_verified`, `user_tt_seller`, avatares, `url`, `share_cover`, `music_play_url`.
- **Descartadas por redundancia o degeneración:** 
`city`, `poi_tt_type_name_super`, `poi_tt_type_code`, `country_code`, `duet_display`, `stitch_display`.
- **Target modelado:** `log(1 + play_count)`.
- **Preparación (`notebooks/02-preparacion.py`, pandas, `.pipe()`):**
  `select_columns → cast_types → impute_numericals → drop_residual_nulls`.
  - `cast_types`: booleanas a `boolean` nullable (evita convertir nulos a
    `True` silenciosamente) y a `bool` nativo al final; `*_id` a `category`;
    nominales a `string`; numéricas y target con coerción numérica.
  - `impute_numericals`: univariada por columna, mediana si |skew| > 1,
    media en caso contrario (Little & Rubin, 2019).
  - Salida prevista: `data/processed/data_clean.parquet` (hoy solo se escribe
    si se descomenta la última celda; no versionado).
- **Validaciones a implementar:** esquema de columnas y tipos; rango de
  `duration`, `music_duration` y `vq_score`; ausencia de columnas de
  engagement y de usuario en la matriz de features; proporción de nulos por
  columna antes de imputar; unicidad de `id`.

### Supuestos

- Imputar solo numéricas es suficiente; las nominales con nulos se agrupan
  (`other`/`unknown`) o se descartan.
- Los datos son representativos de contenido orgánico *trending* en EE. UU.;
  el desempeño no generaliza a otros países ni periodos y el sesgo de muestreo
  puede inflar el desempeño aparente.
- Una muestra del 20 % (`random_state=69`) reproduce las distribuciones del
  total.
- Eliminar los `id` duplicados no introduce sesgo.

### Preguntas pendientes

- ¿`log(1 + y)` o Box-Cox? Se decidirá comparando residuales.
- ¿Umbral para agrupar colas en `other` en `address`, `city_code`,
  `poi_tt_type_name_*` y `diversification_id`?
- ¿`music_title` vs. `music_id`: se conserva uno o los dos?
- ¿Bandera `no_vq` para `vq_score == 0` o discretización?
- ¿Es el 15 % descartado por nulos un sesgo sistemático (p. ej., videos sin
  POI o sin música)?
- ¿Cómo se representa `challenges` (lista JSON de hashtags) y `desc` (texto
  libre con menciones y hashtags) como features?


### Estrategia de particiones

Train / busqued/ prueba = 80 / 10 / 10, estratificado por cuantiles de
`log(1 + play_count)`. El conjunto de prueba se toca una sola vez al final.
Alternativa a evaluar: partición temporal por `create_time` para simular
"predecir videos futuros".


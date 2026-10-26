"""Interfaz de Streamlit: reúne los datos del video y pide la predicción a la API.

Arranque desde la raíz del repositorio (con la API ya encendida):

    uv run streamlit run src/frontend/app.py --server.port 8501
"""

import math

import streamlit as st

from frontend.cliente import (
    API_URL,
    RMSLE_CHAMPION,
    consultar_salud,
    orden_de_magnitud,
    parsear_hashtags,
    predecir,
    rango_tipico,
)

CATEGORIAS_POI = [
    "Education",
    "Food and Drink",
    "Health",
    "Hotels and Lodging",
    "Office and Industrial",
    "Other",
    "Outdoors and Traveling",
    "Place and Address",
    "Recreation and Sport",
    "Residential",
    "Service",
    "Shopping",
]

st.set_page_config(page_title="TEE — TikTok Engage Estimator", layout="wide")
st.title("TEE — TikTok Engage Estimator")
st.write(
    "Estima cuántas reproducciones tendrá un video de TikTok usando sólo información "
    "disponible **antes de publicarlo**."
)

with st.sidebar:
    st.header("Estado de la API")
    salud = consultar_salud()
    if salud.ok:
        st.success(f"Disponible · modelo versión {salud.cuerpo['version']} ({salud.cuerpo['familia']})")
    elif salud.codigo == 503:
        st.warning("La API responde, pero no tiene modelo cargado.")
    else:
        st.error("API no disponible.")
    st.caption(f"Endpoint: `{API_URL}`")

with st.form("prediccion"):
    video, configuracion, musica, ubicacion = st.tabs(
        ["Video", "Configuración", "Música", "Ubicación"]
    )
    with video:
        desc = st.text_area(
            "Descripción", "Amazon!!!! #amazonfinds #affordablefashion", max_chars=20_000
        )
        hashtags = st.text_input(
            "Hashtags (challenges)",
            "amazonfinds, affordablefashion",
            help="Separados por comas o espacios; el '#' es opcional.",
        )
        duration = st.number_input("Duración del video (s)", 0, 3600, 28, step=1)
        vq_score = st.number_input(
            "Calidad del video (vq_score)",
            0.0,
            100.0,
            74.62,
            step=0.5,
            help="Calidad técnica estimada por TikTok; 0 = sin calificación.",
        )
    with configuracion:
        duet_enabled = st.toggle("Permitir duetos", value=True)
        stitch_enabled = st.toggle("Permitir stitch", value=True)
        item_mute = st.toggle("Audio silenciado", value=False)
        is_ad = st.toggle("Es contenido publicitario", value=False)
        item_comment_status = st.selectbox(
            "Estado de comentarios (item_comment_status)",
            ["0", "3"],
            help="Valor tal como lo reporta TikTok; en los datos sólo aparecen 0 y 3.",
        )
        duet_info_duet_from_id = st.text_input(
            "ID del video original si es un dueto", "0.0", help="'0.0' si no es un dueto."
        )
    with musica:
        music_original = st.toggle("Sonido original del creador", value=True)
        music_author_name = st.text_input("Autor del sonido", "So Future")
        music_id = st.text_input(
            "ID de la pista (music_id)",
            "7.099057143395667e+18",
            help="Con el mismo formato del dataset; un ID desconocido se trata como nuevo.",
        )
        music_duration = st.number_input(
            "Duración de la pista (s)", 0.0, 3600.0, 44.0, step=1.0
        )
    with ubicacion:
        poi_category = st.selectbox(
            "Categoría del lugar", CATEGORIAS_POI, index=CATEGORIAS_POI.index("Place and Address")
        )
        poi_tt_type_name_medium = st.text_input("Tipo de lugar (intermedio)", "Places")
        poi_tt_type_name_tiny = st.text_input("Subtipo de lugar", "Province")
        poi_id = st.text_input("ID del lugar (poi_id)", "22535865205217746")
        city_code = st.text_input(
            "Código de ciudad", "5096337.0", help="Con el formato del dataset, p. ej. '5096337.0'."
        )
        diversification_id = st.text_input(
            "ID de diversificación", "10029.0", help="'Unknown' si no se conoce."
        )
    enviado = st.form_submit_button("Estimar reproducciones", type="primary")

if enviado:
    # Mismas llaves que SolicitudPrediccion en src/api/esquemas.py
    entrada = {
        "duet_info_duet_from_id": duet_info_duet_from_id.strip(),
        "duet_enabled": bool(duet_enabled),
        "is_ad": bool(is_ad),
        "item_mute": bool(item_mute),
        "stitch_enabled": bool(stitch_enabled),
        "music_original": bool(music_original),
        "music_duration": float(music_duration),
        "vq_score": float(vq_score),
        "duration": int(duration),
        "desc": desc,
        "challenges": parsear_hashtags(hashtags),
        "poi_category": poi_category,
        "item_comment_status": item_comment_status,
        "city_code": city_code.strip(),
        "poi_tt_type_name_medium": poi_tt_type_name_medium.strip(),
        "poi_tt_type_name_tiny": poi_tt_type_name_tiny.strip(),
        "music_id": music_id.strip(),
        "music_author_name": music_author_name.strip(),
        "poi_id": poi_id.strip(),
        "diversification_id": diversification_id.strip(),
    }
    resultado = predecir(entrada)

    if resultado.ok:
        r = resultado.cuerpo
        minimo, maximo = rango_tipico(r["log_reproducciones"])
        st.metric("Reproducciones estimadas", f"{r['reproducciones_estimadas']:,}")
        st.caption(
            f"Modelo: {r['modelo']} · alias: {r['alias']} · versión: {r['version']} · "
            f"familia: {r['familia']}"
        )
        st.info(
            f"Orden de magnitud: **{orden_de_magnitud(r['reproducciones_estimadas'])}** "
            f"reproducciones. Con el error típico del modelo (RMSLE ≈ {RMSLE_CHAMPION:.2f}, un "
            f"factor de ~{math.exp(RMSLE_CHAMPION):.1f}×), el valor real suele estar entre "
            f"**{minimo:,}** y **{maximo:,}**.\n\n"
            "Es una estimación del alcance, no una garantía: el modelo se entrenó con contenido "
            "*trending* con ubicación en EE. UU. (primavera 2025) y no conoce seguidores ni "
            "historial del creador."
        )
    else:
        st.error(resultado.mensaje)

    with st.expander("Solicitud y respuesta de la API"):
        st.write(f"`POST {API_URL}` → **{resultado.codigo or 'sin respuesta'}**")
        st.write("Solicitud enviada")
        st.json(entrada)
        st.write("Respuesta recibida")
        st.json(resultado.cuerpo or {})

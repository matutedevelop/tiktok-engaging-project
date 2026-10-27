"""Contrato HTTP de la API: qué se recibe, qué se devuelve y cómo se valida.

Los campos de `SolicitudPrediccion` son exactamente `COLUMNAS_ENTRADA`, las columnas crudas
de `data/interim` que recibe el modelo registrado. Este módulo no carga MLflow, así que el
notebook puede validar ejemplos sin levantar el servidor.
"""

import json
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from entrenamiento.configuracion import COLUMNAS_ENTRADA

# Las 12 categorías que aparecen en data/interim
CategoriaPOI = Literal[
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

EJEMPLO_VALIDO = {
    "duet_info_duet_from_id": "0.0",
    "duet_enabled": True,
    "is_ad": False,
    "item_mute": False,
    "stitch_enabled": True,
    "music_original": True,
    "music_duration": 44.0,
    "vq_score": 74.62,
    "duration": 28,
    "desc": "Amazon!!!! #amazonfinds #affordablefashion",
    "challenges": ["amazonfinds", "affordablefashion"],
    "poi_category": "Place and Address",
    "item_comment_status": "0",
    "city_code": "5096337.0",
    "poi_tt_type_name_medium": "Places",
    "poi_tt_type_name_tiny": "Province",
    "music_id": "7.099057143395667e+18",
    "music_author_name": "So Future",
    "poi_id": "22535865205217746",
    "diversification_id": "10029.0",
}


def _identificador(descripcion: str):
    return Field(min_length=1, max_length=40, description=descripcion)


class SolicitudPrediccion(BaseModel):
    """Un video que todavía no se publica, descrito con las columnas crudas del modelo."""

    # Un campo con otro nombre (p. ej. un typo) es un error, no se ignora en silencio
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EJEMPLO_VALIDO]})

    duet_info_duet_from_id: str = Field(
        default="0.0",
        max_length=40,
        description="ID del video original si la publicación es un dueto; '0.0' si no lo es.",
    )
    duet_enabled: bool = Field(description="El creador permite hacer duetos.")
    is_ad: bool = Field(description="Contenido publicitario patrocinado.")
    item_mute: bool = Field(description="El audio del video está silenciado.")
    stitch_enabled: bool = Field(description="El creador permite hacer stitch.")
    music_original: bool = Field(description="Sonido original del creador (no música comercial).")
    music_duration: float = Field(ge=0, le=3600, description="Duración de la pista, en segundos.")
    vq_score: float = Field(
        ge=0, le=100, description="Calidad técnica estimada del video (0 = sin calificación)."
    )
    duration: int = Field(ge=0, le=3600, description="Duración del video, en segundos.")
    desc: str = Field(
        max_length=20_000, description="Pie de foto con menciones y hashtags en texto."
    )
    challenges: list[str] = Field(
        max_length=100, description="Hashtags etiquetados, sin '#'. Puede ser una lista vacía."
    )
    poi_category: CategoriaPOI = Field(description="Categoría del punto de interés etiquetado.")
    item_comment_status: Literal["0", "3"] = Field(
        description="Estado de los comentarios tal como lo reporta TikTok ('0' o '3')."
    )
    city_code: str = _identificador("Código de la ciudad, p. ej. '5096337.0'.")
    poi_tt_type_name_medium: str = _identificador("Tipo intermedio del POI, p. ej. 'Places'.")
    poi_tt_type_name_tiny: str = _identificador("Subtipo del POI, p. ej. 'Province'.")
    music_id: str = _identificador("ID de la pista de audio, p. ej. '7.099057143395667e+18'.")
    music_author_name: str = Field(
        min_length=1, max_length=300, description="Artista o creador del sonido."
    )
    poi_id: str = _identificador("ID del punto de interés.")
    diversification_id: str = _identificador(
        "ID interno de diversificación, p. ej. '10029.0'; 'Unknown' si no se conoce."
    )

    def a_dataframe(self) -> pd.DataFrame:
        """Una fila con las columnas y los tipos con los que se registró el modelo."""
        fila = self.model_dump()
        # 02-preparacion-datos imputó las descripciones vacías con "NA"
        fila["desc"] = fila["desc"].strip() or "NA"
        # En data/interim `challenges` es la lista serializada como JSON
        fila["challenges"] = json.dumps(fila["challenges"])
        return pd.DataFrame([fila])[COLUMNAS_ENTRADA]


class RespuestaPrediccion(BaseModel):
    reproducciones_estimadas: int = Field(
        description="Estimación de play_count: expm1 de la salida del modelo, redondeada."
    )
    log_reproducciones: float = Field(
        description="Salida directa del modelo, en escala log(1 + play_count)."
    )
    modelo: str = Field(description="Nombre del modelo registrado que respondió.")
    alias: str = Field(description="Alias con el que se resolvió la versión.")
    version: str = Field(description="Versión concreta del modelo registrado que se cargó.")
    familia: str | None = Field(default=None, description="Familia del estimador (tag `family`).")


class EstadoServicio(BaseModel):
    status: Literal["ok", "degradado"]
    modelo_cargado: bool
    modelo: str | None = None
    alias: str | None = None
    version: str | None = None
    familia: str | None = None
    version_api: str

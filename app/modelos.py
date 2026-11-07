"""Contratos de entrada de la API de chat documental."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MensajeChat(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    conversacion_id: int | None = Field(default=None, description="Vacío = conversación nueva")
    mensaje: str = Field(min_length=1, max_length=1000)

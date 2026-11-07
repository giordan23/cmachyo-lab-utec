"""Comportamiento del LLM para respuestas documentales, sin llamadas externas en pruebas."""
import pytest

from app.llm import ESQUEMA_RESPUESTA, LLMError, MockLLM


def test_esquema_solo_permite_respuesta_y_citas_de_fuentes():
    assert set(ESQUEMA_RESPUESTA["required"]) == {"respuesta", "fuentes_citadas"}
    assert "dictamen" not in ESQUEMA_RESPUESTA["properties"]


def test_mock_responde_con_fragmento_y_pagina():
    contexto = {"fuentes": [{"id": "p1-1", "pagina": 1, "titulo": "Página 1", "texto": "TEA 0.30%"}]}
    respuesta = MockLLM().generar("instrucciones", contexto, ESQUEMA_RESPUESTA, "respuesta")
    assert respuesta.datos["fuentes_citadas"] == ["p1-1"]
    assert "0.30%" in respuesta.datos["respuesta"] and "página 1" in respuesta.datos["respuesta"].lower()


def test_mock_no_fabrica_respuesta_si_no_hay_fuentes():
    with pytest.raises(LLMError, match="sin fuentes"):
        MockLLM().generar("instrucciones", {"fuentes": []}, ESQUEMA_RESPUESTA, "respuesta")

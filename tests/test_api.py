"""Pruebas de la API para el prototipo de ahorro de Caja Huancayo."""
import pytest
from fastapi.testclient import TestClient

from app import metricas
from app.agente_riesgo import AgenteRiesgo
from app.db import Repositorio
from app.llm import MockLLM
from app.main import Servicios, create_app
from app.router import ResultadoRuta


class RouterFijo:
    def enrutar(self, texto, historial=None):
        t = texto.lower()
        ruta = "consulta_ahorros" if any(k in t for k in ("ahorro", "depósito", "documentos", "requisito", "condición")) else "fuera_de_alcance"
        return ResultadoRuta(ruta, 0.9, "fijo", modelo="fijo")


@pytest.fixture
def cliente(settings, rag):
    s = Servicios(settings, RouterFijo(), AgenteRiesgo(rag, MockLLM()), Repositorio(settings.db_path))
    with TestClient(create_app(s)) as c:
        yield c


def test_consulta_de_ahorro_responde_directo_con_citas(cliente, sana):
    sana["pregunta"] = "¿Qué documentos mencionan cuentas de ahorro?"
    d = cliente.post("/api/solicitudes", json=sana).json()
    assert d["estado"] == "RESPONDIDA"
    assert d["citas"]


def test_fuera_de_alcance_se_deriva(cliente, sana):
    sana["pregunta"] = "¿A qué hora cierra la agencia?"
    d = cliente.post("/api/solicitudes", json=sana).json()
    assert d["estado"] == "DERIVADA" and "Caja Huancayo" in d["respuesta"]


def test_metricas_expuestas(cliente, sana):
    cliente.post("/api/solicitudes", json={**sana, "pregunta": "¿Qué información hay sobre ahorro?"})
    texto = cliente.get("/metrics").text
    assert 'lab06_solicitudes_total' in texto
    assert 'lab06_rag_score_bucket' in texto


def test_paginas_html(cliente):
    pagina = cliente.get("/").text
    assert "Caja Huancayo" in pagina and "Asistente de ahorro" in pagina
    assert cliente.get("/health").json() == {"status": "ok"}

"""Flujo HTTP de consulta sobre el PDF real de benchmarking."""
import pytest
from fastapi.testclient import TestClient

from app.agente_documental import AgenteDocumental
from app.db import Repositorio
from app.llm import MockLLM
from app.main import Servicios, create_app
from app.router import ResultadoRuta


class RouterFijo:
    def enrutar(self, texto, historial=None):
        ruta = "fuera_de_alcance" if "hora" in texto.lower() else "consulta_benchmark"
        return ResultadoRuta(ruta, 0.9, "fijo", modelo="fijo")


@pytest.fixture
def cliente(settings, rag):
    s = Servicios(settings, RouterFijo(), AgenteDocumental(rag, MockLLM()), Repositorio(settings.db_path))
    with TestClient(create_app(s)) as c:
        yield c


def test_consulta_del_pdf_responde_con_cita_del_documento(cliente):
    d = cliente.post("/api/chat", json={"mensaje": "¿Cuánto cuesta consultar saldo en Kasnet?"}).json()
    asistente = d["mensajes"][1]
    assert asistente["meta"]["ruta"] == "consulta_benchmark"
    assert asistente["meta"]["citas"]
    assert any(c["pagina"] == 3 for c in asistente["meta"]["citas"])
    assert "Kasnet" in asistente["texto"]


def test_sin_evidencia_se_abstiene(cliente):
    d = cliente.post("/api/chat", json={"mensaje": "¿A qué hora abre la agencia?"}).json()
    assert d["mensajes"][1]["meta"]["ruta"] == "fuera_de_alcance"
    assert "benchmarking" in d["mensajes"][1]["texto"].lower()


def test_consulta_invalida_es_422(cliente):
    for cuerpo in ({"mensaje": ""}, {"mensaje": "x" * 1001}):
        assert cliente.post("/api/chat", json=cuerpo).status_code == 422


def test_conversacion_persiste_y_se_puede_reanudar(cliente):
    primera = cliente.post("/api/chat", json={"mensaje": "Compara ahorro en soles"})
    assert primera.status_code == 200
    conversacion = primera.json()["conversacion_id"]
    respuesta = cliente.get(f"/api/chat/{conversacion}").json()
    assert len(respuesta["mensajes"]) == 2
    assert cliente.get("/api/chat/999").status_code == 404


def test_endpoints_del_flujo_de_credito_ya_no_existen(cliente):
    assert cliente.post("/api/solicitudes", json={}).status_code == 404
    assert cliente.get("/api/cola").status_code == 404
    assert cliente.get("/analista").status_code == 404


def test_metricas_expuestas_son_del_flujo_documental(cliente):
    cliente.post("/api/chat", json={"mensaje": "Compara ahorro en soles"})
    texto = cliente.get("/metrics").text
    for serie in ("lab06_chat_mensajes_total", "lab06_rag_score_bucket", "lab06_rag_sin_contexto_total",
                  "lab06_router_confianza_bucket"):
        assert serie in texto
    for serie in ("lab06_dictamen_agente_total", "lab06_cola_pendiente", "lab06_decision_analista_total"):
        assert serie not in texto


def test_paginas_html(cliente):
    html = cliente.get("/").text
    assert "Benchmarking" in html
    assert "tasas" in html.lower()

"""El flujo completo por HTTP: asesor → router → agente → cola → analista → asesor."""
import pytest
from fastapi.testclient import TestClient

from app import metricas
from app.agente_riesgo import AgenteRiesgo
from app.db import Repositorio
from app.llm import MockLLM
from app.main import Servicios, create_app
from app.router import ResultadoRuta

TOKEN = {"X-Analista-Token": "token-de-prueba"}


class RouterFijo:
    """El router se prueba en test_router.py; aquí elegimos la ruta con una palabra clave para aislar la API."""

    def enrutar(self, texto, historial=None):
        ruta = "consulta_producto" if "documentos" in texto else "fuera_de_alcance" if "hora" in texto else "evaluacion_credito"
        return ResultadoRuta(ruta, 0.9, "fijo", modelo="fijo")


@pytest.fixture
def cliente(settings, rag):
    s = Servicios(settings, RouterFijo(), AgenteRiesgo(rag, MockLLM()), Repositorio(settings.db_path))
    with TestClient(create_app(s)) as c:
        yield c


def test_flujo_completo_aprobacion(cliente, sana):
    r = cliente.post("/api/solicitudes", json=sana)
    assert r.status_code == 200
    d = r.json()
    assert d["estado"] == "PENDIENTE_ANALISTA" and d["ruta"] == "evaluacion_credito"
    assert "dictamen" not in str(d).lower() and "APROBAR" not in str(d)      # el asesor no ve la propuesta antes del humano

    cola = cliente.get("/api/cola", headers=TOKEN).json()
    assert [f["id"] for f in cola] == [d["id"]] and cola[0]["resultado"]["dictamen"] == "APROBAR"

    r = cliente.post(f"/api/solicitudes/{d['id']}/decision", headers=TOKEN, json={"decision": "APROBADA", "analista": "Ana"})
    assert r.status_code == 200
    final = cliente.get(f"/api/solicitudes/{d['id']}").json()
    assert final["estado"] == "APROBADA" and final["analista"] == "Ana" and final["citas"] and final["cuota"] > 0
    assert cliente.get("/api/cola", headers=TOKEN).json() == []


def test_apartarse_del_agente_exige_comentario(cliente, sana):
    id_ = cliente.post("/api/solicitudes", json=sana).json()["id"]
    r = cliente.post(f"/api/solicitudes/{id_}/decision", headers=TOKEN, json={"decision": "RECHAZADA", "analista": "Ana"})
    assert r.status_code == 422 and "comentario" in r.json()["detail"]
    r = cliente.post(f"/api/solicitudes/{id_}/decision", headers=TOKEN,
                     json={"decision": "RECHAZADA", "analista": "Ana", "comentario": "Ingresos no sustentados con boletas."})
    assert r.status_code == 200 and r.json()["decision"] == "RECHAZADA"


def test_no_se_decide_dos_veces(cliente, sana):
    id_ = cliente.post("/api/solicitudes", json=sana).json()["id"]
    cuerpo = {"decision": "APROBADA", "analista": "Ana"}
    assert cliente.post(f"/api/solicitudes/{id_}/decision", headers=TOKEN, json=cuerpo).status_code == 200
    assert cliente.post(f"/api/solicitudes/{id_}/decision", headers=TOKEN, json=cuerpo).status_code == 409


def test_solicitud_inexistente(cliente):
    assert cliente.get("/api/solicitudes/999").status_code == 404
    assert cliente.post("/api/solicitudes/999/decision", headers=TOKEN, json={"decision": "APROBADA", "analista": "Ana"}).status_code == 404


@pytest.mark.parametrize("headers, codigo", [({}, 401), ({"X-Analista-Token": "otro"}, 401)])
def test_el_asesor_no_puede_ver_la_cola_ni_aprobar(cliente, sana, headers, codigo):
    id_ = cliente.post("/api/solicitudes", json=sana).json()["id"]
    assert cliente.get("/api/cola", headers=headers).status_code == codigo
    assert cliente.post(f"/api/solicitudes/{id_}/decision", headers=headers, json={"decision": "APROBADA", "analista": "x1"}).status_code == codigo


def test_sin_token_configurado_la_aprobacion_esta_deshabilitada(cliente, monkeypatch):
    monkeypatch.setattr(cliente.app.state.s.settings, "analista_token", "")
    assert cliente.get("/api/cola", headers=TOKEN).status_code == 503


@pytest.mark.parametrize("campo, valor", [("ingreso_mensual", 0), ("dni", "4012345A"), ("producto", "hipotecario"),
                                          ("plazo_meses", 0), ("calificacion_sbs", "Perdida"), ("pregunta", "")])
def test_formulario_invalido_es_422_sin_llamar_a_nadie(cliente, sana, campo, valor):
    sana[campo] = valor
    assert cliente.post("/api/solicitudes", json=sana).status_code == 422


def test_consulta_de_producto_responde_directo_con_citas(cliente, sana):
    sana["pregunta"] = "¿Qué documentos debe presentar un trabajador independiente?"
    d = cliente.post("/api/solicitudes", json=sana).json()
    assert d["estado"] == "RESPONDIDA" and d["citas"][0]["id"] == "3.3"
    assert cliente.get("/api/cola", headers=TOKEN).json() == []


def test_fuera_de_alcance_se_deriva(cliente, sana):
    sana["pregunta"] = "¿A qué hora cierra la agencia?"
    d = cliente.post("/api/solicitudes", json=sana).json()
    assert d["estado"] == "DERIVADA" and "Derívala" in d["respuesta"]


def test_metricas_expuestas(cliente, sana):
    cliente.post("/api/solicitudes", json=sana)
    texto = cliente.get("/metrics").text
    for serie in ("lab06_solicitudes_total{ruta=\"evaluacion_credito\"}", "lab06_cola_pendiente", "lab06_dictamen_agente_total",
                  "lab06_decision_analista_total", "lab06_router_confianza_bucket", "lab06_rag_score_bucket"):
        assert serie in texto, serie


def test_cola_pendiente_refleja_la_base(cliente, sana):
    cliente.post("/api/solicitudes", json=sana)
    cliente.post("/api/solicitudes", json=sana)
    assert metricas.COLA_PENDIENTE._value.get() == 2


def test_paginas_html(cliente):
    assert "Asesor de ventas" in cliente.get("/asesor").text
    assert "Analista de riesgos" in cliente.get("/analista").text
    assert cliente.get("/", follow_redirects=False).status_code == 307

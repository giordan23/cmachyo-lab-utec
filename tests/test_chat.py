"""Flujo del chat del asistente documental sobre ahorro de Caja Huancayo."""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import metricas
from app.agente_ahorro import AgenteRiesgo
from app.db import Repositorio
from app.llm import LLMError, MockLLM
from app.main import Servicios, create_app
from app.router import ResultadoRuta


class RouterFijo:
    """Elige la ruta por palabras clave"""

    def __init__(self):
        self.historiales = []

    def enrutar(self, texto, historial=None):
        self.historiales.append(historial)
        t = texto.lower()
        ruta = "consulta_ahorros" if any(k in t for k in ("ahorro", "depósito", "documentos", "requisito", "condición")) else "fuera_de_alcance"
        return ResultadoRuta(ruta, 0.9, "fijo", modelo="fijo", inyeccion=0.02)


class LLMSinReformular(MockLLM):
    def generar(self, instrucciones, contexto, esquema, nombre):
        if nombre == "reformulacion":
            raise LLMError("caído")
        return super().generar(instrucciones, contexto, esquema, nombre)


def armar(settings, rag, llm=None):
    router = RouterFijo()
    s = Servicios(settings, router, AgenteRiesgo(rag, llm or MockLLM()), Repositorio(settings.db_path))
    return s, router


@pytest.fixture
def chat(settings, rag):
    s, router = armar(settings, rag)
    with TestClient(create_app(s)) as c:
        c.router_fijo = router
        yield c


def decir(c, texto, conv=None):
    r = c.post("/api/chat", json={"conversacion_id": conv, "mensaje": texto})
    assert r.status_code == 200, r.text
    return r.json()


def test_primera_pregunta_crea_conversacion_y_responde_con_citas(chat):
    d = decir(chat, "¿Qué documentos hablan de cuentas de ahorro?")
    usuario, asistente = d["mensajes"]
    assert usuario["rol"] == "usuario" and asistente["rol"] == "asistente"
    assert asistente["meta"]["ruta"] == "consulta_ahorros" and asistente["meta"]["citas"][0]["id"] == "3.3"


def test_seguimiento_se_reformula_con_el_historial(chat):
    d1 = decir(chat, "¿Qué requisitos hay para un depósito a plazo?")
    d2 = decir(chat, "¿Y para ahorro programado?", d1["conversacion_id"])
    meta = d2["mensajes"][1]["meta"]
    assert "depósito" in meta["consulta"] or "ahorro" in meta["consulta"]
    assert chat.router_fijo.historiales[-1] == ["¿Qué requisitos hay para un depósito a plazo?"]


def test_si_falla_la_reformulacion_se_busca_con_el_mensaje_tal_cual(settings, rag):
    s, _ = armar(settings, rag, LLMSinReformular())
    with TestClient(create_app(s)) as c:
        d1 = decir(c, "¿Qué requisitos hay para un depósito a plazo?")
        d2 = decir(c, "¿Y para ahorro programado?", d1["conversacion_id"])
        assert d2["mensajes"][1]["meta"]["consulta"] == "¿Y para ahorro programado?"


def test_fuera_de_alcance(chat):
    assert "alcance" in decir(chat, "¿A qué hora cierra la agencia?")["mensajes"][1]["texto"].lower()


@pytest.mark.parametrize("cuerpo", [{"mensaje": "   "}, {"mensaje": ""}, {"mensaje": "x" * 1001}])
def test_mensaje_invalido_es_422(chat, cuerpo):
    assert chat.post("/api/chat", json=cuerpo).status_code == 422


def test_conversacion_inexistente_es_404(chat):
    assert chat.post("/api/chat", json={"conversacion_id": 999, "mensaje": "hola"}).status_code == 404
    assert chat.get("/api/chat/999").status_code == 404


def test_migracion_de_una_base_sin_chat(tmp_path):
    """La base del volumen de Docker de la versión anterior no tiene conversacion_id: se agrega al abrirla."""
    ruta = tmp_path / "vieja.db"
    cx = sqlite3.connect(ruta)
    cx.execute(
        "CREATE TABLE solicitudes (id INTEGER PRIMARY KEY AUTOINCREMENT, creada_en REAL NOT NULL, estado TEXT NOT NULL, "
        "ruta TEXT NOT NULL, solicitud TEXT NOT NULL, enrutamiento TEXT NOT NULL, resultado TEXT NOT NULL, decision TEXT, "
        "analista TEXT, comentario TEXT, decidida_en REAL)"
    )
    cx.commit()
    cx.close()
    repo = Repositorio(ruta)
    assert repo.obtener(repo.crear("PENDIENTE_ANALISTA", "consulta_ahorros", {}, {}, {}, conversacion_id=None))["conversacion_id"] is None

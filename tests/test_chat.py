"""Persistencia, contexto y respuestas del chat documental."""
import pytest
from fastapi.testclient import TestClient

from app.agente_documental import AgenteDocumental
from app.db import Repositorio
from app.llm import LLMError, MockLLM
from app.main import Servicios, create_app
from app.router import ResultadoRuta


class RouterFijo:
    def __init__(self):
        self.historiales = []

    def enrutar(self, texto, historial=None):
        self.historiales.append(historial)
        ruta = "fuera_de_alcance" if "contraseña" in texto.lower() else "consulta_benchmark"
        return ResultadoRuta(ruta, 0.9, "fijo", modelo="fijo")


class LLMSinReformular(MockLLM):
    def generar(self, instrucciones, contexto, esquema, nombre):
        if nombre == "reformulacion":
            raise LLMError("caído")
        return super().generar(instrucciones, contexto, esquema, nombre)


def armar(settings, rag, llm=None):
    router = RouterFijo()
    servicios = Servicios(settings, router, AgenteDocumental(rag, llm or MockLLM()), Repositorio(settings.db_path))
    return servicios, router


@pytest.fixture
def chat(settings, rag):
    servicios, router = armar(settings, rag)
    with TestClient(create_app(servicios)) as cliente:
        cliente.router_fijo = router
        yield cliente


def decir(cliente, texto, conv=None):
    r = cliente.post("/api/chat", json={"conversacion_id": conv, "mensaje": texto})
    assert r.status_code == 200, r.text
    return r.json()


def test_pregunta_documental_responde_con_fuentes(chat):
    d = decir(chat, "¿Cuál es la TEA del ahorro en soles?")
    asesor, asistente = d["mensajes"]
    assert asesor["rol"] == "asesor" and asistente["rol"] == "asistente"
    assert asistente["meta"]["ruta"] == "consulta_benchmark"
    assert asistente["meta"]["citas"] and asistente["meta"]["citas"][0]["pagina"] == 1


def test_pregunta_de_seguimiento_conserva_historial(chat):
    d1 = decir(chat, "¿Qué TEA tiene el ahorro en soles?")
    d2 = decir(chat, "¿Y el ahorro en dólares?", d1["conversacion_id"])
    assert chat.router_fijo.historiales[-1] == ["¿Qué TEA tiene el ahorro en soles?"]
    assert "dólares" in d2["mensajes"][1]["meta"]["consulta"]


def test_si_falla_la_reformulacion_usa_el_mensaje_original(settings, rag):
    servicios, _ = armar(settings, rag, LLMSinReformular())
    with TestClient(create_app(servicios)) as cliente:
        d1 = decir(cliente, "¿Qué TEA tiene el ahorro en soles?")
        d2 = decir(cliente, "¿Y el ahorro en dólares?", d1["conversacion_id"])
        assert d2["mensajes"][1]["meta"]["consulta"] == "¿Y el ahorro en dólares?"


def test_fuera_de_alcance_se_deriva(chat):
    assert "benchmarking" in decir(chat, "¿Cuál es la contraseña de mi correo?")["mensajes"][1]["texto"].lower()


@pytest.mark.parametrize("mensaje", ["", "   ", "x" * 1001])
def test_mensaje_invalido_es_422(chat, mensaje):
    assert chat.post("/api/chat", json={"mensaje": mensaje}).status_code == 422


def test_conversacion_inexistente_es_404(chat):
    assert chat.post("/api/chat", json={"conversacion_id": 999, "mensaje": "hola"}).status_code == 404
    assert chat.get("/api/chat/999").status_code == 404

import pytest

from app.db import NoExiste, Repositorio


@pytest.fixture
def repo(tmp_path):
    return Repositorio(tmp_path / "chat.db")


def test_conversacion_y_mensajes_persisten_con_tildes(repo):
    conv = repo.crear_conversacion()
    repo.agregar_mensaje(conv, "asesor", "¿Qué TEA tiene el ahorro en dólares?")
    repo.agregar_mensaje(conv, "asistente", "Según la página 1.", {"ruta": "consulta_benchmark"})
    mensajes = repo.mensajes(conv)
    assert mensajes[0]["texto"] == "¿Qué TEA tiene el ahorro en dólares?"
    assert mensajes[1]["meta"]["ruta"] == "consulta_benchmark"


def test_paginacion_de_mensajes(repo):
    conv = repo.crear_conversacion()
    primero = repo.agregar_mensaje(conv, "asesor", "uno")
    repo.agregar_mensaje(conv, "asistente", "dos")
    assert [m["texto"] for m in repo.mensajes(conv, primero["id"])] == ["dos"]


def test_conversacion_inexistente(repo):
    assert not repo.existe_conversacion(99)

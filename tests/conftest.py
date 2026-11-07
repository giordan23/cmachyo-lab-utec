"""Configuración de pruebas: proveedores simulados, PDF real del caso y datos aislados por prueba."""
import os

import pytest

# Antes de importar app.*: que ningún .env del alumno cambie el comportamiento de las pruebas.
for k in ("ROUTER_PROVIDER", "LLM_PROVIDER", "EMBEDDINGS_PROVIDER", "TYPESAFE_API_KEY", "OPENAI_API_KEY",
          "PDF_PATH", "INDICE_PATH", "DB_PATH", "ROUTER_MIN_CONFIDENCE", "ROUTER_MIN_SCORE", "RAG_MIN_SCORE", "RAG_TOP_K", "INYECCION_UMBRAL"):
    os.environ.pop(k, None)
os.environ.update(ROUTER_PROVIDER="hash", LLM_PROVIDER="mock", EMBEDDINGS_PROVIDER="hash")

# Los import de app.* van DENTRO de cada fixture: así las pruebas del paso 3 corren aunque todavía no hayas creado
# los archivos de los pasos siguientes (cada fixture solo importa lo que necesita).


@pytest.fixture(scope="session")
def pdf(tmp_path_factory):
    from app.config import DATA_DIR
    ruta = DATA_DIR / "Benchmarking.pdf"
    assert ruta.is_file(), f"No se encontró el PDF de evaluación: {ruta}"
    return ruta


@pytest.fixture
def settings(tmp_path, pdf, monkeypatch):
    from app.config import Settings
    monkeypatch.setenv("PDF_PATH", str(pdf))
    monkeypatch.setenv("INDICE_PATH", str(tmp_path / "indice.npz"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "chat.db"))
    return Settings()


@pytest.fixture
def rag(settings):
    from app.embeddings import HashEmbedder
    from app.ingesta import construir_indice
    from app.rag import RAG
    construir_indice(settings, HashEmbedder())
    return RAG(settings.indice_path, HashEmbedder(), settings.rag_min_score, settings.rag_top_k)


# Solicitud ficticia usada en pruebas de guardas, herramientas y agente.
# Corresponde al manual educativo ficticio de rules/politicas.py; no es un producto real de Caja Huancayo.
SOLICITUD_SANA = {
    "dni": "40123456", "nombre": "Rosa Quispe", "edad": 35, "ingreso_mensual": 3500, "deuda_mensual": 400,
    "calificacion_sbs": "Normal", "antiguedad_meses": 24, "tipo_trabajador": "dependiente", "producto": "consumo",
    "monto": 12000, "plazo_meses": 24, "pregunta": "¿Cuáles son los requisitos para este producto de ahorro?",
}


@pytest.fixture
def sana():
    return dict(SOLICITUD_SANA)

import pytest

from app.embeddings import HashEmbedder
from app.ingesta import extraer_fragmentos, indice_vigente
from app.rag import IndiceError, RAG


def test_indice_usa_el_pdf_real_y_cubre_sus_paginas(pdf, rag):
    assert {f["pagina"] for f in extraer_fragmentos(pdf)} == {1, 2, 3}
    assert len(rag.fragmentos) > 3
    assert all(f.texto.strip() for f in rag.fragmentos)


@pytest.mark.parametrize("pregunta,pagina", [
    ("¿Cuál es la TEA del ahorro en soles en cada banco?", 1),
    ("¿Qué TEA tiene el Ahorro Cash en dólares?", 1),
    ("¿Qué tasas tiene el plazo fijo entre 91 y 180 días?", 2),
    ("¿Qué tasas se muestran para el crédito vehicular?", 3),
    ("¿Cuánto cuesta consultar el saldo en Kasnet?", 3),
    ("¿Qué porcentaje de ITF indica el documento?", 3),
    ("¿Qué ocurre si cancelo anticipadamente un plazo fijo?", 2),
])
def test_busqueda_recupera_pagina_de_la_evidencia(rag, pregunta, pagina):
    assert pagina in {f.pagina for f in rag.buscar(pregunta)}


def test_consulta_ajena_al_pdf_no_recupera_contexto(rag):
    assert rag.buscar("xqzv wkjh plmn") == []


def test_indice_de_otro_embedder_falla_con_mensaje_claro(rag, settings):
    with pytest.raises(IndiceError, match="python -m app.ingesta"):
        RAG(settings.indice_path, HashEmbedder(dim=512), 0.2, 3)


def test_sin_indice_falla_con_mensaje_claro(tmp_path):
    with pytest.raises(IndiceError, match="python -m app.ingesta"):
        RAG(tmp_path / "no_existe.npz", HashEmbedder(), 0.2, 3)


def test_hash_del_pdf_invalida_el_indice_al_cambiar_el_documento(pdf, settings, tmp_path):
    from app.ingesta import construir_indice

    embedder = HashEmbedder()
    construir_indice(settings, embedder)
    assert indice_vigente(settings, embedder)
    pdf_cambiado = tmp_path / "benchmark_modificado.pdf"
    pdf_cambiado.write_bytes(settings.pdf_path.read_bytes() + b"\n")
    settings.pdf_path = pdf_cambiado
    assert not indice_vigente(settings, embedder)

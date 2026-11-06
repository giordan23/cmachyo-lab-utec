"""Guardas deterministas de la respuesta (C6): grounding, citas, reservas y abstención.

Las pruebas usan el corpus real de benchmarking (`data/Benchmarking.pdf`) y los 22 casos del set de oro
(`data/rag_gold.json`). No hay LLM en ninguno de estos tests: las guardas son funciones puras sobre el
texto de la respuesta, el corpus y las páginas del PDF.
"""
import json
from pathlib import Path

import pytest

from app.guardas_respuesta import (ESTADO_ABSTENER, ESTADO_CON_RESERVA, ESTADO_PUBLICAR,
                                   GuardasRespuesta, PaginasPDF, formatear_cita, normalizar, numeros)
from app.ingesta import extraer_fragmentos
from app.rag import Fragmento

RAIZ = Path(__file__).resolve().parents[1]
PDF = RAIZ / "data" / "Benchmarking.pdf"
GOLD = RAIZ / "data" / "rag_gold.json"


@pytest.fixture(scope="module")
def fragmentos():
    return [Fragmento(**f) for f in extraer_fragmentos(PDF)]


@pytest.fixture(scope="module")
def guardas(fragmentos):
    return GuardasRespuesta(fragmentos, PaginasPDF(PDF))


@pytest.fixture(scope="module")
def gold():
    return json.loads(GOLD.read_text(encoding="utf-8"))


def _fragmento_de_caso(caso, fragmentos):
    """El fragmento del corpus que contiene lo que el caso espera como fuente."""
    for fuente in caso.get("fuentes_esperadas", []):
        for token in fuente.get("contiene", []):
            for f in fragmentos:
                if normalizar(token).lower() in normalizar(f.texto).lower():
                    return f
    return None


def _revisar(guardas, fragmentos, caso):
    f = _fragmento_de_caso(caso, fragmentos)
    return guardas.revisar(caso["respuesta_esperada"], [f.id] if f else [], caso["pregunta"])


# --------------------------------------------------------------------- los 22 casos del set de oro
def test_los_22_casos_del_set_de_oro_pasan(guardas, fragmentos, gold):
    """Cada caso del gold: el estado esperado (publicar / con reserva / abstener) y la página real."""
    fallos = []
    for caso in gold:
        r = _revisar(guardas, fragmentos, caso)
        esperado = ESTADO_ABSTENER if caso.get("debe_abstenerse") else None
        if esperado and r.estado != esperado:
            fallos.append(f"{caso['id']}: esperaba abstención y dio {r.estado}")
        if not esperado and r.estado == ESTADO_ABSTENER:
            fallos.append(f"{caso['id']}: abstención inesperada ({r.motivo})")
        paginas_esperadas = {f["pagina"] for f in caso.get("fuentes_esperadas", [])}
        paginas_obtenidas = {p for c in r.citas for p in c["paginas"]}
        if not paginas_esperadas <= paginas_obtenidas:
            fallos.append(f"{caso['id']}: páginas {sorted(paginas_obtenidas)} != {sorted(paginas_esperadas)}")
    assert not fallos, fallos


def test_reparto_de_estados_es_el_del_set_de_oro(guardas, fragmentos, gold):
    """2 publicar / 17 con reserva / 3 abstener (docs/specs/2026-11-05-c6-validacion-gold-22.md)."""
    estados = [_revisar(guardas, fragmentos, c).estado for c in gold]
    assert estados.count(ESTADO_PUBLICAR) == 2
    assert estados.count(ESTADO_CON_RESERVA) == 17
    assert estados.count(ESTADO_ABSTENER) == 3


# --------------------------------------------------------------------- G2 · grounding
def test_cifra_inexistente_en_el_corpus_se_abstiene(guardas, fragmentos):
    f = fragmentos[0]
    r = guardas.revisar("El ahorro en soles rinde 7.77% anual.", [f.id], "¿Cuál es la TEA del ahorro en soles?")
    assert r.estado == ESTADO_ABSTENER
    assert r.motivo == "cifra_sin_grounding"
    assert "7.77" in r.cifras_sin_grounding
    assert "7.77" in r.respuesta and "abstención" in r.respuesta


def test_las_cifras_de_las_citas_no_cuentan_como_grounding(guardas, fragmentos):
    """'Art. 1 (pág. 2)' no es una cifra de la respuesta: no puede tumbarla."""
    f = fragmentos[0]
    r = guardas.revisar("Banco A publica 0.30% (Art. 1, pág. 1).", [f.id], "¿Cuál es la TEA del ahorro en soles?")
    assert r.cifras_sin_grounding == []


def test_cifra_del_corpus_pasa_el_grounding(guardas, fragmentos):
    f = fragmentos[0]
    r = guardas.revisar("Banco A publica 0.30% TEA.", [f.id], "¿Cuál es la TEA del ahorro en soles?")
    assert r.cifras_sin_grounding == []


# --------------------------------------------------------------------- G1 · citas
def test_la_pagina_de_la_cita_es_la_pagina_real(guardas, fragmentos):
    """La ingesta etiqueta el fragmento con la página de su título; el texto se derrama a la siguiente."""
    f = fragmentos[0]
    assert f.pagina == 1
    r = guardas.revisar("Para cancelaciones de hasta 30 días no se generan intereses.",
                        [f.id], "¿Qué ocurre si cancelo anticipadamente un plazo fijo?")
    assert r.citas[0]["paginas"] == [2]
    assert r.citas[0]["pagina_declarada"] == 1


def test_cita_de_un_fragmento_con_texto_en_dos_paginas(guardas, fragmentos):
    """'Préstamo personal' (pág. 2) y 'TCEA ... no se puede calcular' (pág. 3) están en el mismo fragmento."""
    f = fragmentos[1]
    r = guardas.revisar("No se puede determinar la TCEA exacta del préstamo personal.",
                        [f.id], "¿Cuál es la TCEA exacta del préstamo personal en Banco A?")
    assert r.citas[0]["paginas"] == [2, 3]


def test_la_pagina_no_se_toma_del_primer_numero_que_aparece(guardas, fragmentos):
    """'1080 días a más' está en la página 2, aunque '1080' también aparezca en la 1 (720–1080 días)."""
    f = fragmentos[0]
    r = guardas.revisar("Banco A indica 1.00% para todos los rangos de monto de la tabla.",
                        [f.id], "¿Qué TEA aparece para depósitos a plazo de 1080 días a más?")
    assert r.citas[0]["paginas"] == [2]


def test_formateo_de_cita(guardas, fragmentos):
    r = guardas.revisar("Banco A publica 0.30% TEA.", [fragmentos[0].id], "¿Cuál es la TEA del ahorro en soles?")
    texto = formatear_cita(r.citas[0])
    assert texto.startswith("Art. 1 (")
    assert "pág. 1" in texto


# --------------------------------------------------------------------- G3 · reservas
def test_reserva_de_dato_ficticio(guardas, fragmentos):
    f = fragmentos[2]
    r = guardas.revisar("Banco A cobra S/ 3.10 y Banco B supone S/ 2.50 en el cajero Global Net.",
                        [f.id], "Compara la comisión de retiro en cajeros Global Net.")
    assert r.estado == ESTADO_CON_RESERVA
    assert any("supuesto ficticio" in x for x in r.reservas)


def test_reserva_de_tasas_inventadas(guardas, fragmentos):
    f = fragmentos[1]
    r = guardas.revisar("El crédito vehicular muestra 16.90% en Banco A y 15.90% en Banco B.",
                        [f.id], "¿Qué tasas se muestran para el crédito vehicular?")
    assert r.estado == ESTADO_CON_RESERVA
    assert any("ficticias" in x for x in r.reservas)


def test_reserva_de_fila_uniforme_solo_de_su_propia_fila(guardas, fragmentos):
    """La fila '1080 días a más -> 1.00% x 6' se marca; una comisión de S/ 1.00, no."""
    f = fragmentos[0]
    con_reserva = guardas.revisar("Banco A indica 1.00% para todos los rangos de monto de la tabla.",
                                   [f.id], "¿Qué TEA aparece para depósitos a plazo de 1080 días a más?")
    assert any("uniforme" in x for x in con_reserva.reservas)
    sin_reserva = guardas.revisar("Banco A cobra S/ 1.20 y Banco B supone S/ 1.00.",
                                  [fragmentos[2].id], "¿Cuánto cuesta consultar el saldo en Kasnet?")
    assert not any("uniforme" in x for x in sin_reserva.reservas)


def test_caso_sin_reserva_se_publica_plano(guardas, fragmentos):
    """La regla de cancelación anticipada es real y no necesita reservas (caso 12 del gold)."""
    f = fragmentos[0]
    r = guardas.revisar("Hasta 30 días no se generan intereses; desde el día 31 se paga la menor tasa aplicable "
                        "al ahorro corriente.", [f.id], "¿Qué ocurre si cancelo anticipadamente un plazo fijo?")
    assert r.estado == ESTADO_PUBLICAR
    assert r.reservas == []


# --------------------------------------------------------------------- G5 · abstención
def test_abstiene_cuando_el_documento_dice_que_no_se_puede_calcular(guardas, fragmentos):
    f = fragmentos[1]
    r = guardas.revisar("La TCEA del préstamo personal es 41.00%.",
                        [f.id], "¿Cuál es la TCEA exacta del préstamo personal en Banco A?")
    assert r.estado == ESTADO_ABSTENER
    assert r.motivo == "dato_no_disponible"
    assert "TCEA" in r.respuesta


def test_abstiene_cuando_el_documento_no_tiene_la_tabla(guardas, fragmentos):
    f = fragmentos[0]
    r = guardas.revisar("La CTS de Banco A rinde 4.00%.",
                        [f.id], "¿Cuál es la tasa exacta de CTS de Banco A?")
    assert r.estado == ESTADO_ABSTENER
    assert r.motivo == "dato_no_disponible"


def test_abstiene_cuando_la_pregunta_no_toca_el_corpus(guardas, fragmentos):
    r = guardas.revisar("La agencia cierra a las 5 de la tarde.",
                        [fragmentos[0].id], "¿A qué hora cierra la agencia de Huancayo?")
    assert r.estado == ESTADO_ABSTENER
    assert r.motivo == "fuera_de_alcance"


def test_abstiene_cuando_la_respuesta_no_cita_nada(guardas, fragmentos):
    r = guardas.revisar("El ahorro en soles rinde 0.30%.",
                        [], "¿Cuál es la TEA del ahorro en soles?")
    assert r.estado == ESTADO_ABSTENER
    assert r.motivo == "sin_citas"


def test_la_tea_del_prestamo_sigue_publicandose(guardas, fragmentos):
    """Pedir la TEA no es pedir la TCEA: el documento solo declara la TCEA como no disponible."""
    f = fragmentos[1]
    r = guardas.revisar("Banco A 39.90% y Banco B 35.90%; para el costo real conviene pedir la TCEA.",
                        [f.id], "¿Cuál es la TEA del préstamo personal en ambos bancos?")
    assert r.estado == ESTADO_CON_RESERVA


# --------------------------------------------------------------------- helpers
def test_numeros_ignora_las_citas():
    assert numeros("0.30% (Art. 1, pág. 2)") == {"0.30"}
    assert numeros("S/ 40 y US$ 1.06") == {"40", "1.06"}


def test_normalizar_quita_acentos_y_unifica_guiones():
    assert normalizar("Depósitos y Ahorros — 720–1080 días") == "Depositos y Ahorros - 720-1080 dias"

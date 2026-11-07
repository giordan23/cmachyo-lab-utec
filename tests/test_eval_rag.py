import json

from app.config import DATA_DIR
from app.ingesta import extraer_fragmentos
from eval.eval_rag import evaluar


def test_set_de_oro_usa_paginas_y_evidencia_del_pdf_activo(pdf, rag):
    casos = json.loads((DATA_DIR / "rag_gold.json").read_text(encoding="utf-8"))
    paginas = {}
    for f in extraer_fragmentos(pdf):
        paginas.setdefault(f["pagina"], "")
        paginas[f["pagina"]] += " " + f["texto"]
    assert casos
    for caso in casos:
        for fuente in caso["fuentes_esperadas"]:
            assert fuente["pagina"] in paginas, caso["id"]
            texto = paginas[fuente["pagina"]].casefold()
            for frase in fuente["contiene"]:
                assert frase.casefold() in texto, f"{caso['id']}: no se encontró {frase!r} en el PDF"


def test_evaluador_retorna_metricas_reproducibles(pdf, rag):
    casos = json.loads((DATA_DIR / "rag_gold.json").read_text(encoding="utf-8"))
    resultado = evaluar(rag, casos)
    assert resultado["casos"] == len(casos)
    assert 0 <= resultado["cobertura_pagina"] <= 1
    assert 0 <= resultado["mrr_pagina"] <= 1
    assert resultado["cobertura_pagina"] >= 0.95
    assert resultado["precision_pagina"] >= 0.45
    assert len(resultado["resultados"]) == len(casos)

"""Mide recuperación documental contra data/rag_gold.json sin generar respuestas ni llamar servicios externos.

Uso: python -m eval.eval_rag [--proveedor hash|openai]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from app.config import DATA_DIR, ConfigError, Settings
from app.embeddings import HashEmbedder, crear_embedder
from app.ingesta import construir_indice
from app.rag import RAG


def evaluar(rag: RAG, casos: list[dict]) -> dict:
    resultados = []
    paginas_correctas = paginas_esperadas = paginas_recuperadas = 0
    rr = []
    for caso in casos:
        esperadas = sorted({f["pagina"] for f in caso["fuentes_esperadas"]})
        recuperadas = rag.buscar(caso["pregunta"])
        paginas = list(dict.fromkeys(f.pagina for f in recuperadas))
        coincidentes = set(esperadas) & set(paginas)
        if esperadas:
            paginas_recuperadas += len(paginas)
            paginas_correctas += len(coincidentes)
            paginas_esperadas += len(esperadas)
        primer_rango = next((i for i, p in enumerate(paginas, 1) if p in esperadas), None)
        if primer_rango is not None and esperadas:
            rr.append(1 / primer_rango)
        resultados.append({
            "id": caso["id"],
            "paginas_esperadas": esperadas,
            "paginas_recuperadas": paginas,
            "debe_abstenerse": caso["debe_abstenerse"],
            "hit": bool(coincidentes) if esperadas else None,
        })
    evaluables = len(resultados)
    casos_con_evidencia = sum(bool(c["fuentes_esperadas"]) for c in casos)
    mrr = sum(rr) / casos_con_evidencia if casos_con_evidencia else 0
    hits = [r["hit"] for r in resultados if r["hit"] is not None]
    return {
        "casos": evaluables,
        "casos_con_evidencia": casos_con_evidencia,
        "casos_sin_evidencia_en_el_set": evaluables - casos_con_evidencia,
        "precision_pagina": paginas_correctas / paginas_recuperadas if paginas_recuperadas else 0,
        "cobertura_pagina": paginas_correctas / paginas_esperadas if paginas_esperadas else 0,
        "mrr_pagina": mrr,
        "tasa_hit": sum(hits) / len(hits) if hits else 0,
        "casos_que_requieren_abstencion": sum(c["debe_abstenerse"] for c in casos),
        "nota_abstencion": "El evaluador mide recuperación, no juzga el texto generado. Una abstención puede requerir "
                           "citar evidencia que indique que el dato no está disponible.",
        "resultados": resultados,
    }


def main(argv: list[str]) -> None:
    settings = Settings()
    proveedor = argv[argv.index("--proveedor") + 1] if "--proveedor" in argv else "hash"
    if proveedor == "hash":
        embedder = HashEmbedder()
        settings.rag_min_score = 0.18
    elif proveedor == "openai":
        settings.embeddings_provider = "openai"
        settings.validar()
        embedder = crear_embedder(settings)
    else:
        raise ConfigError(f"Proveedor de embeddings desconocido: {proveedor}")
    construir_indice(settings, embedder)
    rag = RAG(settings.indice_path, embedder, settings.rag_min_score, settings.rag_top_k)
    casos = json.loads((DATA_DIR / "rag_gold.json").read_text(encoding="utf-8"))
    print(json.dumps(evaluar(rag, casos), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except ConfigError as e:
        sys.exit(f"Configuración: {e}")

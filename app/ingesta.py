"""Paso 5: PDF → texto → fragmentos (un artículo por fragmento) → embeddings → data/indice.npz

Uso:   python -m app.ingesta            (re-indexa siempre)
       python -m app.ingesta --si-falta (solo si no hay índice o es de otro embedder; lo usa Docker al arrancar)

¿Por qué un fragmento por ARTÍCULO y no cada N caracteres? Porque las reglas tienen límites naturales: si cortamos a
mitad de un artículo, el LLM recibe "no puede superar el" sin el número. Cortar por estructura conserva el sentido y
además da una cita exacta (artículo + página) para mostrar al analista.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from app.config import ConfigError, Settings
from app.embeddings import Embedder, crear_embedder
from rules import politicas

RX_ARTICULO = re.compile(r"^Artículo (\d+\.\d+) - (.+)$")
RX_RUIDO = re.compile(r"^(Página \d+|Capítulo \d+\. .*)$")   # pie de página y títulos de capítulo


def extraer_fragmentos(pdf: Path) -> list[dict]:
    """Devuelve [{id, titulo, pagina, texto}] en el orden del documento."""
    fragmentos: list[dict] = []
    actual = None
    for pagina, hoja in enumerate(PdfReader(str(pdf)).pages, 1):
        for linea in (hoja.extract_text() or "").splitlines():
            linea = linea.strip()
            if not linea or RX_RUIDO.match(linea) or linea.startswith(politicas.ENTIDAD):
                continue
            m = RX_ARTICULO.match(linea)
            if m:
                actual = {"id": m.group(1), "titulo": m.group(2).strip(), "pagina": pagina, "texto": ""}
                fragmentos.append(actual)
            elif actual is not None:                          # las líneas antes del primer artículo son la portada
                actual["texto"] = f"{actual['texto']} {linea}".strip()
    return fragmentos


def texto_para_embedding(f: dict) -> str:
    return f"Artículo {f['id']} - {f['titulo']}. {f['texto']}"


def construir_indice(settings: Settings, embedder: Embedder) -> dict:
    if not settings.pdf_path.exists():
        raise FileNotFoundError(f"No existe {settings.pdf_path}. Genera el PDF: python -m rules.generar_pdf")
    fragmentos = extraer_fragmentos(settings.pdf_path)
    if not fragmentos:
        raise ValueError(f"No se encontró ningún 'Artículo N.N - …' en {settings.pdf_path}")
    vectores = embedder.embed([texto_para_embedding(f) for f in fragmentos])
    settings.indice_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(settings.indice_path, vectores=vectores, embedder=np.array(embedder.nombre),
             fragmentos=np.array(json.dumps(fragmentos, ensure_ascii=False)))
    return {"fragmentos": len(fragmentos), "dim": int(vectores.shape[1]), "embedder": embedder.nombre,
            "paginas": sorted({f["pagina"] for f in fragmentos})}


def indice_vigente(settings: Settings, embedder: Embedder) -> bool:
    if not settings.indice_path.exists():
        return False
    with np.load(settings.indice_path) as z:
        return str(z["embedder"]) == embedder.nombre


def main(argv: list[str]) -> None:
    settings = Settings()
    if settings.embeddings_provider == "openai" and not settings.openai_api_key:
        raise ConfigError("EMBEDDINGS_PROVIDER=openai pero OPENAI_API_KEY está vacío en .env")
    embedder = crear_embedder(settings)
    if "--si-falta" in argv and indice_vigente(settings, embedder):
        print(f"Índice vigente ({embedder.nombre}): no se re-indexa.")
        return
    r = construir_indice(settings, embedder)
    print(f"Índice creado: {r['fragmentos']} artículos · {r['dim']} dimensiones · embedder {r['embedder']} · páginas {r['paginas']}")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except ConfigError as e:                       # sin clave: mensaje claro, sin traza de Python
        sys.exit(f"Configuración: {e}")

"""PDF → texto → fragmentos por página → embeddings → índice persistente.

Uso:   python -m app.ingesta            (re-indexa siempre)
       python -m app.ingesta --si-falta (reindexa si cambió el PDF o el embedder)

Los documentos de benchmarking no necesariamente tienen artículos numerados. Se dividen por página en fragmentos
solapados para conservar el contexto y permitir citas verificables.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from app.config import ConfigError, Settings
from app.embeddings import Embedder, crear_embedder
RX_RUIDO = re.compile(r"^(Página \d+|Capítulo \d+\. .*)$")   # pie de página y títulos de capítulo
MAX_FRAGMENTO = 800
SOLAPAMIENTO = 120


def extraer_fragmentos(pdf: Path) -> list[dict]:
    """Devuelve fragmentos acotados con ids estables y página de origen."""
    fragmentos: list[dict] = []
    for pagina, hoja in enumerate(PdfReader(str(pdf)).pages, 1):
        lineas = []
        for linea in (hoja.extract_text() or "").splitlines():
            linea = " ".join(linea.strip().split())
            if not linea or RX_RUIDO.match(linea):
                continue
            lineas.append(linea)
        texto = " ".join(lineas)
        inicio, numero = 0, 1
        while inicio < len(texto):
            fin = min(inicio + MAX_FRAGMENTO, len(texto))
            if fin < len(texto):
                limite = texto.rfind(" ", inicio, fin)
                if limite > inicio:
                    fin = limite
            contenido = texto[inicio:fin].strip()
            if contenido:
                fragmentos.append({
                    "id": f"p{pagina}-{numero}",
                    "titulo": f"Página {pagina}, fragmento {numero}",
                    "pagina": pagina,
                    "texto": contenido,
                })
                numero += 1
            if fin >= len(texto):
                break
            siguiente = max(fin - SOLAPAMIENTO, inicio + 1)
            espacio = texto.find(" ", siguiente, fin)
            inicio = espacio + 1 if espacio >= 0 else siguiente
    return fragmentos


def texto_para_embedding(f: dict) -> str:
    return f"{f['titulo']}. {f['texto']}"


def construir_indice(settings: Settings, embedder: Embedder) -> dict:
    if not settings.pdf_path.exists():
        raise FileNotFoundError(f"No existe el PDF configurado: {settings.pdf_path}")
    fragmentos = extraer_fragmentos(settings.pdf_path)
    if not fragmentos:
        raise ValueError(f"No se encontró contenido indexable en {settings.pdf_path}")
    vectores = embedder.embed([texto_para_embedding(f) for f in fragmentos])
    huella_pdf = hashlib.sha256(settings.pdf_path.read_bytes()).hexdigest()
    settings.indice_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(settings.indice_path, vectores=vectores, embedder=np.array(embedder.nombre),
             fragmentos=np.array(json.dumps(fragmentos, ensure_ascii=False)), pdf_hash=np.array(huella_pdf))
    return {"fragmentos": len(fragmentos), "dim": int(vectores.shape[1]), "embedder": embedder.nombre,
            "paginas": sorted({f["pagina"] for f in fragmentos})}


def indice_vigente(settings: Settings, embedder: Embedder) -> bool:
    if not settings.indice_path.exists() or not settings.pdf_path.exists():
        return False
    try:
        with np.load(settings.indice_path) as z:
            return (str(z["embedder"]) == embedder.nombre
                    and str(z["pdf_hash"]) == hashlib.sha256(settings.pdf_path.read_bytes()).hexdigest())
    except (OSError, KeyError, ValueError):
        return False


def main(argv: list[str]) -> None:
    settings = Settings()
    if settings.embeddings_provider == "openai" and not settings.openai_api_key:
        raise ConfigError("EMBEDDINGS_PROVIDER=openai pero OPENAI_API_KEY está vacío en .env")
    embedder = crear_embedder(settings)
    if "--si-falta" in argv and indice_vigente(settings, embedder):
        print(f"Índice vigente ({embedder.nombre}): no se re-indexa.")
        return
    r = construir_indice(settings, embedder)
    print(f"Índice creado: {r['fragmentos']} fragmentos · {r['dim']} dimensiones · embedder {r['embedder']} · páginas {r['paginas']}")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except ConfigError as e:                       # sin clave: mensaje claro, sin traza de Python
        sys.exit(f"Configuración: {e}")

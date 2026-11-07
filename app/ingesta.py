"""Ingesta de documentos: PDF → texto → fragmentos → embeddings → data/indice.npz

Uso:   python -m app.ingesta            (re-indexa siempre)
       python -m app.ingesta --si-falta (reindexa si cambió el PDF o el embedder)

¿Por qué un fragmento por ARTÍCULO o SECCIÓN y no cada N caracteres? Porque los documentos suelen tener límites
naturales: si cortamos a mitad de una sección, el LLM recibe texto incompleto. Cortar por estructura conserva el
sentido y además da una cita exacta (sección + página) para mostrar al usuario.

Cuando el PDF no tiene estructura de artículos numéricos, el extractor divide el texto continuo en bloques
consecutivos que no exceden el límite del modelo y conserva la página inicial de cada bloque.
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
from app.privacidad import enmascarar
from rules import politicas

RX_ARTICULO = re.compile(r"^Artículo\s+(\d+(?:\.\d+)*)\s*-\s*(.+)$")
RX_SECCION = re.compile(r"^(?P<id>\d+(?:\.\d+)*)\s*[.-]\s*(?P<titulo>[A-ZÁÉÍÓÚÜÑ].+)$")
RX_RUIDO = re.compile(r"^(Página \d+|Capítulo \d+\. .*)$")   # pie de página y títulos de capítulo
MAX_TOKENS_EMBEDDING = 8192 - 512


def _titulo_del_fragmento(linea: str) -> tuple[str, str] | None:
    """Acepta el formato de reglas del laboratorio y títulos numéricos de textos arbitrarios."""
    m = RX_ARTICULO.match(linea)
    if m:
        return m.group(1), m.group(2).strip()
    m = RX_SECCION.match(linea)
    if m:
        return m.group("id"), m.group("titulo").strip()
    return None


def _contar_tokens(texto: str, modelo: str | None = None) -> int:
    if modelo:
        try:
            import tiktoken
            try:
                encoder = tiktoken.encoding_for_model(modelo)
            except KeyError:
                encoder = tiktoken.get_encoding("cl100k_base")
            return len(encoder.encode(texto))
        except Exception:
            pass
    palabras = len(re.findall(r"\S+", texto))
    bytes_utf8 = len(texto.encode("utf-8"))
    return max(1, max(palabras, bytes_utf8 // 3))


def _particionar_texto(texto: str, modelo: str | None = None, max_tokens: int = MAX_TOKENS_EMBEDDING) -> list[str]:
    """Divide un texto largo, conservando orden y sin descartar contenido."""
    if not texto.strip():
        return []
    palabras = texto.split()
    if not palabras:
        return []
    chunks: list[str] = []
    actual = ""
    for palabra in palabras:
        propuesto = f"{actual} {palabra}".strip() if actual else palabra
        if actual and _contar_tokens(propuesto, modelo) > max_tokens:
            chunks.append(actual)
            actual = palabra
        else:
            actual = propuesto
    if actual:
        chunks.append(actual)
    return chunks


def _fragmentar_lineas(lineas: list[tuple[int, str]], base_id: str, titulo: str,
                      modelo: str | None = None, max_tokens: int = MAX_TOKENS_EMBEDDING) -> list[dict]:
    """Segmenta texto continuo en bloques consecutivos cuya longitud no supera el límite del modelo."""
    if not lineas:
        return []
    prefix = f"Artículo {base_id} - {titulo}. "
    max_texto = max(1, max_tokens - _contar_tokens(prefix, modelo))
    fragmentos: list[dict] = []
    texto_actual = ""
    pagina_actual = lineas[0][0]
    for pagina, linea in lineas:
        if not linea:
            continue
        for tramo in _particionar_texto(linea, modelo=modelo, max_tokens=max_texto):
            propuesto = f"{texto_actual} {tramo}".strip() if texto_actual else tramo
            if texto_actual and _contar_tokens(f"{prefix}{propuesto}", modelo) > max_tokens:
                fragmentos.append({
                    "id": str(base_id) if not fragmentos else f"{base_id}-{len(fragmentos) + 1}",
                    "titulo": titulo,
                    "pagina": pagina_actual,
                    "texto": texto_actual,
                })
                texto_actual = tramo
                pagina_actual = pagina
            else:
                texto_actual = propuesto
                if not fragmentos:
                    pagina_actual = pagina
    if texto_actual:
        fragmentos.append({
            "id": str(base_id) if not fragmentos else f"{base_id}-{len(fragmentos) + 1}",
            "titulo": titulo,
            "pagina": pagina_actual,
            "texto": texto_actual,
        })
    return fragmentos


def _subdividir_articulo(fragmento: dict, modelo: str | None = None, max_tokens: int = MAX_TOKENS_EMBEDDING) -> list[dict]:
    """Mantiene el artículo o sección y lo corta solo cuando supera el límite del modelo."""
    texto = " ".join(fragmento["texto"].split())
    if not texto or _contar_tokens(texto, modelo) <= max_tokens:
        return [fragmento]
    lineas = [(fragmento["pagina"], linea) for linea in texto.splitlines() if linea.strip()]
    base = str(fragmento["id"])
    subfragmentos = _fragmentar_lineas(lineas, base, fragmento["titulo"], modelo=modelo, max_tokens=max_tokens)
    if not subfragmentos:
        return [fragmento]
    if len(subfragmentos) == 1:
        return [fragmento]
    primer = subfragmentos[0]
    primer["id"] = base
    for i, sub in enumerate(subfragmentos[1:], start=2):
        sub["id"] = f"{base}-{i}"
    return subfragmentos


def extraer_fragmentos(pdf: Path, modelo: str | None = "text-embedding-3-small") -> list[dict]:
    """Devuelve [{id, titulo, pagina, texto}] en el orden del documento."""
    fragmentos: list[dict] = []
    actual = None
    texto_total: list[tuple[int, str]] = []
    for pagina, hoja in enumerate(PdfReader(str(pdf)).pages, 1):
        lineas = []
        for linea in (hoja.extract_text() or "").splitlines():
            linea = " ".join(linea.strip().split())
            if not linea or RX_RUIDO.match(linea):
                continue
            texto_total.append((pagina, linea))
            titulo = _titulo_del_fragmento(linea)
            if titulo:
                _id, _titulo = titulo
                actual = {"id": _id, "titulo": _titulo, "pagina": pagina, "texto": ""}
                fragmentos.append(actual)
            elif actual is not None:                          # las líneas antes del primer artículo son la portada
                actual["texto"] = f"{actual['texto']} {linea}".strip()

    if not fragmentos and texto_total:
        return _fragmentar_lineas(texto_total, "1", "Documento", modelo=modelo)

    fragmentos = [
        sub
        for f in fragmentos
        for sub in _subdividir_articulo(f, modelo=modelo)
    ]
    return fragmentos


def texto_para_embedding(f: dict) -> str:
    return f"{f['titulo']}. {f['texto']}"


def validar_fragmentos_para_embedding(fragmentos: list[dict], modelo: str | None = None) -> list[str]:
    """Registra y valida cada entrada final antes de enviar al embedder."""
    textos: list[str] = []
    for f in fragmentos:
        texto = texto_para_embedding(f)
        texto_enviado = enmascarar(texto)
        tokens = _contar_tokens(texto_enviado, modelo)
        print(f"[ingesta] fragmento={f['id']} página={f['pagina']} tokens={tokens} titulo={f['titulo']}", flush=True)
        if tokens > MAX_TOKENS_EMBEDDING:
            raise ValueError(
                f"Fragmento {f['id']} supera el límite del modelo ({tokens} > {MAX_TOKENS_EMBEDDING} tokens). "
                "Se requiere volver a segmentar antes de invocar a OpenAI."
            )
        textos.append(texto_enviado)
    return textos


def construir_indice(settings: Settings, embedder: Embedder) -> dict:
    if not settings.pdf_path.exists():
        raise FileNotFoundError(f"No existe {settings.pdf_path}. Genera el PDF: python -m rules.generar_pdf")
    fragmentos = extraer_fragmentos(settings.pdf_path, modelo=getattr(embedder, "modelo", None))
    if not fragmentos:
        raise ValueError(f"No se encontró contenido indexable en {settings.pdf_path}")
    textos = validar_fragmentos_para_embedding(fragmentos, getattr(embedder, "modelo", None))
    vectores = embedder.embed(textos)
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

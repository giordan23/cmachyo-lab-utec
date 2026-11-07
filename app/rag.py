"""Búsqueda en el índice de los PDF (la recuperación de RAG).

El índice se mantiene en memoria como una matriz NumPy; cada fragmento conserva la página de su documento de origen.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app import metricas
from app.embeddings import Embedder


class IndiceError(RuntimeError):
    pass


def similitudes(matriz: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Coseno de q contra cada fila (todo está normalizado, así que coseno = producto punto).
    No usamos `matriz @ q`: con numpy 2.x en Mac con chip M, la librería BLAS del sistema emite advertencias falsas
    ('divide by zero encountered in matmul') aunque el resultado sea correcto. Multiplicar y sumar evita BLAS."""
    return (matriz * q).sum(axis=1)


@dataclass
class Fragmento:
    id: str
    titulo: str
    pagina: int
    texto: str
    score: float = 0.0

    def cita(self) -> str:
        return f"{self.titulo} (pág. {self.pagina})"


class RAG:
    def __init__(self, indice_path: Path, embedder: Embedder, min_score: float, top_k: int):
        if not indice_path.exists():
            raise IndiceError(f"No existe el índice {indice_path}. Ejecuta: python -m app.ingesta")
        with np.load(indice_path) as z:
            self.vectores = z["vectores"]
            nombre = str(z["embedder"])
            self.fragmentos = [Fragmento(**f) for f in json.loads(str(z["fragmentos"]))]
        if nombre != embedder.nombre:
            raise IndiceError(f"El índice se creó con '{nombre}' y ahora usas '{embedder.nombre}'. "
                              f"Los vectores de modelos distintos no son comparables: ejecuta python -m app.ingesta")
        self.embedder = embedder
        self.min_score = min_score
        self.top_k = top_k

    def buscar(self, consulta: str, k: int | None = None) -> list[Fragmento]:
        q = self.embedder.embed([consulta])[0]
        if q.shape[0] != self.vectores.shape[1]:
            raise IndiceError(f"Dimensión de la consulta ({q.shape[0]}) distinta de la dimensión del índice ({self.vectores.shape[1]})")
        scores = similitudes(self.vectores, q)
        orden = np.argsort(-scores)[: k or self.top_k]
        if len(orden):
            metricas.RAG_SCORE.observe(float(max(scores[orden[0]], 0)))
        res = [Fragmento(**{**vars(self.fragmentos[i]), "score": round(float(scores[i]), 4)})
               for i in orden if scores[i] >= self.min_score]
        if not res:
            metricas.RAG_SIN_CONTEXTO.inc()
        return res

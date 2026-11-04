"""API del proyecto de benchmarking bancario.

    GET  /                             interfaz del chat benchmarking
    POST /api/chat                     consulta al chat de benchmarking
    POST /api/chat/nueva               nueva conversación
    GET  /api/chat/{id}?desde=N        mensajes nuevos de una conversación
    POST /api/solicitudes              consulta directa (compatibilidad)
    GET  /api/solicitudes/{id}         detalle de una respuesta
    GET  /metrics  ·  GET /health
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import metricas
from app.agente_riesgo import AgenteRiesgo
from app.config import WEB_DIR, Settings
from app.conversacion import Chat
from app.db import NoExiste, Repositorio
from app.embeddings import HashEmbedder, OpenAIEmbedder, crear_embedder
from app.llm import crear_llm
from app.modelos import MensajeChat, Solicitud
from app.rag import RAG
from app.router import crear_router

MENSAJE_DERIVACION = ("Esta consulta no corresponde al benchmarking de Banco A vs Banco B. Puedes consultar tasas, "
                      "comisiones, ITF o diferencias entre ambos bancos.")


@dataclass
class Servicios:
    settings: Settings
    router: object
    agente: AgenteRiesgo
    repo: Repositorio
    chat: Chat | None = None

    def __post_init__(self):
        if self.chat is None:
            self.chat = Chat(self.router, self.agente, self.repo, self.settings.inyeccion_umbral)


def _embedder_compatible(settings: Settings, preferido):
    if not settings.indice_path.exists():
        return preferido
    try:
        with np.load(settings.indice_path) as z:
            nombre = str(z["embedder"])
    except Exception:
        return preferido
    if nombre == preferido.nombre:
        return preferido
    if nombre.startswith("hash-"):
        return HashEmbedder(dim=int(nombre.split("-")[-1]))
    if nombre.startswith("openai-") and settings.openai_api_key:
        return OpenAIEmbedder(settings)
    return preferido


def construir_servicios(settings: Settings | None = None) -> Servicios:
    settings = settings or Settings()
    settings.validar()
    embedder = _embedder_compatible(settings, crear_embedder(settings))
    rag = RAG(settings.indice_path, embedder, settings.rag_min_score, settings.rag_top_k)
    return Servicios(settings, crear_router(settings, embedder), AgenteRiesgo(rag, crear_llm(settings)),
                     Repositorio(settings.db_path))


def vista_asesor(f: dict) -> dict:
    """Vista simplificada para el chat de benchmarking."""
    r = f["resultado"]
    base = {"id": f["id"], "estado": f["estado"], "ruta": f["ruta"],
            "router": {k: f["enrutamiento"].get(k) for k in ("proveedor", "confianza", "modelo", "bajo_umbral")}}
    if f["estado"] == "RESPONDIDA":
        return {**base, "respuesta": r["respuesta"], "citas": [x for x in r.get("reglas", []) if x["id"] in r.get("reglas_citadas", [])]}
    return {**base, "respuesta": r.get("respuesta", "Consulta atendida en el contexto de benchmarking.")}


def create_app(servicios: Servicios | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.s = servicios or construir_servicios()
        metricas.inicializar_series()
        metricas.COLA_PENDIENTE.set(app.state.s.repo.contar_pendientes())
        yield

    app = FastAPI(title="Benchmarking bancario", lifespan=lifespan)

    def svc(request: Request) -> Servicios:
        return request.app.state.s

    @app.get("/", include_in_schema=False)
    def pagina_asesor():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/api/solicitudes")
    def crear_solicitud(sol: Solicitud, s: Servicios = Depends(svc)):
        ruta = s.router.enrutar(sol.pregunta)
        if ruta.inyeccion is not None and ruta.inyeccion >= s.settings.inyeccion_umbral:
            metricas.INYECCION.inc()
        metricas.SOLICITUDES.labels(ruta.ruta).inc()
        if ruta.ruta in {"consulta_producto", "consulta_benchmark"}:
            estado, resultado = "RESPONDIDA", s.agente.responder_consulta(sol.pregunta)
        else:
            estado, resultado = "DERIVADA", {"respuesta": MENSAJE_DERIVACION}
        id_ = s.repo.crear(estado, ruta.ruta, sol.model_dump(), ruta.como_dict(), resultado)
        metricas.COLA_PENDIENTE.set(s.repo.contar_pendientes())
        return vista_asesor(s.repo.obtener(id_))

    @app.get("/api/solicitudes/{id_}")
    def ver_solicitud(id_: int, s: Servicios = Depends(svc)):
        try:
            return vista_asesor(s.repo.obtener(id_))
        except NoExiste:
            raise HTTPException(404, f"No existe la solicitud {id_}") from None

    @app.post("/api/chat")
    def chat_enviar(m: MensajeChat, s: Servicios = Depends(svc)):
        try:
            return s.chat.enviar(m.conversacion_id, m.mensaje)
        except NoExiste:
            raise HTTPException(404, f"No existe la conversación {m.conversacion_id}") from None

    @app.post("/api/chat/nueva")
    def chat_nueva(s: Servicios = Depends(svc)):
        return {"conversacion_id": s.repo.crear_conversacion()}

    @app.get("/api/chat/{id_}")
    def chat_leer(id_: int, desde: int = Query(0, ge=0), s: Servicios = Depends(svc)):
        if not s.repo.existe_conversacion(id_):
            raise HTTPException(404, f"No existe la conversación {id_}")
        return {"conversacion_id": id_, "mensajes": s.repo.mensajes(id_, desde)}

    return app


app = create_app()

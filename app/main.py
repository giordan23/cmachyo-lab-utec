"""Paso 9: la API y las dos páginas web.

    GET  /asesor                        chat del asesor de ventas (web/asesor.html)
    GET  /analista                      bandeja del analista de riesgos (web/analista.html)
    POST /api/chat                      mensaje del asesor → Jev → RAG conversacional / pedir datos / derivar
    POST /api/chat/nueva                conversación vacía (para escalar sin haber escrito antes)
    GET  /api/chat/{id}?desde=N         mensajes nuevos de una conversación (el chat sondea aquí)
    POST /api/chat/{id}/evaluacion      datos del cliente desde el chat → agente de riesgo → cola del analista
    POST /api/solicitudes               formulario directo (sin chat): lo usan las pruebas y loadgen
    GET  /api/solicitudes/{id}          lo que VE el asesor (sin la propuesta preliminar del agente)
    GET  /api/cola                      pendientes, con todo el detalle            (requiere X-Analista-Token)
    POST /api/solicitudes/{id}/decision decisión humana final                      (requiere X-Analista-Token)
    GET  /metrics  ·  GET /health

Segregación de funciones (Art. 1.3): quien registra la venta no puede aprobarla. El token separa los dos roles.
No es autenticación de producción (no hay usuarios ni sesiones): es el mínimo para que el control exista en el lab.
"""
from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import metricas
from app.agente_riesgo import AgenteRiesgo
from app.config import WEB_DIR, Settings
from app.conversacion import Chat
from app.db import NoExiste, Repositorio, YaDecidida
from app.embeddings import HashEmbedder, OpenAIEmbedder, crear_embedder
from app.llm import crear_llm
from app.modelos import Decision, MensajeChat, Solicitud
from app.rag import RAG
from app.router import crear_router

MENSAJE_DERIVACION = ("Esta consulta no corresponde a productos de crédito ni a evaluación de riesgo. "
                      "Derívala al área correspondiente (agencia, RR. HH., soporte o atención al cliente).")
COINCIDE = {"APROBADA": "APROBAR", "RECHAZADA": "RECHAZAR"}


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
    """Lo que ve el asesor. Mientras está PENDIENTE no ve la propuesta del agente: evita que adelante al cliente
    una decisión que todavía no tomó un humano."""
    r = f["resultado"]
    base = {"id": f["id"], "estado": f["estado"], "ruta": f["ruta"],
            "router": {k: f["enrutamiento"].get(k) for k in ("proveedor", "confianza", "modelo", "bajo_umbral")}}
    if f["estado"] == "RESPONDIDA":
        return {**base, "respuesta": r["respuesta"], "citas": [x for x in r["reglas"] if x["id"] in r["reglas_citadas"]]}
    if f["estado"] == "DERIVADA":
        return {**base, "respuesta": r["respuesta"]}
    if f["estado"] == "PENDIENTE_ANALISTA":
        return {**base, "respuesta": "Solicitud en revisión por un analista de riesgos. Esta página se actualiza sola."}
    return {**base, "decision": f["decision"], "analista": f["analista"], "comentario": f["comentario"],
            "respuesta": r["justificacion"], "condiciones": r["condiciones"], "cuota": r["calculos"]["cuota"],
            "citas": [x for x in r["reglas"] if x["id"] in r["reglas_citadas"]]}


def vista_conversacion(mensajes: list[dict], ultimos: int = 12) -> list[dict]:
    """Para el analista: la marca de sospecha la guarda la RESPUESTA (el router corre después de guardar el mensaje);
    aquí se traslada al mensaje del asesor que la causó."""
    vista = [{"rol": m["rol"], "texto": m["texto"], "inyeccion_sospechosa": False} for m in mensajes]
    for i, m in enumerate(mensajes):
        if m["meta"].get("inyeccion_sospechosa") and i > 0 and vista[i - 1]["rol"] == "asesor":
            vista[i - 1]["inyeccion_sospechosa"] = True
    return vista[-ultimos:]


def create_app(servicios: Servicios | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.s = servicios or construir_servicios()
        metricas.inicializar_series()
        metricas.COLA_PENDIENTE.set(app.state.s.repo.contar_pendientes())
        yield

    app = FastAPI(title="Lab 06 · Asesor -> Jev -> Agente de riesgo -> Analista", lifespan=lifespan)

    def svc(request: Request) -> Servicios:
        return request.app.state.s

    def solo_analista(request: Request, x_analista_token: str = Header(default="")) -> None:
        esperado = svc(request).settings.analista_token
        if not esperado:
            raise HTTPException(503, "ANALISTA_TOKEN no está configurado en .env: la aprobación está deshabilitada")
        if not hmac.compare_digest(x_analista_token.encode(), esperado.encode()):   # comparación en tiempo constante
            raise HTTPException(401, "Token de analista inválido")

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
        if ruta.ruta == "evaluacion_credito":
            estado, resultado = "RESPONDIDA", {"respuesta": "Este proyecto está orientado a benchmarking y comparación de Banco A vs Banco B. No incluye evaluación de riesgo ni escalado a analista."}
        elif ruta.ruta in {"consulta_producto", "consulta_benchmark"}:
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

    @app.post("/api/chat/{id_}/evaluacion")
    def chat_escalar(id_: int, sol: Solicitud, origen: Literal["router", "boton"] = "boton", s: Servicios = Depends(svc)):
        try:
            if not s.repo.existe_conversacion(id_):
                raise NoExiste
        except NoExiste:
            raise HTTPException(404, f"No existe la conversación {id_}") from None
        return {"conversacion_id": id_, "solicitud_id": None,
                "mensajes": [{"rol": "sistema", "texto": "Este proyecto está orientado a benchmarking. No se usa escalado a analista de riesgo."}]}

    @app.get("/api/cola", dependencies=[Depends(solo_analista)])
    def cola(s: Servicios = Depends(svc)):
        filas = s.repo.pendientes()
        for f in filas:                       # el analista ve la conversación que originó la solicitud
            f["conversacion"] = vista_conversacion(s.repo.mensajes(f["conversacion_id"])) if f["conversacion_id"] else []
            f["inyeccion_sospechosa"] = (any(m["inyeccion_sospechosa"] for m in f["conversacion"])
                                         or bool((f["enrutamiento"].get("inyeccion") or 0) >= s.settings.inyeccion_umbral))
        return filas

    @app.post("/api/solicitudes/{id_}/decision", dependencies=[Depends(solo_analista)])
    def decidir(id_: int, d: Decision, s: Servicios = Depends(svc)):
        try:
            fila = s.repo.obtener(id_)
        except NoExiste:
            raise HTTPException(404, f"No existe la solicitud {id_}") from None
        if fila["estado"] != "PENDIENTE_ANALISTA":
            raise HTTPException(409, f"La solicitud {id_} ya está en estado {fila['estado']}")
        dictamen = fila["resultado"]["dictamen"]
        coincide = COINCIDE[d.decision] == dictamen
        if not coincide and len(d.comentario.strip()) < 10:
            raise HTTPException(422, f"El agente propuso {dictamen}: para decidir {d.decision} escribe un comentario "
                                     f"que lo justifique (mínimo 10 caracteres, Art. 1.3)")
        try:
            fila = s.repo.decidir(id_, d.decision, d.analista.strip(), d.comentario.strip())
        except YaDecidida as e:                      # otro analista decidió entre nuestra lectura y la escritura
            raise HTTPException(409, str(e)) from None
        s.chat.notificar_decision(fila)
        metricas.DECISION_ANALISTA.labels(d.decision, "si" if coincide else "no").inc()
        metricas.TIEMPO_APROBACION.observe(fila["decidida_en"] - fila["creada_en"])
        metricas.COLA_PENDIENTE.set(s.repo.contar_pendientes())
        return vista_asesor(fila)

    return app


app = create_app()

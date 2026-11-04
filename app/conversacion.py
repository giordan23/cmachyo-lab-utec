"""Paso 10: la conversación del asesor. Un chat sobre las reglas que, cuando hace falta, escala al analista de riesgo.

    asesor escribe ──► Jev (con los mensajes anteriores como contexto)
                         ├─ consulta_producto  → RAG conversacional: reformular → buscar en el PDF → responder con citas
                         ├─ evaluacion_credito → "necesito los datos del cliente" + formulario en el chat
                         └─ fuera_de_alcance   → derivación
    formulario enviado ─► agente de riesgo (cálculos + guardas + RAG + LLM) → PENDIENTE_ANALISTA
    analista decide ────► la decisión vuelve al chat como mensaje del sistema

¿Por qué los datos del cliente van en un formulario y no "conversados"? Porque las guardas dependen de números
EXACTOS (ingreso, deuda, monto, plazo). Si un LLM los extrajera del texto libre, un error de extracción ("3,500" leído
como 3.5) cambiaría el dictamen sin que ninguna guarda lo note. El chat sirve para entender; el formulario, para decidir.

El asesor también puede escalar SIN que el router lo decida (botón "Escalar al analista"): el router ayuda, no manda.
"""
from __future__ import annotations

from app import metricas
from app.db import NoExiste
from app.modelos import Solicitud
from rules import politicas

HISTORIAL_MAX = 6          # mensajes previos que recibe el LLM: más contexto = más costo y más ruido
BENCHMARK_SOLO = ("Este proyecto está orientado a benchmarking y comparación entre Banco A y Banco B. "
                  "No incluye evaluación de riesgo de clientes ni escalado a un analista de riesgo.")
DERIVAR = ("Esta consulta no corresponde a benchmarking ni a comparación entre Banco A y Banco B. "
           "Puedes preguntar sobre tasas, comisiones, ITF, depósitos o diferencias entre ambos bancos.")


class Chat:
    def __init__(self, router, agente, repo, umbral_inyeccion: float = 0.8):
        self.router, self.agente, self.repo = router, agente, repo
        self.umbral_inyeccion = umbral_inyeccion

    def _validar(self, conversacion_id: int) -> None:
        if not self.repo.existe_conversacion(conversacion_id):
            raise NoExiste

    def historial(self, conversacion_id: int) -> list[dict]:
        """Los últimos turnos, en el formato que recibe el LLM: [{rol, texto}]."""
        return [{"rol": m["rol"], "texto": m["texto"][:600]}
                for m in self.repo.mensajes(conversacion_id)[-HISTORIAL_MAX:]]

    def enviar(self, conversacion_id: int | None, texto: str) -> dict:
        if conversacion_id is None:
            conversacion_id = self.repo.crear_conversacion()
        else:
            self._validar(conversacion_id)
        previos = self.historial(conversacion_id)
        nuevo = self.repo.agregar_mensaje(conversacion_id, "asesor", texto)

        ruta = self.router.enrutar(texto, [m["texto"] for m in previos if m["rol"] == "asesor"])
        metricas.CHAT_MENSAJES.labels(ruta.ruta).inc()
        sospechoso = ruta.inyeccion is not None and ruta.inyeccion >= self.umbral_inyeccion
        if sospechoso:
            metricas.INYECCION.inc()
        meta = {"ruta": ruta.ruta, "router": ruta.como_dict(), "inyeccion_sospechosa": sospechoso, "accion": None}

        if ruta.ruta in {"consulta_producto", "consulta_benchmark"}:
            r = self.agente.responder_consulta(texto, previos)
            meta.update(consulta=r["consulta"], llm_error=r["llm_error"],
                        citas=[x for x in r["reglas"] if x["id"] in r["reglas_citadas"]])
            respuesta = r["respuesta"]
        elif ruta.ruta == "evaluacion_credito":
            respuesta = BENCHMARK_SOLO
        else:
            respuesta = DERIVAR
        return {"conversacion_id": conversacion_id,
                "mensajes": [nuevo, self.repo.agregar_mensaje(conversacion_id, "asistente", respuesta, meta)]}

    def escalar(self, conversacion_id: int, s: Solicitud, origen: str) -> dict:
        """El asesor envía los datos del cliente desde el chat → agente de riesgo → cola del analista."""
        self._validar(conversacion_id)
        producto = politicas.PRODUCTOS[s.producto]["nombre"]
        nuevo = self.repo.agregar_mensaje(conversacion_id, "asesor",
                                          f"Datos del cliente enviados para evaluación: {producto}, S/ {s.monto:,.0f} a {s.plazo_meses} meses.",
                                          {"formulario": True})
        mensajes = self.repo.mensajes(conversacion_id)
        sospecha = [m["meta"]["router"]["inyeccion"] for m in mensajes
                    if m["rol"] == "asistente" and (m["meta"].get("router") or {}).get("inyeccion") is not None]
        resultado = self.agente.evaluar(s)
        resultado["inyeccion_max_conversacion"] = max(sospecha, default=None)
        enrutamiento = {"ruta": "evaluacion_credito", "proveedor": "chat", "modelo": f"escalado por {origen}",
                        "confianza": None, "bajo_umbral": False, "respaldo": ""}
        id_ = self.repo.crear("PENDIENTE_ANALISTA", "evaluacion_credito", s.model_dump(), enrutamiento, resultado,
                              conversacion_id)
        metricas.SOLICITUDES.labels("evaluacion_credito").inc()
        metricas.CHAT_ESCALADAS.labels(origen).inc()
        metricas.COLA_PENDIENTE.set(self.repo.contar_pendientes())
        aviso = self.repo.agregar_mensaje(conversacion_id, "sistema",
                                          f"Solicitud #{id_} enviada al analista de riesgos. Te aviso aquí cuando decida.",
                                          {"solicitud_id": id_, "estado": "PENDIENTE_ANALISTA"})
        return {"conversacion_id": conversacion_id, "solicitud_id": id_, "mensajes": [nuevo, aviso]}

    def notificar_decision(self, fila: dict) -> None:
        """La decisión humana vuelve a la conversación de origen (si la solicitud vino del chat)."""
        if not fila.get("conversacion_id"):
            return
        r = fila["resultado"]
        texto = f"Solicitud #{fila['id']}: {fila['decision']} por {fila['analista']}."
        if fila["comentario"]:
            texto += f" Comentario: {fila['comentario']}"
        self.repo.agregar_mensaje(fila["conversacion_id"], "sistema", texto, {
            "solicitud_id": fila["id"], "decision": fila["decision"], "justificacion": r["justificacion"],
            "condiciones": r["condiciones"], "cuota": r["calculos"]["cuota"],
            "citas": [x for x in r["reglas"] if x["id"] in r["reglas_citadas"]]})

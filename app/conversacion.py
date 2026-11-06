"""Chat del asistente documental sobre ahorros y condiciones de Caja Huancayo.

El usuario consulta documentos indexados y el sistema responde con contexto y citas.
"""
from __future__ import annotations

from app import metricas
from app.db import NoExiste

HISTORIAL_MAX = 6          # mensajes previos que recibe el LLM: más contexto = más costo y más ruido
FUERA_DE_ALCANCE_CREDITICIO = ("Este prototipo está orientado a consultar productos y condiciones de ahorro de Caja Huancayo "
                               "a partir de los documentos proporcionados. No brinda evaluación de crédito ni aprobaciones.")
DERIVAR = ("Esta consulta no corresponde al alcance del prototipo de ahorro de Caja Huancayo. "
           "Puedes preguntar por cuentas, requisitos, condiciones, plazos, tasas o productos documentados.")


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
        nuevo = self.repo.agregar_mensaje(conversacion_id, "usuario", texto)

        ruta = self.router.enrutar(texto, [m["texto"] for m in previos if m["rol"] == "usuario"])
        metricas.CHAT_MENSAJES.labels(ruta.ruta).inc()
        sospechoso = ruta.inyeccion is not None and ruta.inyeccion >= self.umbral_inyeccion
        if sospechoso:
            metricas.INYECCION.inc()
        meta = {"ruta": ruta.ruta, "router": ruta.como_dict(), "inyeccion_sospechosa": sospechoso, "accion": None}

        if ruta.ruta in {"consulta_ahorros", "consulta_producto", "consulta_benchmark"}:
            r = self.agente.responder_consulta(texto, previos)
            meta.update(consulta=r["consulta"], llm_error=r["llm_error"], estado=r.get("estado"),
                        motivo=r.get("motivo"), reservas=r.get("reservas", []),
                        citas=r.get("citas") or [x for x in r["reglas"] if x["id"] in r["reglas_citadas"]])
            respuesta = r["respuesta"]
        else:
            respuesta = DERIVAR
        return {"conversacion_id": conversacion_id,
                "mensajes": [nuevo, self.repo.agregar_mensaje(conversacion_id, "asistente", respuesta, meta)]}

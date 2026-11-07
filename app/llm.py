"""Paso 8: el LLM. Una interfaz (`generar`) y dos implementaciones.

- OpenAILLM: OpenAI real con la Responses API y SALIDA ESTRUCTURADA (JSON Schema estricto). El modelo no puede
  devolver texto libre ni inventar campos: o devuelve un JSON que cumple el esquema, o falla y lo tratamos.
- MockLLM: reglas fijas, sin red. NO es inteligencia; existe para que pytest corra sin clave ni costo.

`store=False`: desactiva el almacenamiento de las respuestas en el proveedor.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from app import metricas
from app.privacidad import enmascarar


class LLMError(RuntimeError):
    """Fallo del proveedor: red, clave, límite, respuesta incompleta o JSON inválido."""


@dataclass
class RespuestaLLM:
    datos: dict
    modelo: str
    input_tokens: int = 0
    output_tokens: int = 0


ESQUEMA_RESPUESTA = {
    "type": "object", "additionalProperties": False,
    "required": ["respuesta", "fuentes_citadas"],
    "properties": {"respuesta": {"type": "string"}, "fuentes_citadas": {"type": "array", "items": {"type": "string"}}},
}


ESQUEMA_REFORMULA = {
    "type": "object", "additionalProperties": False, "required": ["consulta"],
    "properties": {"consulta": {"type": "string"}},
}


class OpenAILLM:
    nombre = "openai"

    def __init__(self, settings, client=None):
        from openai import OpenAI

        self.modelo = settings.openai_model
        self.esfuerzo = settings.openai_reasoning_effort
        self.max_tokens = settings.openai_max_output_tokens
        self.precio_in, self.precio_out = settings.precio_openai_input, settings.precio_openai_output
        self.client = client or OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout_s, max_retries=2)

    def generar(self, instrucciones: str, contexto: dict, esquema: dict, nombre: str) -> RespuestaLLM:
        import openai

        kwargs = {"model": self.modelo, "instructions": instrucciones, "store": False,
                  "input": enmascarar(json.dumps(contexto, ensure_ascii=False, indent=1)),   # DNI, teléfonos… nunca salen
                  "text": {"format": {"type": "json_schema", "name": nombre, "schema": esquema, "strict": True}},
                  "max_output_tokens": self.max_tokens}
        if self.esfuerzo:
            kwargs["reasoning"] = {"effort": self.esfuerzo}
        t0 = time.perf_counter()
        try:
            r = self.client.responses.create(**kwargs)
        except openai.APIError as e:
            metricas.API_ERRORES.labels("openai_llm", type(e).__name__).inc()
            raise LLMError(f"OpenAI falló: {type(e).__name__}: {e}") from e
        finally:
            metricas.API_LATENCIA.labels("openai_llm").observe(time.perf_counter() - t0)

        uso_in = r.usage.input_tokens if r.usage else 0
        uso_out = r.usage.output_tokens if r.usage else 0
        metricas.TOKENS.labels("openai_llm", "input").inc(uso_in)
        metricas.TOKENS.labels("openai_llm", "output").inc(uso_out)
        metricas.COSTO.labels("openai_llm").inc((uso_in * self.precio_in + uso_out * self.precio_out) / 1e6)

        if r.status != "completed":          # p. ej. max_output_tokens agotado por el razonamiento
            metricas.API_ERRORES.labels("openai_llm", "incompleta").inc()
            raise LLMError(f"respuesta {r.status}: {r.incomplete_details}")
        try:
            datos = json.loads(r.output_text)  # vacío si el modelo se negó (refusal)
        except json.JSONDecodeError as e:
            metricas.API_ERRORES.labels("openai_llm", "json_invalido").inc()
            raise LLMError(f"JSON inválido: {r.output_text[:200]!r}") from e
        return RespuestaLLM(datos, r.model, uso_in, uso_out)


class MockLLM:
    """Simulador determinista que responde con el primer fragmento documental recibido."""
    nombre = "mock"
    modelo = "mock"

    def generar(self, instrucciones: str, contexto: dict, esquema: dict, nombre: str) -> RespuestaLLM:
        fuentes = contexto.get("fuentes", [])
        if nombre == "reformulacion":
            # regla fija: un mensaje corto con historial se completa con la pregunta anterior del asesor
            actual = contexto["mensaje_actual"]
            previas = [m["texto"] for m in contexto.get("historial", []) if m["rol"] == "asesor"]
            consulta = f"{previas[-1]} {actual}" if previas and len(actual.split()) < 6 else actual
            return RespuestaLLM({"consulta": consulta}, self.modelo)
        if not fuentes:
            raise LLMError("No se puede generar una respuesta sin fuentes")
        top = fuentes[0]
        pagina = top.get("pagina", "?")
        return RespuestaLLM({"respuesta": f"[simulado] Según la página {pagina}: {top['texto']}",
                             "fuentes_citadas": [top["id"]]}, self.modelo)


def crear_llm(settings):
    return OpenAILLM(settings) if settings.llm_provider == "openai" else MockLLM()

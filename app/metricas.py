"""Métricas documentales expuestas en GET /metrics para Prometheus y Grafana.

Convención de nombres de Prometheus: `<sistema>_<qué>_<unidad>` y `_total` para contadores.
Todas llevan el prefijo `lab06_` para no mezclarse con otras aplicaciones en el mismo Prometheus.

Cuidado con la CARDINALIDAD: cada combinación de etiquetas es una serie temporal nueva. Por eso las etiquetas son
valores de una lista cerrada (ruta, proveedor…) y NUNCA datos personales ni texto libre.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

RUTAS = ["consulta_benchmark", "fuera_de_alcance"]
PROVEEDORES_API = ["jev", "openai_llm", "openai_emb"]

PROB = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

ROUTER_CONFIANZA = Histogram("lab06_router_confianza", "Confianza de la ruta elegida (Jev: probabilidad; embeddings: coseno)",
                             ["proveedor"], buckets=PROB)
ROUTER_RESPALDO = Counter("lab06_router_respaldo_total", "Veces que Jev falló y se usó el router de respaldo", ["motivo"])
RAG_SCORE = Histogram("lab06_rag_score", "Similitud del mejor fragmento recuperado del PDF", buckets=PROB)
RAG_SIN_CONTEXTO = Counter("lab06_rag_sin_contexto_total", "Consultas sin ningún fragmento sobre el umbral")
API_LATENCIA = Histogram("lab06_api_latencia_segundos", "Latencia de cada llamada a un proveedor externo", ["proveedor"],
                         buckets=[0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60])
API_ERRORES = Counter("lab06_api_errores_total", "Errores de proveedores externos", ["proveedor", "tipo"])
TOKENS = Counter("lab06_tokens_total", "Tokens consumidos", ["proveedor", "tipo"])
COSTO = Counter("lab06_costo_estimado_usd_total", "Costo estimado en USD (precios de .env)", ["proveedor"])
CITAS_INVALIDAS = Counter("lab06_citas_invalidas_total", "Fuentes citadas por el LLM que no estaban en el contexto")
CHAT_MENSAJES = Counter("lab06_chat_mensajes_total", "Mensajes de usuario en el chat, por ruta", ["ruta"])
CHAT_REFORMULADAS = Counter("lab06_chat_reformuladas_total", "Preguntas de seguimiento reescritas como consulta autónoma")
INYECCION = Counter("lab06_inyeccion_sospechosa_total", "Mensajes que Jev marcó como intento de manipular al sistema")
PII_ENMASCARADO = Counter("lab06_pii_enmascarado_total", "Datos personales enmascarados antes de salir a un tercero", ["tipo"])
COLA_PENDIENTE = Gauge("lab06_cola_pendiente", "Solicitudes esperando al analista")
GUARDA_RESPUESTA = Counter("lab06_guarda_respuesta_total", "Estado de la respuesta tras las guardas deterministas",
                            ["estado"])
GUARDA_RESERVAS = Counter("lab06_guarda_reservas_total", "Reservas aplicadas a la respuesta del LLM", ["tipo"])
GUARDA_CIFRAS = Counter("lab06_guarda_cifras_sin_grounding_total", "Cifras de la respuesta que no existen en el corpus")
CITA_PAGINA_CORREGIDA = Counter("lab06_cita_pagina_corregida_total",
                                 "Citas cuya página se corrigió porque el texto citado vive en otra página")
TIEMPO_APROBACION = Histogram("lab06_tiempo_aprobacion_segundos", "Desde que llega la solicitud hasta la decisión humana",
                              buckets=[5, 15, 30, 60, 120, 300, 600, 1800, 3600])


def inicializar_series() -> None:
    """Un contador que nunca se incrementó NO aparece en /metrics, y Grafana muestra 'No data' en vez de 0.
    Crear las series conocidas en cero hace que los paneles muestren 0 desde el arranque."""
    for r in RUTAS:
        CHAT_MENSAJES.labels(r)
    for p in ("jev", "embeddings", "hash"):
        ROUTER_CONFIANZA.labels(p)
    for p in PROVEEDORES_API:
        API_LATENCIA.labels(p)
        COSTO.labels(p)
        for t in ("input", "output"):
            TOKENS.labels(p, t)
    for t in ("email", "ruc", "telefono", "dni"):
        PII_ENMASCARADO.labels(t)
    for m in ("auth", "validacion", "limite", "servidor", "red"):
        ROUTER_RESPALDO.labels(m)
    for e in ("PUBLICAR", "PUBLICAR_CON_RESERVA", "ABSTENER"):
        GUARDA_RESPUESTA.labels(e)
    for t in ("ficticio_total", "dato_ficticio", "documento_incompleto", "fila_uniforme", "otra"):
        GUARDA_RESERVAS.labels(t)

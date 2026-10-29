# Lab 06 · Asesor (chat RAG) → Jev → RAG o agente de riesgo → Aprobación humana

UTEC Posgrado × Caja Huancayo · IA Generativa para Desarrolladores (Módulos 3, 4 y 5).

Un asesor de ventas **conversa** con el sistema (chat RAG sobre las reglas). **Jev** (TypeSafe) enruta cada
mensaje; si un cliente necesita evaluación, el chat pide sus datos y un **agente especialista en riesgo** calcula, aplica guardas deterministas, recupera las reglas de un
**PDF de políticas** (RAG con embeddings de OpenAI) y propone un dictamen con **OpenAI**; un **analista humano**
aprueba o rechaza en una web HTML simple; el asesor recibe la respuesta. Todo medido con **Prometheus** y **Grafana**.

> Datos y reglas **ficticios** (entidad inventada "Caja Demo Andina"). No es la política de ninguna entidad real.

## Abrir la guía

`laboratorio_riesgo_aprobacion_utec.html` — guía paso a paso (15 pasos, teoría, código, salidas reales,
macOS/Linux y Windows PowerShell). Ábrela con doble clic en cualquier navegador.

## Inicio rápido (si ya conoces el laboratorio)

```bash
python -m venv .venv                      # Windows: py -3 -m venv .venv
source .venv/bin/activate                 # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
cp .env.example .env                      # Windows: Copy-Item .env.example .env   → pon tus claves
python -m pytest                          # sin red ni claves
python -m scripts.verificar_claves        # prueba de humo de Jev y OpenAI reales
python -m scripts.normalizar_env          # .env en UTF-8 sin BOM (Docker no lee UTF-16)
docker compose up -d --build
```

| URL | Qué es |
|---|---|
| http://localhost:8000/asesor | Chat del asesor (RAG conversacional + escalamiento) |
| http://localhost:8000/analista | Bandeja del analista (pide `ANALISTA_TOKEN`) |
| http://localhost:9090/targets | Prometheus (espera `lab06` en UP antes de generar carga) |
| http://localhost:3000/d/lab06 | Tablero de Grafana |

## Estructura

```
app/        config, metricas, privacidad, embeddings, rag, ingesta, router (Jev), modelos, herramientas, guardas, llm,
            agente_riesgo, db, conversacion (chat), main
rules/      politicas.py (fuente única de las reglas) y generar_pdf.py
data/       rutas.json, router_gold.json, politicas_riesgo.pdf (índice y SQLite se generan)
web/        asesor.html, analista.html
tests/      132 pruebas, ninguna llama a una API real
eval/       eval_router.py (exactitud del router con set de oro)
scripts/    verificar_claves.py, normalizar_env.py
loadgen/    generar_carga.py
docs/       ANALISIS_PROFUNDO.md, ANALISIS_WINDOWS.md, specs/ y plans/
_guide/     fuentes de la guía: parts/*.html → capture.py → build.py
```

## Estado de verificación

| Pieza | Estado |
|---|---|
| Pruebas (132) | Pasan en Python 3.10 (macOS); 39/39 mutaciones detectadas. Simulación de alumno siguiendo la guía en orden, en carpeta con espacios y "ñ": todos los pasos OK |
| Docker + Prometheus + Grafana | Ejecutado de verdad (modo sin claves): app healthy, target UP, 23 consultas del tablero sin error |
| Chat → analista → chat | Recorrido real en el navegador (modo sin claves) |
| Jev (TypeSafe) | **Preparado, no ejecutado con clave real.** Formato tomado de documentación de terceros; probado contra servidor simulado |
| OpenAI | **Preparado, no ejecutado con clave real.** Probado con el SDK oficial contra respuestas HTTP simuladas |
| Windows | Comandos revisados por analizador, trampas simuladas (cp1252, UTF-16, BOM, CRLF). **No ejecutado en Windows real** |

## Regenerar la guía

```bash
python _guide/capture.py             # pruebas y scripts (modo sin claves)
python _guide/capture.py --docker    # con el stack levantado: compose ps y carga
python _guide/build.py               # falla si un comando de Windows usa sintaxis de bash
```

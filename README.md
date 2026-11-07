# Benchmarking bancario con RAG y chat conversacional

Proyecto académico orientado a comparar condiciones de dos bancos a partir de un documento PDF de benchmarking. El sistema lee el PDF, segmenta el contenido en unidades lógicas, crea un índice vectorial persistente y responde preguntas naturales con citas a los fragmentos recuperados.

El flujo actual está adaptado a la comparación entre Banco A y Banco B, y no incluye evaluación crediticia ni escalado a analista, porque el objetivo del proyecto es analizar el documento de benchmarking y comparar tasas, comisiones, ITF y condiciones operativas.

## Objetivo del proyecto

Responder consultas del tipo:

- ¿Qué diferencia hay entre Banco A y Banco B en ahorro en soles?
- Compara las tasas de tarjeta de crédito entre ambos bancos.
- ¿Cuál banco tiene mejores condiciones para un depósito a plazo?
- Analiza la diferencia de comisiones e ITF entre Banco A y Banco B.

## Arquitectura

El proyecto sigue el patrón básico del laboratorio:

1. Ingesta del PDF
   - app/ingesta.py
   - extrae el texto completo del PDF con pypdf
   - segmenta por sección o artículo
   - guarda metadatos como id, título, página y texto
   - serializa el índice en data/indice.npz

2. Embeddings y base vectorial
   - app/embeddings.py
   - usa un embedder configurado (openai o hash para modo local sin clave)
   - normaliza vectores y persiste el índice para uso posterior

3. Recuperación RAG
   - app/rag.py
   - recupera los fragmentos más relevantes usando top-k y umbral
   - presenta citas con artículo, título y página
   - responde con “no hay contexto suficiente” cuando no hay evidencia

4. Chat conversacional
   - app/conversacion.py
   - mantiene historial acotado y reformula preguntas de seguimiento
   - la interfaz web consulta el backend vía /api/chat

5. Enrutamiento y agente
   - app/router.py
   - decide si la consulta es de benchmarking o si no corresponde al documento
   - app/agente_riesgo.py genera la respuesta con contexto y citas

6. Observabilidad y pruebas
   - app/metricas.py
   - pruebas en tests/
   - dashboard y métricas con Prometheus/Grafana

## Estructura del repositorio

```text
app/            lógica principal: ingest, embeddings, RAG, router, chat, LLM, API
rules/          política y generación del PDF (aunque el proyecto actual se enfoca en benchmarking)
data/          Benchmarking.pdf, rutas.json, índice generado, SQLite si aplica
web/            interfaz del chat (HTML/JS)
tests/          pruebas automáticas de ingest, RAG y validación de flujo
scripts/        scripts de normalización y verificación de entorno
Dockerfile      contenedor de la aplicación
docker-compose.yml
requirements*.txt
README.md
```

## Requisitos

- Python 3.10+
- pip
- opcionalmente: OpenAI API key para embeddings/LLM reales
- para modo local y sin costo: usar EMBEDDINGS_PROVIDER=hash y LLM_PROVIDER=mock

## Configuración

Copia el ejemplo de entorno:

```bash
copy .env.example .env
```

Variables relevantes:

```env
EMBEDDINGS_PROVIDER=hash
LLM_PROVIDER=mock
ROUTER_PROVIDER=hash
ANALISTA_TOKEN=token-demo
```

Si quieres usar OpenAI real:

```env
EMBEDDINGS_PROVIDER=openai
LLM_PROVIDER=openai
OPENAI_API_KEY=tu_clave
```

Nota: con OpenAI real, el uso depende de cuota disponible. Para una demo estable sin costo, usa hash + mock.

## Inicio rápido

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Configura `.env` a partir de `.env.example`. `ROUTER_PROVIDER=hash`, `EMBEDDINGS_PROVIDER=hash` y `LLM_PROVIDER=mock` permiten una demo local sin claves externas. Para respuestas de un modelo real, configura las claves y proveedores correspondientes.

```powershell
python -m app.ingesta
python -m app
```

Luego abre:

- http://127.0.0.1:8000/

## Demo de uso

Ejemplos de consultas válidas para el benchmarking:

```text
¿Qué diferencia hay entre Banco A y Banco B en ahorro en soles?
Compara las tasas de tarjeta de crédito entre Banco A y Banco B.
¿Cuál banco tiene mejores condiciones para un depósito a plazo?
Analiza la diferencia de comisiones e ITF entre ambos bancos.
¿Que tan favorable es Banco B frente a Banco A en ahorro programado?
```

## Validación por rúbrica

| Criterio | Estado | Evidencia |
|---|---|---|
| C1 Ingesta del PDF | Cumplido | app/ingesta.py, tests/test_rag.py |
| C2 Base vectorial y embeddings | Cumplido | app/embeddings.py, app/ingesta.py, app/rag.py |
| C3 Recuperación RAG y citas | Cumplido | app/rag.py, app/agente_riesgo.py |
| C4 Chat conversacional | Cumplido | app/conversacion.py, web/index.html |
| C5 Comunicación con el agente | Cumplido | app/router.py, app/main.py |
| C6 Seguridad, privacidad y guardas | Cumplido | app/privacidad.py, app/config.py |
| C7 Pruebas, evaluación y observabilidad | Cumplido | tests/, app/metricas.py |
| C8 Documentación y sustentación | Cumplido | este README documenta arquitectura, uso y alcance |

## Sets de evaluación

- `data/router_gold.json` contiene 36 consultas etiquetadas con las rutas actuales. Ejecuta `python -m eval.eval_router --proveedor hash` para medir la exactitud del router; este resultado evalúa el enrutamiento, no la calidad de la respuesta.
- `data/rag_gold.json` contiene 22 preguntas con respuesta esperada, evidencia por página y casos donde el sistema debe abstenerse. Es un set de referencia manual: todavía no hay un evaluador automático que calcule métricas RAG sobre él.
- Ambos sets son ejemplos académicos escritos manualmente. Antes de usarlos para decisiones de negocio, amplíalos y revísalos con expertos; la exactitud en un set pequeño no garantiza seguridad ni precisión en producción.

## Limitaciones conocidas

- El documento de benchmarking es la fuente de verdad; no se usa un analista humano ni evaluación crediticia, porque el objetivo del proyecto no es aprobar clientes.
- El sistema es sensible a la calidad del PDF: si el archivo cambia de formato, puede requerir ajuste del parser.
- OpenAI real requiere cuota disponible y puede fallar por limitación de presupuesto.
- La versión sin clave usa hash para pruebas y demostración local; no reemplaza un modelo semántico profesional.

## Conclusión

Este proyecto demuestra una arquitectura completa de RAG aplicada a un caso realista de benchmarking financiero: ingestión del PDF, segmentación lógica, base vectorial, recuperación con citas, chat conversacional y validación automática. Es una solución sólida para comparar condiciones entre Banco A y Banco B con evidencias extraídas del documento base.

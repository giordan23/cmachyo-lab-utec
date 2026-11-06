# Asistente de información sobre ahorros de Caja Huancayo

Este repositorio contiene un prototipo académico de búsqueda y conversación con RAG orientado a responder preguntas sobre productos y condiciones de ahorro de Caja Huancayo a partir de los documentos documentados por el usuario. La aplicación indexa el contenido, recupera citas relevantes y responde con evidencias sobre el material fuente sin inventar información no documentada.

El proyecto no es un asesor de crédito, no evalúa riesgo ni ofrece aprobación o rechazo, ni compara bancos ficticios. Su alcance se limita a la información de ahorro y condiciones que se encuentre en los documentos cargados.

## Objetivo del proyecto

Responder consultas del tipo:

- ¿Qué información existe sobre cuentas de ahorro de Caja Huancayo?
- ¿Qué requisitos o condiciones se mencionan para un depósito a plazo?
- ¿Qué se sabe sobre ahorro programado, requisitos o beneficios documentados?
- ¿Dónde aparece en los documentos una condición o producto específico?

## Arquitectura

1. Ingesta del documento
   - app/ingesta.py
   - extrae texto del documento base con pypdf
   - segmenta por artículo o sección
   - persiste metadatos y crea el índice vectorial en data/indice.npz

2. Embeddings y base vectorial
   - app/embeddings.py
   - usa un proveedor configurado (OpenAI o hash para entorno local)
   - normaliza y almacena los vectores para recuperación posterior

3. Recuperación RAG
   - app/rag.py
   - consulta los fragmentos más relevantes con top-k y umbral
   - muestra citas con id, título y página cuando el documento las incluye
   - responde con evidencia o con una indicación clara de falta de contexto

4. Chat conversacional
   - app/conversacion.py
   - conserva el historial útil y reformula preguntas de seguimiento
   - la interfaz web consulta al backend con streaming y citas

5. Enrutamiento y agente
   - app/router.py
   - decide si la consulta entra en el alcance del proyecto
   - app/agente_riesgo.py genera la respuesta con contexto y fuentes

6. Observabilidad y pruebas
   - app/metricas.py
   - pruebas en tests/
   - dashboard y métricas con Prometheus/Grafana

## Estructura del repositorio

```text
app/            lógica principal: ingest, embeddings, RAG, router, chat, LLM, API
rules/          reglas y utilidades complementarias del proyecto
data/          documentos fuente, rutas, índice generado, SQLite y otros artefactos
web/            interfaz del chat (HTML/JS)
tests/          pruebas de ingest, RAG, chat y validación del flujo
scripts/        scripts de entorno y verificación
docker-compose.yml
Dockerfile
requirements*.txt
README.md
```

## Requisitos

- Python 3.10+
- pip
- clave de OpenAI opcional para embeddings/LLM reales
- para entorno local sin costo: usar EMBEDDINGS_PROVIDER=hash y LLM_PROVIDER=mock

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
```

Si quieres usar OpenAI real:

```env
EMBEDDINGS_PROVIDER=openai
LLM_PROVIDER=openai
OPENAI_API_KEY=tu_clave
```

Nota: con OpenAI real, el uso depende de cuota disponible. Para un prototipo local estable, usa hash + mock.

## Inicio rápido

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

python -m pip install -r requirements-dev.txt
python -m app.ingesta
python -m app
```

Luego abre:

- http://127.0.0.1:8000/

## Ejemplos de consultas válidas

```text
¿Qué productos de ahorro ofrece Caja Huancayo según los documentos?
¿En qué condiciones se menciona un depósito a plazo?
¿Cuáles son los requisitos para una cuenta de ahorro?
¿Hay información documentada sobre ahorro programado o ahorro para jóvenes?
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

- El sistema responde con base en los documentos indexados; si el contenido no incluye la respuesta, la respuesta debe indicar falta de evidencia.
- No se debe asumir información oficial que no aparezca en los materiales cargados.
- OpenAI real requiere cuota disponible y puede fallar por limitación de presupuesto.
- La versión sin clave usa hash para pruebas locales y no reemplaza un modelo semántico profesional.

## Conclusión

Este proyecto demuestra un flujo completo de RAG para un asistente documental financiero: ingesta de documentos, segmentación, base vectorial, recuperación con citas, chat conversacional y respuestas basadas en evidencias. El objetivo es servir como prototipo informativo sobre productos y condiciones de ahorro de Caja Huancayo, sin presentar recomendaciones personales ni decisiones crediticias.

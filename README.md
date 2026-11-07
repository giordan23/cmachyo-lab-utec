# Asistente documental de benchmarking bancario

Aplicación de chat que responde preguntas sobre el documento PDF de comparación entre Banco A y Banco B. La respuesta se limita a la evidencia recuperada y muestra las páginas usadas como fuentes. No evalúa solicitudes ni solvencia de clientes.

## Alcance

- Consulta y comparación de tasas, comisiones, depósitos, ITF y condiciones descritas en `data/Benchmarking.pdf`.
- Distingue la información del tarifario de Banco A de los supuestos ficticios de Banco B y de las tasas de crédito que el PDF identifica como ficticias.
- Deriva consultas fuera del documento y se abstiene cuando no recupera evidencia.
- El PDF activo se configura con `PDF_PATH`; por defecto es `data/Benchmarking.pdf`.

## Componentes

- `app/ingesta.py`: extrae texto por página, lo divide en fragmentos solapados y crea el índice vectorial. El índice guarda una huella SHA-256 del PDF para reindexar al cambiar el documento.
- `app/rag.py` y `app/agente_documental.py`: recuperación, respuesta fundamentada y validación de citas.
- `app/router.py` y `data/rutas.json`: clasificación entre consulta del benchmarking y fuera de alcance.
- `app/conversacion.py`, `app/db.py` y `web/index.html`: chat conversacional persistido.
- `data/rag_gold.json`: preguntas etiquetadas con páginas y evidencia esperada.
- `eval/eval_rag.py`: evaluación reproducible de recuperación por páginas.
- `app/metricas.py`, Prometheus y Grafana: métricas de operación del chat y proveedores.

## Instalación y ejecución

Requisitos: Python 3.10+ y pip. Para ejecución local:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Configura `.env` a partir de `.env.example`. `ROUTER_PROVIDER=hash`, `EMBEDDINGS_PROVIDER=hash` y `LLM_PROVIDER=mock` permiten una demo local sin claves externas. Para respuestas de un modelo real, configura las claves y proveedores correspondientes.

```powershell
python -m app.ingesta
python -m app
```

Abre <http://127.0.0.1:8000/>. La API ofrece `POST /api/chat`, `GET /api/chat/{id}`, `GET /health` y `GET /metrics`.

Con Docker Compose, el contenedor indexa el PDF configurado si el índice falta, cambia de embedder o el hash del PDF ya no coincide:

```powershell
docker compose up --build
```

## Pruebas y evaluación

Las pruebas usan el **PDF real `data/Benchmarking.pdf`** y proveedores simulados, sin llamadas ni costos externos:

```powershell
python -m pytest
```

Evalúa el set de oro de recuperación con embeddings hash locales:

```powershell
python -m eval.eval_rag --proveedor hash
```

El informe incluye precisión y cobertura por página, MRR y tasa de preguntas con evidencia esperada recuperada. La prueba `tests/test_eval_rag.py` verifica una línea base de estas métricas y que las páginas/frases etiquetadas existan en el PDF. Evalúa el router con `python -m eval.eval_router --proveedor hash`; `tests/test_eval_router.py` protege su línea base. Para comparar el RAG con embeddings reales, usa `--proveedor openai`; requiere clave y puede tener costo.

La marca `debe_abstenerse` identifica preguntas que no deben recibir una afirmación no respaldada. Algunas sí tienen evidencia para explicar que un dato no está disponible; por eso el evaluador de recuperación no interpreta abstención como “no devolver contexto” ni puntúa la calidad semántica de la respuesta generada.

El set es pequeño y manual. Revísalo con especialistas y amplíalo con paráfrasis, preguntas ambiguas, fuera de alcance y casos de abstención. La cobertura de páginas mide recuperación, no exactitud factual ni calidad completa de las respuestas.

## Observabilidad

`docker-compose.yml` levanta Prometheus en el puerto 9090 y Grafana en el 3000. El dashboard principal presenta consultas por ruta, similitud de recuperación, consultas sin contexto, latencia, errores, tokens y costo estimado.

Los scores y volúmenes del tablero son métricas **operativas en tiempo real**, no métricas de precisión del set de oro. La calidad del recuperador se mide ejecutando `eval.eval_rag`; debe compararse entre cambios y revisarse junto con los casos que fallaron.

## Datos y limitaciones

`data/Benchmarking.pdf` es la fuente de verdad. El documento advierte que Banco B es ficticio y que algunas tasas de crédito son supuestos educativos; las respuestas deben conservar esas advertencias. El modo `hash` es una alternativa de prueba léxica, no sustituye embeddings semánticos para producción.

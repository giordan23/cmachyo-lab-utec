"""Paso 10: enmascarar datos personales ANTES de que salgan hacia Jev u OpenAI.

En el formulario el DNI tenía su propio campo y era fácil no enviarlo. En un chat el asesor escribe lo que quiere:
"mi cliente 40123456, Rosa, cel 987654321, ¿califica?". Por eso el enmascarado vive en los ADAPTADORES de salida
(JevRouter, OpenAILLM, OpenAIEmbedder): cualquier camino que llegue a un tercero pasa por aquí, aunque quien llama
se olvide. En la base de datos interna se guarda el texto original (es el registro de la entidad).

Límite honesto: son expresiones regulares. Detectan números con forma conocida; NO detectan nombres propios
("Rosa Quispe") ni direcciones. Para eso haría falta un detector de entidades (NER) y aun así no es perfecto.
"""
from __future__ import annotations

import re

from app import metricas

# (?<![\d.]) y (?![\d.]) evitan enmascarar parte de un número más largo o de un decimal (10000000.0 es un monto)
PATRONES = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("ruc", re.compile(r"(?<![\d.])(10|15|17|20)\d{9}(?![\d.])")),          # RUC peruano: 11 dígitos
    ("telefono", re.compile(r"(?<![\d.])(\+?51[\s-]?)?9\d{2}[\s-]?\d{3}[\s-]?\d{3}(?![\d.])")),
    ("dni", re.compile(r"(?<![\d.])\d{8}(?![\d.])")),
]


def enmascarar(texto: str) -> str:
    for tipo, rx in PATRONES:
        texto, n = rx.subn(f"[{tipo.upper()}]", texto)
        if n:
            metricas.PII_ENMASCARADO.labels(tipo).inc(n)
    return texto

"""Genera consultas documentales para observar métricas en Grafana.

Uso: python -m loadgen.generar_carga [--n 20] [--url http://localhost:8000]
"""
from __future__ import annotations

import random
import sys

import httpx

PREGUNTAS = [
    "¿Qué diferencia hay entre el ahorro en soles del Banco A y el Banco B?",
    "Compara las tasas de ahorro clásico en dólares.",
    "¿Qué TEA aparece para el depósito a plazo entre 91 y 180 días?",
    "¿Qué cuesta consultar el saldo en Kasnet?",
    "¿Qué porcentaje de ITF indica el documento?",
    "¿Qué ocurre si cancelo anticipadamente un plazo fijo?",
    "¿A qué hora abre la agencia?",
    "¿Cómo cambio la contraseña de mi correo?",
]


def main(argv: list[str]) -> None:
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 20
    url = argv[argv.index("--url") + 1] if "--url" in argv else "http://localhost:8000"
    rnd = random.Random(6)
    with httpx.Client(base_url=url, timeout=120) as cliente:
        for i in range(n):
            pregunta = rnd.choice(PREGUNTAS)
            respuesta = cliente.post("/api/chat", json={"mensaje": pregunta})
            respuesta.raise_for_status()
            datos = respuesta.json()
            meta = datos["mensajes"][1]["meta"]
            print(f"{i + 1:>3}/{n}  {meta['ruta']:<19} citas={len(meta.get('citas', []))}  {pregunta}")


if __name__ == "__main__":
    main(sys.argv[1:])

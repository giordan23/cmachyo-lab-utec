"""Arranque local:   python -m app            (http://127.0.0.1:8000/asesor)

Igual en Windows, macOS y Linux. Este laboratorio no usa drivers async de base de datos, así que el bucle de eventos
por defecto de Windows (ProactorEventLoop) no causa problemas.
"""
import os

import uvicorn

if __name__ == "__main__":
    try:
        uvicorn.run("app.main:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
    except KeyboardInterrupt:          # Ctrl+C: salir sin traza de error
        pass

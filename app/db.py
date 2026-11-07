"""Persistencia SQLite para conversaciones y mensajes del chat documental."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

ESQUEMA = """
CREATE TABLE IF NOT EXISTS conversaciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    creada_en REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS mensajes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversacion_id INTEGER NOT NULL REFERENCES conversaciones(id),
    creado_en REAL NOT NULL,
    rol TEXT NOT NULL,
    texto TEXT NOT NULL,
    meta TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_mensajes ON mensajes(conversacion_id, id);
"""


class NoExiste(LookupError):
    pass


class Repositorio:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.cx = sqlite3.connect(str(path), check_same_thread=False)
        self.cx.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.cx.executescript(ESQUEMA)

    def crear_conversacion(self) -> int:
        with self.lock, self.cx:
            return int(self.cx.execute("INSERT INTO conversaciones (creada_en) VALUES (?)",
                                       (time.time(),)).lastrowid)

    def existe_conversacion(self, id_: int) -> bool:
        with self.lock:
            return self.cx.execute("SELECT 1 FROM conversaciones WHERE id = ?", (id_,)).fetchone() is not None

    def agregar_mensaje(self, conversacion_id: int, rol: str, texto: str, meta: dict | None = None) -> dict:
        with self.lock, self.cx:
            cur = self.cx.execute(
                "INSERT INTO mensajes (conversacion_id, creado_en, rol, texto, meta) VALUES (?,?,?,?,?)",
                (conversacion_id, time.time(), rol, texto, json.dumps(meta or {}, ensure_ascii=False)))
            fila = self.cx.execute("SELECT * FROM mensajes WHERE id = ?", (cur.lastrowid,)).fetchone()
        return {**dict(fila), "meta": json.loads(fila["meta"])}

    def mensajes(self, conversacion_id: int, desde_id: int = 0) -> list[dict]:
        with self.lock:
            filas = self.cx.execute("SELECT * FROM mensajes WHERE conversacion_id = ? AND id > ? ORDER BY id",
                                    (conversacion_id, desde_id)).fetchall()
        return [{**dict(f), "meta": json.loads(f["meta"])} for f in filas]

"""Paso 9: persistencia con SQLite (módulo sqlite3 de la librería estándar).

¿Por qué no PostgreSQL como en el lab 04? Porque aquí basta un archivo, y `sqlite3` es SÍNCRONO: no depende del bucle
de eventos de asyncio, así que no aparece el problema de ProactorEventLoop que tuvimos en Windows con psycopg.

FastAPI ejecuta los endpoints `def` (no `async def`) en varios hilos; por eso una sola conexión protegida con un Lock.

Estados de solicitudes (tabla de compatibilidad): RESPONDIDA | DERIVADA | PENDIENTE_ANALISTA → APROBADA | RECHAZADA
  (La tabla solicitudes se conserva por compatibilidad con las pruebas de integración; en el flujo activo del asistente
  de ahorro se usan conversaciones y mensajes.)

Roles de mensaje: usuario | asistente | sistema
  - usuario: consulta enviada desde la interfaz
  - asistente: respuesta generada con RAG
  - sistema: avisos internos del flujo
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

ESQUEMA = """
CREATE TABLE IF NOT EXISTS solicitudes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    creada_en   REAL NOT NULL,
    estado      TEXT NOT NULL,
    ruta        TEXT NOT NULL,
    solicitud   TEXT NOT NULL,      -- JSON del formulario
    enrutamiento TEXT NOT NULL,     -- JSON de ResultadoRuta
    resultado   TEXT NOT NULL,      -- JSON del agente (dictamen, reglas, cálculos) o de la consulta
    decision    TEXT,
    analista    TEXT,
    comentario  TEXT,
    decidida_en REAL
);
CREATE INDEX IF NOT EXISTS ix_estado ON solicitudes(estado);
CREATE TABLE IF NOT EXISTS conversaciones (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    creada_en   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS mensajes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversacion_id INTEGER NOT NULL REFERENCES conversaciones(id),
    creado_en       REAL NOT NULL,
    rol             TEXT NOT NULL,      -- asesor | asistente | sistema
    texto           TEXT NOT NULL,
    meta            TEXT NOT NULL       -- JSON: ruta, citas, consulta reformulada, router…
);
CREATE INDEX IF NOT EXISTS ix_mensajes ON mensajes(conversacion_id, id);
"""


class NoExiste(LookupError):
    pass


class YaDecidida(RuntimeError):
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
            columnas = {c[1] for c in self.cx.execute("PRAGMA table_info(solicitudes)")}
            if "conversacion_id" not in columnas:       # migración: bases creadas por la versión sin chat
                self.cx.execute("ALTER TABLE solicitudes ADD COLUMN conversacion_id INTEGER")
                self.cx.commit()

    def _fila(self, row: sqlite3.Row | None) -> dict:
        if row is None:
            raise NoExiste
        d = dict(row)
        for k in ("solicitud", "enrutamiento", "resultado"):
            d[k] = json.loads(d[k])
        return d

    def crear(self, estado: str, ruta: str, solicitud: dict, enrutamiento: dict, resultado: dict,
              conversacion_id: int | None = None) -> int:
        with self.lock, self.cx:
            cur = self.cx.execute(
                "INSERT INTO solicitudes (creada_en, estado, ruta, solicitud, enrutamiento, resultado, conversacion_id) "
                "VALUES (?,?,?,?,?,?,?)",
                (time.time(), estado, ruta, *(json.dumps(x, ensure_ascii=False) for x in (solicitud, enrutamiento, resultado)),
                 conversacion_id))
            return int(cur.lastrowid)

    # ---------------------------------------------------------------------------------------------- conversaciones
    def crear_conversacion(self) -> int:
        with self.lock, self.cx:
            return int(self.cx.execute("INSERT INTO conversaciones (creada_en) VALUES (?)", (time.time(),)).lastrowid)

    def existe_conversacion(self, id_: int) -> bool:
        with self.lock:
            return self.cx.execute("SELECT 1 FROM conversaciones WHERE id = ?", (id_,)).fetchone() is not None

    def agregar_mensaje(self, conversacion_id: int, rol: str, texto: str, meta: dict | None = None) -> dict:
        with self.lock, self.cx:
            cur = self.cx.execute("INSERT INTO mensajes (conversacion_id, creado_en, rol, texto, meta) VALUES (?,?,?,?,?)",
                                  (conversacion_id, time.time(), rol, texto, json.dumps(meta or {}, ensure_ascii=False)))
            fila = self.cx.execute("SELECT * FROM mensajes WHERE id = ?", (cur.lastrowid,)).fetchone()
        return {**dict(fila), "meta": json.loads(fila["meta"])}

    def mensajes(self, conversacion_id: int, desde_id: int = 0) -> list[dict]:
        with self.lock:
            filas = self.cx.execute("SELECT * FROM mensajes WHERE conversacion_id = ? AND id > ? ORDER BY id",
                                    (conversacion_id, desde_id)).fetchall()
        return [{**dict(f), "meta": json.loads(f["meta"])} for f in filas]

    def obtener(self, id_: int) -> dict:
        with self.lock:
            return self._fila(self.cx.execute("SELECT * FROM solicitudes WHERE id = ?", (id_,)).fetchone())

    def pendientes(self) -> list[dict]:
        with self.lock:
            filas = self.cx.execute("SELECT * FROM solicitudes WHERE estado = 'PENDIENTE_ANALISTA' ORDER BY id").fetchall()
        return [self._fila(f) for f in filas]

    def contar_pendientes(self) -> int:
        with self.lock:
            return self.cx.execute("SELECT COUNT(*) FROM solicitudes WHERE estado = 'PENDIENTE_ANALISTA'").fetchone()[0]

    def decidir(self, id_: int, decision: str, analista: str, comentario: str) -> dict:
        """El WHERE estado='PENDIENTE_ANALISTA' hace la operación ATÓMICA: si dos analistas deciden a la vez,
        solo uno actualiza la fila (rowcount=1); el otro recibe YaDecidida."""
        with self.lock, self.cx:
            cur = self.cx.execute(
                "UPDATE solicitudes SET estado = ?, decision = ?, analista = ?, comentario = ?, decidida_en = ? "
                "WHERE id = ? AND estado = 'PENDIENTE_ANALISTA'", (decision, decision, analista, comentario, time.time(), id_))
            actualizadas = cur.rowcount
        if actualizadas == 0:
            fila = self.obtener(id_)                     # lanza NoExiste si no está
            raise YaDecidida(f"la solicitud {id_} está en estado {fila['estado']}")
        return self.obtener(id_)

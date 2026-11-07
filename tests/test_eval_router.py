import json

from app.config import DATA_DIR, Settings
from eval.eval_router import crear


def test_router_hash_supera_linea_base_del_set_de_oro():
    settings = Settings()
    settings.embeddings_provider = "hash"
    router = crear("hash", settings)
    casos = json.loads((DATA_DIR / "router_gold.json").read_text(encoding="utf-8"))
    aciertos = sum(router.enrutar(caso["texto"]).ruta == caso["ruta"] for caso in casos)
    assert aciertos / len(casos) >= 0.9

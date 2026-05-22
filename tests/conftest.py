"""tests/conftest.py
Socle de tests pour la couche persistance (défi D4 — feedback / attribution).

Fournit une fixture `ltm` adossée à une base SQLite in-memory, neuve à chaque
test, qui reproduit le comportement Postgres utile aux tests :

  - La migration Postgres-only de `LongTermMemory` est neutralisée par le garde
    de dialecte (`engine.dialect.name != 'postgresql'` → no-op), donc le schéma
    provient uniquement de `Base.metadata.create_all`.
  - SQLite désactive les clés étrangères par défaut : on les active par
    connexion via un event listener SQLAlchemy (pas en dur dans le code applicatif).
  - La vue `reponses_ratees` (créée côté Supabase en production) est matérialisée
    ici pour rendre `get_failed_responses()` testable.

Invocation : `venv/bin/python -m pytest tests/ -v` depuis la racine du projet
(le cwd est alors sur sys.path, ce qui résout `from utils...`).
"""

from __future__ import annotations

import pytest
from sqlalchemy import event, text

from utils.memory.long_term import LongTermMemory


# Vue des réponses mal notées (👎), variante SQLite (syntaxe compatible Postgres).
# En production, la vue est gérée côté Supabase ; ici on la crée pour le test.
_VIEW_REPONSES_RATEES = """
CREATE VIEW IF NOT EXISTS reponses_ratees AS
SELECT
    f.id                AS feedback_id,
    f.created_at        AS created_at,
    f.session_id        AS session_id,
    f.user_id           AS user_id,
    f.message_id        AS message_id,
    f.question_snapshot AS question,
    f.response_snapshot AS response,
    f.langfuse_trace_id AS langfuse_trace_id
FROM feedback f
WHERE f.rating = -1
ORDER BY f.created_at DESC
"""


@pytest.fixture
def ltm():
    """LongTermMemory sur SQLite in-memory, isolée par test.

    Yields:
        LongTermMemory: instance prête (schéma créé, FK activées, vue présente).
    """
    memory = LongTermMemory(database_url="sqlite:///:memory:")

    # FK SQLite : activées par connexion via un listener (et non en dur).
    # Couvre toute connexion future ouverte par le pool.
    @event.listens_for(memory.engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _record):  # pragma: no cover - hook
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    # La connexion StaticPool de :memory: a déjà été ouverte par __init__
    # (create_all) avant l'enregistrement du listener : on force donc le PRAGMA
    # une fois sur la connexion courante, puis on crée la vue de test.
    with memory.engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(text(_VIEW_REPONSES_RATEES))

    return memory

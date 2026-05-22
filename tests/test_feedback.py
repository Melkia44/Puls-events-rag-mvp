"""tests/test_feedback.py
Tests de la couche feedback / attribution vote ↔ message (défi D4).

Couvre :
  - log_message() retourne désormais l'ID auto-généré (clé d'attribution) ;
  - add_feedback() avec message_id + snapshots + trace_id, relecture cohérente ;
  - add_feedback() avec message_id orphelin : pas de crash, warning, FK nullifiée ;
  - get_failed_responses() via la vue reponses_ratees ;
  - garde de rating invalide.

Toutes les bases sont des SQLite in-memory (fixture `ltm` de conftest.py).
"""

from __future__ import annotations

import logging

from utils.memory.long_term import Feedback


def test_log_message_returns_id(ltm):
    """log_message renvoie un entier > 0 (l'ID Postgres/SQLite du message)."""
    uid = ltm.get_or_create_user("Léa")
    sid = ltm.start_session(uid)

    mid = ltm.log_message(sid, "user", "Quels concerts ce week-end ?")

    assert isinstance(mid, int)
    assert mid > 0


def test_add_feedback_with_message_id(ltm):
    """Vote 👎 rattaché à un message existant : tous les champs persistés."""
    uid = ltm.get_or_create_user("Léa")
    sid = ltm.start_session(uid)
    ltm.log_message(sid, "user", "Quels concerts ce week-end ?")
    aid = ltm.log_message(sid, "assistant", "Voici 3 concerts à Nantes…")

    fb_id = ltm.add_feedback(
        session_id=sid,
        user_id=uid,
        rating=-1,
        message_id=aid,
        question_snapshot="Quels concerts ce week-end ?",
        response_snapshot="Voici 3 concerts à Nantes…",
        langfuse_trace_id="trace-xyz-123",
    )

    assert isinstance(fb_id, int) and fb_id > 0

    # Relecture directe de la table pour vérifier la cohérence des champs.
    with ltm._session() as s:
        row = s.get(Feedback, fb_id)
        assert row is not None
        assert row.rating == -1
        assert row.message_id == aid
        assert row.user_id == uid
        assert row.session_id == sid
        assert row.question_snapshot == "Quels concerts ce week-end ?"
        assert row.response_snapshot == "Voici 3 concerts à Nantes…"
        assert row.langfuse_trace_id == "trace-xyz-123"


def test_add_feedback_invalid_message_id(ltm, caplog):
    """message_id inexistant : pas de crash, warning loggé, FK nullifiée."""
    uid = ltm.get_or_create_user("Léa")
    sid = ltm.start_session(uid)

    with caplog.at_level(logging.WARNING):
        fb_id = ltm.add_feedback(
            session_id=sid,
            user_id=uid,
            rating=1,
            message_id=999999,  # n'existe pas
        )

    # Le vote est tout de même persisté (les snapshots suffisent à l'analyse).
    assert isinstance(fb_id, int) and fb_id > 0
    assert any("introuvable" in r.message for r in caplog.records)

    with ltm._session() as s:
        row = s.get(Feedback, fb_id)
        assert row.message_id is None  # FK nullifiée, pas de violation


# --- Tests complémentaires (renforcement, hors minimum requis) ---------------


def test_get_failed_responses_reads_view(ltm):
    """get_failed_responses ne remonte que les 👎 via la vue reponses_ratees."""
    uid = ltm.get_or_create_user("Léa")
    sid = ltm.start_session(uid)
    aid = ltm.log_message(sid, "assistant", "Réponse décevante")

    ltm.add_feedback(sid, uid, -1, message_id=aid,
                     response_snapshot="Réponse décevante")
    ltm.add_feedback(sid, uid, 1, message_id=aid,
                     response_snapshot="Réponse appréciée")

    df = ltm.get_failed_responses(limit=10)

    assert len(df) == 1                       # seul le 👎 remonte
    assert int(df.iloc[0]["session_id"]) == sid


def test_add_feedback_invalid_rating_returns_none(ltm):
    """Un rating hors {1,-1} est ignoré (None), rien n'est persisté."""
    uid = ltm.get_or_create_user("Léa")
    sid = ltm.start_session(uid)

    assert ltm.add_feedback(sid, uid, 5) is None
    assert ltm.get_csat()["total"] == 0

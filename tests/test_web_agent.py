"""
tests/test_web_agent.py
Tests unitaires pour utils.web_agent — couvre routeur, whitelist,
client Brave (mocké), pipeline complet.

Run :
    pytest tests/test_web_agent.py -v
    pytest tests/test_web_agent.py -v -m "not network"
"""

from __future__ import annotations
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.web_agent import (
    DomainWhitelist,
    BraveSearchClient,
    WebResult,
    web_search_filtered,
    route_to_rag_or_web,
    _FORCE_WEB_REGEX,
    _FORCE_WEB_PROPERNOUN_REGEX,
    ROUTER_PROMPT_TEMPLATE,
)


# ============================================================================
# Fixture — whitelist test
# ============================================================================

@pytest.fixture
def whitelist_test():
    """Whitelist construite depuis un YAML temporaire pour tests isolés."""
    import tempfile
    yaml_content = """
domains:
  - domain: ouest-france.fr
    category: presse
    icon: "📰"
    priority: 1
  - domain: stereolux.org
    category: lieu_culturel
    icon: "🎭"
    priority: 1
  - domain: fnacspectacles.com
    category: billetterie
    icon: "🎫"
    priority: 1
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        path = f.name

    yield DomainWhitelist(path)

    # Cleanup
    os.unlink(path)


# ============================================================================
# Test 1 — Routeur RAG vs WEB
# ============================================================================

class TestRouter:
    """Tests des 3 mécanismes de routage : triggers, retrieval, LLM."""

    @pytest.mark.parametrize("question", [
        "Y a-t-il encore des billets pour ce soir ?",
        "Les réservations sont ouvertes ?",
        "Quels sont les festivals annoncés pour 2026 ?",
        "Le concert est-il complet ?",
        "Dernières actualités culturelles à Nantes ?",
    ])
    def test_force_web_triggers(self, question):
        """Les triggers regex doivent court-circuiter le LLM."""
        decision, reason = route_to_rag_or_web(question, llm_router=None)
        assert decision == "web"
        assert reason == "trigger_keyword"

    def test_low_retrieval_count_routes_to_web(self):
        """Si retrieval renvoie <3 docs, basculer en web."""
        decision, reason = route_to_rag_or_web(
            "Quels concerts ce soir ?",
            llm_router=None,
            retrieval_count=2,
        )
        assert decision == "web"
        assert "retrieval_count" in reason

    def test_low_retrieval_score_routes_to_web(self):
        """Si score similarité < 0.6, basculer en web."""
        decision, reason = route_to_rag_or_web(
            "Quels événements ?",
            llm_router=None,
            retrieval_score=0.45,
        )
        assert decision == "web"
        assert "retrieval_score" in reason

    def test_no_llm_no_triggers_defaults_to_rag(self):
        """Sans LLM dispo, sans trigger, on reste sur RAG (sûr)."""
        decision, reason = route_to_rag_or_web(
            "Recommande-moi un concert ce week-end",
            llm_router=None,
        )
        assert decision == "rag"
        assert reason == "no_llm_default"

    def test_llm_decision_web(self):
        """Le routeur LLM peut décider WEB."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="WEB")

        decision, reason = route_to_rag_or_web(
            "Quelque chose qui demande du web",
            llm_router=mock_llm,
        )
        assert decision == "web"
        assert reason == "llm_router"

    def test_llm_decision_rag(self):
        """Le routeur LLM peut décider RAG."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="RAG")

        decision, reason = route_to_rag_or_web(
            "Quels concerts à Nantes ?",
            llm_router=mock_llm,
        )
        assert decision == "rag"

    def test_llm_error_falls_back_to_rag(self):
        """Si le LLM plante, on retombe sur RAG (safety first)."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API down")

        decision, reason = route_to_rag_or_web(
            "Quels concerts à Nantes ?",
            llm_router=mock_llm,
        )
        assert decision == "rag"
        assert reason == "llm_error"

    # ── Fix D3-2 — triggers « nom propre » (regex sensible à la casse) ──────

    @pytest.mark.parametrize("question", [
        "Y'a-t-il un concert de Stromae prochainement ?",
        "Spectacle d'Aya Nakamura à Paris",
        "Tournée de Taylor Swift en France",
        "Festival de Cannes 2026",
    ])
    def test_trigger_matches_named_person(self, question):
        """Nom propre + terme événementiel → web via trigger_propernoun.

        Testé via route_to_rag_or_web (la regex nom propre est volontairement
        séparée de _FORCE_WEB_REGEX, cf. fix D3-2).
        """
        decision, reason = route_to_rag_or_web(question, llm_router=None)
        assert decision == "web", f"Devrait router web : {question!r}"
        assert reason == "trigger_propernoun"

    @pytest.mark.parametrize("question", [
        "Concert à Nantes ce week-end",
        "Spectacle de théâtre pour enfants",
        "Festival de jazz à Bordeaux",   # 'jazz' minuscule, pas un nom propre
        "Expo de peinture contemporaine",
    ])
    def test_trigger_does_not_match_generic(self, question):
        """Questions génériques sans nom propre → PAS de trigger_propernoun."""
        assert _FORCE_WEB_PROPERNOUN_REGEX.search(question) is None, \
            f"Ne devrait PAS matcher : {question!r}"
        # Sans LLM ni trigger, on reste sur RAG (sûr).
        decision, reason = route_to_rag_or_web(question, llm_router=None)
        assert decision == "rag"
        assert reason != "trigger_propernoun"

    def test_keyword_triggers_priority_over_propernoun(self):
        """Un trigger mot-clé reste prioritaire (raison trigger_keyword)."""
        # "billet" + nom propre : le mot-clé court-circuite en premier.
        decision, reason = route_to_rag_or_web(
            "Où acheter des billets pour Stromae à Paris ?",
            llm_router=None,
        )
        assert decision == "web"
        assert reason == "trigger_keyword"

    def test_router_prompt_v3_has_named_person_rule(self):
        """Le prompt v3 explicite la règle 'nom propre' et les limites catalogue."""
        assert "NOMMÉMENT" in ROUTER_PROMPT_TEMPLATE
        assert "tournées nationales" in ROUTER_PROMPT_TEMPLATE
        assert "OpenAgenda" in ROUTER_PROMPT_TEMPLATE


# ============================================================================
# Test 2 — Whitelist domain matching
# ============================================================================

class TestWhitelist:
    """Vérifie le matching exact + sous-domaines + filtrage hors-liste."""

    def test_exact_match(self, whitelist_test):
        assert whitelist_test.is_allowed("https://ouest-france.fr/article/123")
        assert whitelist_test.is_allowed("https://stereolux.org/agenda")

    def test_www_stripped(self, whitelist_test):
        """www.ouest-france.fr doit matcher ouest-france.fr."""
        assert whitelist_test.is_allowed("https://www.ouest-france.fr/x")

    def test_subdomain_allowed(self, whitelist_test):
        """boutique.fnacspectacles.com doit matcher fnacspectacles.com."""
        assert whitelist_test.is_allowed("https://boutique.fnacspectacles.com/x")

    def test_rejects_unknown_domain(self, whitelist_test):
        assert not whitelist_test.is_allowed("https://lemonde.fr/article")
        assert not whitelist_test.is_allowed("https://random-spam.io")

    def test_rejects_empty_url(self, whitelist_test):
        assert not whitelist_test.is_allowed("")
        assert not whitelist_test.is_allowed(None)

    def test_entry_metadata(self, whitelist_test):
        """get_entry renvoie l'icône et catégorie."""
        entry = whitelist_test.get_entry("https://stereolux.org/agenda")
        assert entry is not None
        assert entry.icon == "🎭"
        assert entry.category == "lieu_culturel"


# ============================================================================
# Test 3 — Pipeline de recherche (Brave mocké, filtrage, fallback)
# ============================================================================

class TestWebSearchPipeline:
    """Pipeline complet — Brave puis DDG, filtrage par whitelist."""

    def test_brave_results_filtered(self, whitelist_test):
        """Les résultats Brave hors-whitelist sont filtrés."""
        # Mock du client Brave : retourne 5 résultats, dont 2 in-list
        mock_client = MagicMock(spec=BraveSearchClient)
        mock_client.is_available.return_value = True
        mock_client.search.return_value = [
            {"url": "https://ouest-france.fr/a1", "title": "Article 1",
             "description": "Snippet 1"},
            {"url": "https://random-spam.io/x", "title": "Spam",
             "description": "Bad"},
            {"url": "https://stereolux.org/agenda", "title": "Agenda",
             "description": "Concerts"},
            {"url": "https://twitter.com/post", "title": "Tweet",
             "description": "Hot take"},
            {"url": "https://lemonde.fr/article", "title": "Le Monde",
             "description": "National"},
        ]

        results = web_search_filtered("test query", whitelist_test, mock_client)

        assert len(results) == 2
        urls = [r.url for r in results]
        assert any("ouest-france.fr" in u for u in urls)
        assert any("stereolux.org" in u for u in urls)

    def test_results_sorted_by_priority(self, whitelist_test):
        """Les résultats priorité 1 passent avant priorité 2+."""
        # Tous les domaines test_yaml ont priorité 1 → fallback ordre original
        # Test plus parlant avec une whitelist mixée (ici tous équivalents)
        mock_client = MagicMock(spec=BraveSearchClient)
        mock_client.is_available.return_value = True
        mock_client.search.return_value = [
            {"url": "https://ouest-france.fr/a", "title": "A", "description": ""},
        ]
        results = web_search_filtered("test", whitelist_test, mock_client)
        assert len(results) == 1

    def test_ddg_fallback_when_brave_empty(self, whitelist_test, monkeypatch):
        """Si Brave retourne [], on bascule sur DDG."""
        mock_brave = MagicMock(spec=BraveSearchClient)
        mock_brave.is_available.return_value = True
        mock_brave.search.return_value = []  # Brave vide

        def fake_ddg(query, max_results=10):
            return [
                {"url": "https://ouest-france.fr/x", "title": "X",
                 "description": "Y"}
            ]

        monkeypatch.setattr("utils.web_agent._ddg_fallback_search", fake_ddg)

        results = web_search_filtered("test", whitelist_test, mock_brave)
        assert len(results) == 1

    def test_no_brave_key_uses_ddg_directly(self, whitelist_test, monkeypatch):
        """Si BRAVE_API_KEY absent, on saute Brave et on utilise DDG."""
        mock_brave = MagicMock(spec=BraveSearchClient)
        mock_brave.is_available.return_value = False  # pas de clé API

        called = {"count": 0}

        def fake_ddg(query, max_results=10):
            called["count"] += 1
            return [
                {"url": "https://stereolux.org/y", "title": "Y",
                 "description": "Z"}
            ]

        monkeypatch.setattr("utils.web_agent._ddg_fallback_search", fake_ddg)

        results = web_search_filtered("test", whitelist_test, mock_brave)
        assert called["count"] == 1
        assert len(results) == 1

    def test_empty_when_nothing_passes_whitelist(self, whitelist_test):
        """Si aucun résultat n'est dans la whitelist, retour vide."""
        mock_client = MagicMock(spec=BraveSearchClient)
        mock_client.is_available.return_value = True
        mock_client.search.return_value = [
            {"url": "https://lemonde.fr/x", "title": "X", "description": "Y"},
            {"url": "https://twitter.com/y", "title": "Y", "description": "Z"},
        ]

        results = web_search_filtered("test", whitelist_test, mock_client)
        assert results == []


# ============================================================================
# Test 4 — Garde-fous citation et erreurs API
# ============================================================================

class TestSafety:
    """Vérifie les garde-fous critiques."""

    def test_brave_429_returns_empty(self):
        """Quota Brave dépassé → retourne [] proprement."""
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_resp.ok = False
            mock_get.return_value = mock_resp

            client = BraveSearchClient(api_key="fake")
            results = client.search("test")
            assert results == []

    def test_brave_401_returns_empty(self):
        """Clé API invalide → retourne [] proprement."""
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_resp.ok = False
            mock_get.return_value = mock_resp

            client = BraveSearchClient(api_key="fake")
            results = client.search("test")
            assert results == []

    def test_brave_network_error(self):
        """Erreur réseau → retourne [] proprement."""
        import requests
        with patch("requests.get", side_effect=requests.Timeout):
            client = BraveSearchClient(api_key="fake")
            results = client.search("test")
            assert results == []


# ============================================================================
# Tests réseau live (skip par défaut)
# ============================================================================

@pytest.mark.network
class TestBraveLive:
    """Tests d'intégration Brave Search live — nécessite BRAVE_API_KEY."""

    def test_brave_real_search(self):
        """Smoke test : search réel renvoie au moins 1 résultat."""
        client = BraveSearchClient()
        if not client.is_available():
            pytest.skip("BRAVE_API_KEY absente")

        results = client.search("Nantes culture concert")
        assert len(results) > 0
        assert "url" in results[0] or "title" in results[0]

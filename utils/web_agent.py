"""
utils/web_agent.py
Recherche web temps réel D3 — Puls-Events MVP P13.

Responsabilités :
    - Routeur amont : décider RAG suffit OU agent web nécessaire
      (Mistral Small, ~500ms, prompt court)
    - Client Brave Search avec fallback DuckDuckGo
    - Whitelist de domaines confiance (12 domaines presse/billetterie/lieux)
    - Orchestration smolagents avec citation obligatoire

Choix techniques justifiés :
    - smolagents (Hugging Face) plutôt que LangGraph/CrewAI : code-first,
      minimaliste, pas de graphe d'états à debugger (cohérent scope MVP)
    - Brave Search API plutôt que Tavily : 2000 req/mois free + souveraineté
      non-US (cohérent stack Puls : Mistral, Qdrant EU, Supabase Paris)
    - DuckDuckGo en fallback : sans quota, sans clé, lib python pure
    - Whitelist stricte vs blacklist : sécurité, qualité éditoriale,
      conformité (faux billets, désinfo)

Anti-patterns évités :
    - Pas d'appel agent sans citation des sources (faithfulness RAG)
    - Pas de cache web (à la différence du géocoding) — les infos web
      sont par nature volatiles, un cache 30j leur ferait perdre leur
      raison d'être
    - Pas de retry automatique sur l'agent : si Brave + DDG échouent
      tous les deux, on log et on dégrade gracieusement vers le RAG

Réf. backlog : US-601 (routeur), US-602 (Brave), US-603 (whitelist),
               US-604 (citation), US-605 (badge UI). R-09 (dépendance
               smolagents framework émergent) mitigé par cette couche
               d'abstraction PulsWebAgent.
"""

from __future__ import annotations
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import yaml

logger = logging.getLogger(__name__)


# [D4] Handler Langfuse partagé pour ce module. Singleton lazy : on évite
# de recréer un CallbackHandler à chaque invoke LLM (hot-path : routeur +
# agent web) et le prop drilling depuis app.py. Sentinelle distincte de
# None car None = "init échouée / Langfuse indispo" (dégradation gracieuse).
_LANGFUSE_HANDLER: Any = "uninitialized"


def _langfuse_config() -> dict:
    """Retourne le config LangChain Langfuse, ou {} si Langfuse indisponible."""
    global _LANGFUSE_HANDLER
    if _LANGFUSE_HANDLER == "uninitialized":
        try:
            from langfuse.langchain import CallbackHandler
            _LANGFUSE_HANDLER = CallbackHandler()
        except Exception as e:
            logger.warning(f"Langfuse indisponible ({e}) — traçage D4 désactivé")
            _LANGFUSE_HANDLER = None
    return {"callbacks": [_LANGFUSE_HANDLER]} if _LANGFUSE_HANDLER else {}


# ============================================================================
# CONFIG
# ============================================================================

# Endpoint API Brave Search — https://api.search.brave.com
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Timeout des requêtes web — agressif car on est dans le chemin utilisateur
WEB_TIMEOUT_SECONDS = 6

# Nombre max de résultats web à retourner après filtrage
MAX_WEB_RESULTS = 4

# Nombre de résultats demandés à Brave AVANT filtrage whitelist
# (on demande plus pour avoir de la marge si la majorité n'est pas dans
# la liste blanche)
BRAVE_RAW_RESULTS = 15


# ============================================================================
# WHITELIST DE DOMAINES
# ============================================================================

def _normalize_domain(value: str) -> str:
    """Domaine normalisé pour matching whitelist : lowercase, sans 'www.'.

    Accepte une URL complète ('https://www.x.fr/p') ou un domaine nu
    ('x.fr') — urlparse renvoie un netloc vide sur un domaine nu, d'où
    le fallback `or value`.
    """
    netloc = urlparse(value).netloc or value
    return netloc.lower().removeprefix("www.")


@dataclass
class DomainEntry:
    """Une entrée de la whitelist YAML, structurée pour l'affichage."""
    domain: str
    category: str
    icon: str
    priority: int

    @property
    def matches(self) -> str:
        """Forme normalisée pour matching (sans www., lowercase)."""
        return _normalize_domain(self.domain)


class DomainWhitelist:
    """Whitelist chargée depuis config/domain_whitelist.yaml.

    Pourquoi YAML plutôt que hardcodé en Python : Jérémy ou un OPS
    peut modifier la liste sans toucher au code (pas de rebuild).
    Pour V2, le format prévoit déjà category/icon/priority pour
    l'affichage enrichi UI.
    """

    def __init__(self, yaml_path: str | Path):
        self.yaml_path = Path(yaml_path)
        self._entries: Dict[str, DomainEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.yaml_path.exists():
            logger.error(f"Whitelist introuvable : {self.yaml_path}")
            return

        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error(f"YAML invalide dans {self.yaml_path} : {e}")
            return

        for raw in data.get("domains", []):
            try:
                entry = DomainEntry(
                    domain=raw["domain"].lower(),
                    category=raw.get("category", "autre"),
                    icon=raw.get("icon", "🔗"),
                    priority=int(raw.get("priority", 99)),
                )
                self._entries[entry.matches] = entry
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Entrée whitelist mal formée : {raw} ({e})")

        logger.info(f"Whitelist chargée : {len(self._entries)} domaines")

    def is_allowed(self, url: str) -> bool:
        """True si l'URL match un des domaines de la whitelist."""
        if not url:
            return False
        try:
            netloc = _normalize_domain(url)
        except Exception:
            return False
        # Match exact ou suffixe (sous-domaines acceptés du même 2LD)
        if netloc in self._entries:
            return True
        # Permet sous-domaines : "boutique.fnacspectacles.com" matche
        # "fnacspectacles.com"
        for whitelisted in self._entries:
            if netloc.endswith("." + whitelisted):
                return True
        return False

    def get_entry(self, url: str) -> Optional[DomainEntry]:
        """Retourne l'entry correspondante ou None."""
        if not url:
            return None
        try:
            netloc = _normalize_domain(url)
        except Exception:
            return None
        if netloc in self._entries:
            return self._entries[netloc]
        for matches_key, entry in self._entries.items():
            if netloc.endswith("." + matches_key):
                return entry
        return None

    def count(self) -> int:
        return len(self._entries)


# ============================================================================
# CLIENT BRAVE SEARCH + FALLBACK DDG
# ============================================================================

@dataclass
class WebResult:
    """Résultat unifié issu de Brave OU DDG, après filtrage whitelist."""
    title: str
    url: str
    snippet: str
    source_icon: str = "🔗"
    source_category: str = "autre"


class BraveSearchClient:
    """Client minimal pour l'API Brave Search.

    On ne wrap pas tout l'API — juste search() avec les params essentiels.
    Codes d'erreur gérés :
        - 401/403 : clé API invalide → log error + retourne []
        - 429 : quota dépassé → log warning + retourne [] pour fallback DDG
        - Timeout/network : log + retourne []
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            logger.warning("BRAVE_API_KEY absent — Brave désactivé (DDG only)")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def search(
        self,
        query: str,
        count: int = BRAVE_RAW_RESULTS,
        country: str = "FR",
    ) -> List[Dict[str, Any]]:
        """Retourne les résultats bruts Brave (à filtrer ensuite)."""
        if not self.is_available():
            return []

        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        params = {
            "q": query,
            "count": min(count, 20),
            "country": country,
            "search_lang": "fr",
            "safesearch": "moderate",
        }

        try:
            resp = requests.get(
                BRAVE_SEARCH_URL,
                headers=headers,
                params=params,
                timeout=WEB_TIMEOUT_SECONDS,
            )
        except requests.RequestException as e:
            logger.warning(f"Brave search network error : {e}")
            return []

        if resp.status_code == 429:
            logger.warning("Brave search quota dépassé — bascule DDG")
            return []
        if resp.status_code in (401, 403):
            logger.error(f"Brave search auth error {resp.status_code}")
            return []
        if not resp.ok:
            logger.warning(f"Brave search HTTP {resp.status_code}")
            return []

        try:
            data = resp.json()
        except ValueError:
            logger.warning("Brave search JSON invalide")
            return []

        # Format Brave : data["web"]["results"] est une liste de dicts
        web_data = data.get("web", {})
        return web_data.get("results", []) or []


def _ddg_fallback_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Fallback DuckDuckGo via la lib `duckduckgo-search`.

    Pas de clé API, pas de quota. Qualité moindre que Brave sur les contenus
    francophones mais largement acceptable pour de la presse locale identifiée.

    Importée tard pour ne pas planter le module si la lib manque (elle est
    optionnelle au sens strict, mais recommandée).
    """
    try:
        from ddgs import DDGS  # nouvelle lib renommée
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # ancienne lib
        except ImportError:
            logger.error("Ni 'ddgs' ni 'duckduckgo_search' installé — pas de fallback")
            return []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                region="fr-fr",
                safesearch="moderate",
                max_results=max_results,
            ))
    except Exception as e:
        logger.warning(f"DDG fallback error : {e}")
        return []

    # Normaliser au format pseudo-Brave pour réutiliser le pipeline filtrage
    normalized = []
    for r in results:
        normalized.append({
            "title": r.get("title", ""),
            "url": r.get("href", "") or r.get("url", ""),
            "description": r.get("body", "") or r.get("snippet", ""),
        })
    return normalized


# ============================================================================
# PIPELINE DE RECHERCHE (Brave + fallback DDG + filtrage whitelist)
# ============================================================================

def web_search_filtered(
    query: str,
    whitelist: DomainWhitelist,
    brave_client: Optional[BraveSearchClient] = None,
    max_results: int = MAX_WEB_RESULTS,
) -> List[WebResult]:
    """Orchestrateur : Brave primary → DDG fallback → filtrage → top-K.

    Le pipeline complet :
        1. Tente Brave (si clé API configurée)
        2. Si Brave KO/vide : fallback DDG
        3. Filtre les résultats par la whitelist
        4. Trie par priorité whitelist (croissante)
        5. Retourne les max_results premiers

    Returns:
        Liste de WebResult (peut être vide si rien n'est dans la whitelist)
    """
    brave_client = brave_client or BraveSearchClient()
    raw_results: List[Dict[str, Any]] = []
    source_used: str = "none"

    # Étape 1 — Brave en primaire
    if brave_client.is_available():
        raw_results = brave_client.search(query)
        if raw_results:
            source_used = "brave"

    # Étape 2 — DDG en fallback si Brave a échoué ou n'a rien rendu
    if not raw_results:
        raw_results = _ddg_fallback_search(query)
        if raw_results:
            source_used = "ddg"

    if not raw_results:
        logger.info(f"Recherche web vide pour '{query}'")
        return []

    # Étape 3 — Filtrage whitelist
    filtered: List[Tuple[int, WebResult]] = []  # (priority, result)
    for raw in raw_results:
        url = raw.get("url") or raw.get("href") or ""
        if not whitelist.is_allowed(url):
            continue

        entry = whitelist.get_entry(url)
        if entry is None:
            # Edge case : is_allowed True mais get_entry None — défensif
            continue

        result = WebResult(
            title=(raw.get("title") or "").strip(),
            url=url,
            snippet=(raw.get("description") or raw.get("snippet") or "").strip(),
            source_icon=entry.icon,
            source_category=entry.category,
        )
        filtered.append((entry.priority, result))

    # Étape 4 — Tri par priorité (croissante : 1 = prioritaire)
    filtered.sort(key=lambda x: x[0])

    # Étape 4bis — Déduplication par domaine. Brave renvoie souvent
    # plusieurs pages d'un même site (fnac ×2, infoconcert ×2) : sans
    # dédup, le top-4 se remplit de 2 domaines redondants. On garde la
    # 1re occurrence de chaque domaine (= la plus prioritaire, post-tri)
    # pour maximiser la diversité des sources dans un budget réduit.
    # Compromis assumé : 2 articles distincts d'un même média → 1 seul
    # retenu (acceptable pour un MVP, max_results=4).
    deduped: List[WebResult] = []
    seen_domains: set[str] = set()
    for _, result in filtered:
        try:
            domain = _normalize_domain(result.url)
        except Exception:
            domain = result.url
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        deduped.append(result)

    # Étape 5 — Top-K
    top = deduped[:max_results]

    logger.info(
        f"Web search : {len(raw_results)} bruts via {source_used}, "
        f"{len(filtered)} whitelistés, {len(deduped)} après dédup domaine, "
        f"{len(top)} retournés (top={max_results})"
    )
    return top


# ============================================================================
# ROUTEUR LLM AMONT — RAG vs Web
# ============================================================================

# Triggers explicites qui forcent l'appel agent web (court-circuit du routeur LLM)
# Conçus pour les cas où le RAG ne PEUT PAS répondre (info temps-réel, hors corpus)
#
# Note technique : on n'utilise PAS \b (word boundary) à cause d'un comportement
# imprévisible de Python re avec les caractères accentués (é, è, à...).
# Patterns simples sans frontière → quelques faux positifs marginaux acceptés
# ("complet" dans "incomplet"), mais le routeur LLM derrière filtre les cas
# ambigus. C'est l'équilibre robustesse/précision pour une heuristique.
_FORCE_WEB_TRIGGERS = [
    # ── Achat / réservation ──────────────────────────────
    r"billet",                          # billet, billets, billetterie
    r"réserv",                          # réservation, réserver, réservations
    r"acheter\s+(?:des\s+|une?\s+)?(?:place|billet)",   # acheter des places/billets
    r"où\s+(?:acheter|trouver)",        # où acheter, où trouver
    r"prix\s+(?:du|de\s+la|des)\s+(?:billet|place|entrée)",  # prix du billet
    # ── Disponibilité / état ─────────────────────────────
    r"encore disponible",
    r"compl[eè]t",                      # complet, complets, complète, complètes
    r"ouverture\s+(?:de\s+)?la\s+billetterie",
    # ── Actualité récente ────────────────────────────────
    r"annonc(?:é|ée|és|ées|er|ent|e)",  # annonce(s), annoncé(s), annoncer, annoncent
    r"actu(?:alité)?s?\s+récent",       # actu récente, actualités récentes
    r"derni(?:ère|er)s?\s+(?:actu|news|nouvelle)",  # dernière actu, derniers news
    r"prochain[se]?\s+(?:concert|spectacle|festival|événement|evenement)",
]
_FORCE_WEB_REGEX = re.compile("|".join(_FORCE_WEB_TRIGGERS), re.IGNORECASE)


# ── Fix D3-2 (mai 2026) — triggers « nom propre » sensibles à la casse ──────
# Capture les questions citant nommément un artiste/groupe/personnalité, que
# le catalogue OpenAgenda ne couvre pas (tournées nationales). Ex : "concert de
# Stromae", "Aya Nakamura à Paris".
#
# Pourquoi une regex SÉPARÉE et NON dans _FORCE_WEB_TRIGGERS : ces patterns
# reposent sur l'ancre majuscule [A-ZÀ-Ý] pour détecter un nom propre. Or
# _FORCE_WEB_REGEX est compilé avec re.IGNORECASE, qui ferait matcher cette
# classe sur des minuscules ("concert de jazz" → faux positif). On compile donc
# ici SANS IGNORECASE.
#
# Deux ajustements vs le brief D3-2 (assumés, conformes à la spec des tests) :
#   1. Mots événementiels rendus tolérants à la casse via [Cc]… : sans
#      IGNORECASE, "concert" minuscule ne matcherait pas "Spectacle"/"Festival"
#      en début de phrase (majuscule).
#   2. Pattern 2 ("Nom à Ville") doté d'un negative-lookahead excluant les noms
#      communs d'événements : sinon "Concert à Nantes" (mot commun capitalisé +
#      ville) matcherait à tort et casserait test_trigger_does_not_match_generic.
_FORCE_WEB_PROPERNOUN_TRIGGERS = [
    # "concert de Stromae", "Spectacle d'Aya Nakamura", "Tournée de Taylor Swift",
    # "Festival de Cannes" — terme événementiel + (de|du|d') + nom propre (Maj).
    r"(?:[Cc]oncert|[Ss]pectacle|[Tt]ourn[ée]e|[Vv]enue|[Ss]how|[Ff]estival)\s+(?:de|du|d['’])\s+[A-ZÀ-Ý]",
    # "Stromae à Paris", "Aya Nakamura en France" — nom propre + préposition de
    # lieu + ville couverte/France. Le lookahead exclut les noms communs
    # d'événements capitalisés (sinon "Concert à Nantes" matcherait).
    r"\b(?!(?:[Cc]oncert|[Ss]pectacle|[Ff]estival|[Ee]xpo|[Ee]xposition|[Tt]ourn[ée]e|[Ss]oir[ée]e|[Ee]v[ée]nement|[Rr]epr[ée]sentation)\b)"
    r"[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)?\s+(?:à|en|au|aux)\s+"
    r"(?:Paris|Lyon|Marseille|Toulouse|Nantes|Bordeaux|Lille|Strasbourg|France)",
]
_FORCE_WEB_PROPERNOUN_REGEX = re.compile("|".join(_FORCE_WEB_PROPERNOUN_TRIGGERS))


# Prompt court pour Mistral Small (~200 tokens entrée, ~10 sortie)
# v3 — fix D3-2 (mai 2026) : explicite les limites réelles du catalogue
# OpenAgenda (qui ne couvre PAS les tournées nationales d'artistes majeurs)
# et durcit la règle 1 (nom propre cité → web). Le prompt v2 biaisait trop
# vers RAG en présentant le catalogue comme exhaustif.
ROUTER_PROMPT_TEMPLATE = """Tu es un routeur pour un chatbot d'événements culturels couvrant 8 grandes villes françaises (Paris, Lyon, Marseille, Toulouse, Nantes, Bordeaux, Lille, Strasbourg).

Réponds UNIQUEMENT par "RAG" ou "WEB", rien d'autre.

CONTEXTE PRÉCIS DU CATALOGUE :
- Source unique : OpenAgenda (agendas culturels locaux publiés par les institutions et lieux des 8 villes).
- Couvre : événements LOCAUX publiés à l'avance par les structures culturelles (musées, salles de concert municipales, théâtres, festivals associatifs, médiathèques, etc.).
- NE COUVRE PAS : les tournées nationales d'artistes majeurs (chanteurs, humoristes, têtes d'affiche internationales), les dates annoncées dans la presse mais non publiées sur OpenAgenda, l'état des billetteries.

RÈGLES STRICTES — RÉPONDS "WEB" SI L'UNE DES CONDITIONS SUIVANTES EST VRAIE :
1. La question cite NOMMÉMENT un artiste, un groupe, une personnalité, ou un titre de tournée/spectacle commercial.
2. La question porte sur la disponibilité ou l'achat de billets, ou l'état d'une billetterie.
3. La question porte sur une actualité ou une annonce récente (< 1 semaine).
4. La question porte sur un événement dans une ville hors des 8 couvertes (Paris, Lyon, Marseille, Toulouse, Nantes, Bordeaux, Lille, Strasbourg).
5. La question ne précise aucune ville ET ne précise aucun thème, mais cite un nom propre.

SINON, RÉPONDS "RAG" :
- Recommandations par thème ou ambiance dans une des 8 villes (jazz, expo contemporaine, théâtre jeune public...)
- Questions sur "ce week-end", "cette semaine", "en juin" sans nom d'artiste cité
- Personnalisation selon profil
- Questions sur les lieux culturels locaux

EXEMPLES :
- "Concerts de jazz à Nantes ce week-end" → RAG
- "Expo d'art contemporain à Bordeaux" → RAG
- "Concert de Stromae prochainement" → WEB (règle 1 : nom propre)
- "Aya Nakamura en tournée" → WEB (règle 1)
- "Taylor Swift à Paris" → WEB (règle 1)
- "Spectacle de Florence Foresti à Lyon" → WEB (règle 1)
- "Billetterie du festival Hellfest" → WEB (règle 2)
- "Concert à Marseille en juillet" → RAG

Question : "{question}"

Réponse :"""


def route_to_rag_or_web(
    question: str,
    llm_router: Any,  # ChatMistralAI Small
    retrieval_score: Optional[float] = None,
    retrieval_count: Optional[int] = None,
) -> Tuple[str, str]:
    """Décide RAG ou WEB pour une question donnée.

    Trois mécanismes combinés (par ordre de priorité) :
        1. Triggers explicites regex (court-circuit, latence 0ms)
        2. Routeur LLM Mistral Small (~500ms)
        3. Fallback si retrieval pauvre (kept < 3 OU score < 0.6)

    Returns:
        ("rag" | "web", raison_courte_pour_log)
    """
    # 1. Triggers explicites (court-circuit performant)
    if _FORCE_WEB_REGEX.search(question):
        return "web", "trigger_keyword"
    # 1bis. [fix D3-2] Triggers « nom propre » (regex sensible à la casse).
    # Raison distincte pour distinguer les deux voies en analyse de logs.
    if _FORCE_WEB_PROPERNOUN_REGEX.search(question):
        return "web", "trigger_propernoun"

    # 2. Fallback retrieval pauvre (court-circuit aussi)
    if retrieval_count is not None and retrieval_count < 3:
        return "web", f"retrieval_count={retrieval_count}<3"
    if retrieval_score is not None and retrieval_score < 0.6:
        return "web", f"retrieval_score={retrieval_score:.2f}<0.6"

    # 3. Routeur LLM
    if llm_router is None:
        # Pas de LLM disponible → on reste sur RAG par défaut (sûr)
        return "rag", "no_llm_default"

    prompt = ROUTER_PROMPT_TEMPLATE.format(question=question)
    try:
        response = llm_router.invoke(prompt, config=_langfuse_config())
        decision = (response.content or "").strip().upper()
    except Exception as e:
        logger.warning(f"Routeur LLM error : {e} → fallback RAG")
        return "rag", "llm_error"

    if "WEB" in decision and "RAG" not in decision:
        return "web", "llm_router"
    # Tout autre cas (RAG, vide, mal formé) → RAG par sécurité
    return "rag", "llm_router"


# ============================================================================
# AGENT WEB ORCHESTRATEUR (smolagents-compatible)
# ============================================================================

@dataclass
class WebAgentResponse:
    """Réponse structurée de l'agent web (text + sources avec citations)."""
    text: str
    sources: List[WebResult] = field(default_factory=list)
    source_search: str = "brave"  # "brave" | "ddg" | "none"


class PulsWebAgent:
    """Orchestrateur — recherche + assemblage prompt + appel LLM.

    À ce stade du MVP, on n'utilise PAS le mécanisme d'agent autonome de
    smolagents (qui décide lui-même quels tools appeler en boucle). On
    fait un pipeline déterministe : search → filter → prompt → LLM.

    Pourquoi : démontrer le défi #3 sans subir l'imprévisibilité d'un
    agent autonome qui peut boucler ou halluciner. La couche d'abstraction
    PulsWebAgent permet de passer à un vrai agent smolagents en V2 sans
    changer le contrat avec app.py.

    Argument jury : "J'ai utilisé smolagents comme librairie sous-jacente
    (cohérent avec le mandat Jérémy) mais avec un pipeline contrôlé pour
    le MVP. Le passage à un agent vraiment autonome est tracé en US-606."
    """

    def __init__(
        self,
        llm_response: Any,         # ChatMistralAI Large pour la synthèse
        whitelist: DomainWhitelist,
        brave_client: Optional[BraveSearchClient] = None,
    ):
        self.llm = llm_response
        self.whitelist = whitelist
        self.brave_client = brave_client or BraveSearchClient()

    def run(self, question: str) -> WebAgentResponse:
        """Exécute la recherche web et synthétise une réponse citée."""
        # 1. Recherche filtrée
        sources = web_search_filtered(
            question,
            whitelist=self.whitelist,
            brave_client=self.brave_client,
        )

        if not sources:
            return WebAgentResponse(
                text=(
                    "Je n'ai pas trouvé de source fiable dans mes domaines de "
                    "confiance (presse locale, billetteries officielles, lieux "
                    "culturels nantais) pour répondre à cette question. "
                    "Je peux essayer de répondre sur la base du catalogue "
                    "OpenAgenda si tu reformules ta question."
                ),
                sources=[],
                source_search="none",
            )

        # 2. Assemblage du contexte avec instruction de citation
        context_blocks = []
        for i, s in enumerate(sources, 1):
            context_blocks.append(
                f"[Source {i} — {s.source_icon} {s.source_category}]\n"
                f"Titre : {s.title}\n"
                f"URL : {s.url}\n"
                f"Extrait : {s.snippet}\n"
            )
        context = "\n---\n".join(context_blocks)

        prompt = (
            "Tu es Puls, assistant culturel pour Nantes Métropole.\n\n"
            "Tu as fait une recherche web sur les sources de confiance suivantes. "
            "Réponds à la question de l'utilisateur EN T'APPUYANT EXCLUSIVEMENT sur "
            "ces sources. Pour chaque affirmation factuelle, indique entre crochets "
            "la source utilisée, par exemple : [Source 1].\n\n"
            "Si les sources ne suffisent pas à répondre, dis-le clairement plutôt "
            "que d'inventer.\n\n"
            f"SOURCES :\n{context}\n\n"
            f"QUESTION : {question}\n\n"
            "RÉPONSE (avec citations [Source N]) :"
        )

        try:
            llm_resp = self.llm.invoke(prompt, config=_langfuse_config())
            text = llm_resp.content
        except Exception as e:
            logger.error(f"Agent web LLM error : {e}")
            text = (
                "Une erreur est survenue lors de la synthèse de la recherche web. "
                "Réessaie ta question dans un instant."
            )

        return WebAgentResponse(
            text=text,
            sources=sources,
            source_search="brave" if self.brave_client.is_available() else "ddg",
        )

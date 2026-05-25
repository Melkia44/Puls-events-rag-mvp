"""
app.py
Puls-Events MVP P13 — Interface Gradio.

Vague 3 : RAG + D1 mémoire + D2 géo + D3 agent web
    - D1 — Mémoire conversationnelle (court + long terme Supabase)
    - D2 — Contexte géographique (Nominatim + Haversine)
    - D3 — Recherche web temps réel (smolagents + Brave + whitelist)

Pas encore dans cette version :
    - D4 monitoring Langfuse (vague 4)

Compatible : Gradio 4.x / 5.x (type="messages" depuis 4.36).

────────────────────────────────────────────────────────────────────────────
Changelog UI :
    P0.1 — Sidebar technique masquée derrière SHOW_DEBUG (12-factor)
    P0.2 — Titre produit "Puls · Événements Nantes Métropole"
    P1.1 — Message d'accueil pré-chargé + 4 chips de suggestion
    P1.2 — Panneau Sources replié sous chaque réponse (traçabilité RAG)
    P2   — Footer dynamique exposant l'état D1
    P3   — Mock d'authentification explicité (cf. R-AUTH-01)
    P4   — D2 géo : champ ville sidebar + badge filtre + distance par source
    P5   — D3 agent web : routage RAG/web + badge "Recherche web" + sources
           web filtrées par whitelist de 12 domaines de confiance
    P6   — TOP-8 multi-villes : titre/chips/footer adaptés au pivot national.
           Détection multi-villes (>=2 villes citées OU mot comparatif)
           désactive le filtre géo et élargit top-K à 10 pour Mistral
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

# ============================================================================
# CHARGEMENT .env EN PREMIER
# ============================================================================
import os
from pathlib import Path
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, verbose=True)

if not os.getenv("MISTRAL_API_KEY"):
    print(f"⚠️  .env non chargé ou MISTRAL_API_KEY manquant. Chemin tenté : {_ENV_PATH}")

# ============================================================================
# Imports==============
import os
from pathlib import Path
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, verbose=True)

if not os.getenv("MISTRAL_API_KEY"):
    print(f"⚠️  .env non chargé ou MISTRAL_API_KEY manquant. Chemin tenté : {_ENV_PATH}")

# ============================================================================
# Imports
# ============================================================================
import logging
import re
from typing import List, Tuple, Dict, Any, Optional

import gradio as gr
from langchain_mistralai import ChatMistralAI

# === Langfuse — Observabilité LLM (D4) ===
# Import défensif : si Langfuse est absent/incompatible, on bascule sur des
# stubs no-op pour que l'app tourne sans observabilité (dégradation gracieuse).
# API v3/v4 : `observe` + `get_client` viennent du package racine `langfuse`
# (le module `langfuse.decorators` de la v2 n'existe plus).
try:
    from langfuse import observe, get_client
    from langfuse.langchain import CallbackHandler
    LANGFUSE_OK = True
except ImportError:
    LANGFUSE_OK = False

    def observe(*args, **kwargs):
        """Stub no-op : supporte @observe et @observe(name=...)."""
        if args and callable(args[0]):
            return args[0]

        def _decorator(fn):
            return fn

        return _decorator

    def get_client():
        return None

    CallbackHandler = None

from utils.config import (
    CHAT_MODEL, EXTRACTION_MODEL, EMBEDDING_MODEL,
    FAISS_INDEX_PATH, RETRIEVER_K, RETRIEVER_K_GEO, MEMORY_WINDOW_SIZE,
    MISTRAL_API_KEY, DATABASE_URL,
    validate_config,
)
from utils.vector_store import load_vector_store
from utils.prompts import build_rag_prompt
from utils.memory import (
    ShortTermMemory, LongTermMemory,
    extract_preferences, persist_extracted_preferences,
)
from utils.geo import (
    geocode_city, filter_by_radius,
    extract_radius_override, extract_location_override,
    DEFAULT_RADIUS_KM,
)
# [D3] Agent web — recherche temps réel
from utils.web_agent import (
    DomainWhitelist, BraveSearchClient, PulsWebAgent,
    route_to_rag_or_web,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

logger.info(f"Gradio version : {gr.__version__}")


# ============================================================================
# STARTUP — singletons
# ============================================================================

config_errors = validate_config()
if config_errors:
    logger.error("Erreurs de configuration au démarrage :")
    for err in config_errors:
        logger.error(f"  - {err}")

logger.info("Chargement du vector store FAISS…")
try:
    VECTOR_STORE = load_vector_store(
        index_path=FAISS_INDEX_PATH,
        embedding_model=EMBEDDING_MODEL,
        api_key=MISTRAL_API_KEY,
    )
    logger.info("Vector store prêt")
except Exception as e:
    logger.error(f"Échec chargement vector store : {e}")
    VECTOR_STORE = None

logger.info("Connexion à Supabase Postgres…")
try:
    LTM = LongTermMemory(database_url=DATABASE_URL)
    logger.info("Mémoire long terme prête")
except Exception as e:
    logger.error(f"Échec connexion Supabase : {e}")
    LTM = None

logger.info("Initialisation LLM Mistral Large (réponses)…")
LLM = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model=CHAT_MODEL,
    temperature=0.3,
) if MISTRAL_API_KEY else None

# [D3] LLM léger pour le routeur RAG vs Web — Mistral Small
# Quasi gratuit (~0.2€/1M tokens), latence ~500ms, suffisant pour binaire
logger.info("Initialisation LLM Mistral Small (routeur D3)…")
LLM_ROUTER = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model=EXTRACTION_MODEL,  # mistral-small-latest
    temperature=0.0,         # déterministe pour le routage
) if MISTRAL_API_KEY else None

# === [D4] Langfuse — traçage automatique des appels LLM ===
# Le handler intercepte les .invoke() LangChain auxquels on passe
# config={"callbacks": [LANGFUSE_HANDLER]} : prompt + completion + tokens
# + latence + coût € remontent au dashboard Langfuse. Dégradation
# gracieuse (None) si clés absentes/invalides — l'observabilité D4 ne
# doit jamais faire tomber l'app.
if not LANGFUSE_OK:
    logger.warning(
        "Langfuse non importable — observabilité D4 ET capture trace_id "
        "désactivées (le vote restera persisté sans langfuse_trace_id)."
    )
    LANGFUSE_HANDLER = None
else:
    try:
        LANGFUSE_HANDLER = CallbackHandler()
        logger.info(
            f"Langfuse callback initialisé "
            f"(host={os.getenv('LANGFUSE_HOST', 'défaut SDK')})"
        )
    except Exception as e:
        logger.error(f"Échec init Langfuse : {e} — observabilité D4 désactivée")
        LANGFUSE_HANDLER = None


# [D3] Whitelist de domaines + agent web
WHITELIST_PATH = Path(__file__).resolve().parent / "config" / "domain_whitelist.yaml"
logger.info(f"Chargement whitelist depuis {WHITELIST_PATH}…")
try:
    WHITELIST = DomainWhitelist(WHITELIST_PATH)
    logger.info(f"Whitelist : {WHITELIST.count()} domaines de confiance")
except Exception as e:
    logger.error(f"Échec chargement whitelist : {e}")
    WHITELIST = None

logger.info("Initialisation agent web D3…")
try:
    BRAVE_CLIENT = BraveSearchClient()  # lit BRAVE_API_KEY depuis env
    WEB_AGENT = PulsWebAgent(
        llm_response=LLM,
        whitelist=WHITELIST,
        brave_client=BRAVE_CLIENT,
    ) if (LLM and WHITELIST) else None
    if WEB_AGENT:
        logger.info(
            f"Agent web prêt (Brave={'OUI' if BRAVE_CLIENT.is_available() else 'NON, fallback DDG seul'})"
        )
    else:
        logger.warning("Agent web indisponible : LLM ou whitelist manquant")
except Exception as e:
    logger.error(f"Échec init agent web : {e}")
    WEB_AGENT = None


SHOW_DEBUG = os.getenv("SHOW_DEBUG", "0") == "1"
logger.info(f"Mode debug UI : {'activé' if SHOW_DEBUG else 'désactivé'}")

# [D3 v2] RETRIEVER_K_GEO est désormais centralisé dans utils/config.py
# (source unique de vérité, lue aussi par les scripts de mesure).


# ============================================================================
# HELPERS FORMATAGE
# ============================================================================

def _format_iso_date(iso_str: str) -> str:
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        mois = ["", "janvier", "février", "mars", "avril", "mai", "juin",
                "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
        date_part = f"{dt.day} {mois[dt.month]} {dt.year}"
        if dt.hour or dt.minute:
            date_part += f" à {dt.hour}h{dt.minute:02d}"
        return date_part
    except Exception:
        return iso_str


def _same_date(iso1: str, iso2: str) -> bool:
    try:
        return iso1[:10] == iso2[:10]
    except Exception:
        return False


def _format_event_with_metadata(doc) -> str:
    """Formate un Document LangChain en bloc structuré pour le prompt RAG."""
    meta = doc.metadata or {}
    lines = []

    if meta.get("title"):
        lines.append(f"Titre : {meta['title']}")
    if meta.get("location"):
        lines.append(f"Lieu : {meta['location']}")

    start = meta.get("start_date")
    end = meta.get("end_date")
    if start:
        start_fmt = _format_iso_date(start)
        if end and end != start and not _same_date(start, end):
            end_fmt = _format_iso_date(end)
            lines.append(f"Du {start_fmt} au {end_fmt}")
        else:
            lines.append(f"Date : {start_fmt}")

    distance = meta.get("distance_km")
    if distance is not None:
        lines.append(f"Distance : {distance} km de l'utilisateur")

    content = (doc.page_content or "").strip()
    if content:
        lines.append(f"Description : {content}")

    if meta.get("url"):
        lines.append(f"Lien : {meta['url']}")

    return "\n".join(lines)


def _format_sources_html(docs, geo_info: Optional[Dict] = None) -> str:
    """Bloc Sources RAG (D2) — replié, avec distances si filtre géo actif."""
    if not docs and not geo_info:
        return ""

    geo_badge = ""
    if geo_info:
        city = geo_info.get("city", "?")
        radius = geo_info.get("radius_km", DEFAULT_RADIUS_KM)
        if geo_info.get("fallback"):
            geo_badge = (
                f"\n\n<div style='font-size:0.82em;opacity:0.75;"
                f"padding:0.4em 0.7em;margin:0.4em 0;border-left:2px solid #f59e0b;"
                f"background:rgba(245,158,11,0.05);border-radius:3px;'>"
                f"📍 Filtre géo élargi — peu de résultats dans le rayon "
                f"<b>{radius} km</b> autour de <b>{city}</b>"
                f"</div>"
            )
        else:
            geo_badge = (
                f"\n\n<div style='font-size:0.82em;opacity:0.75;"
                f"padding:0.4em 0.7em;margin:0.4em 0;border-left:2px solid #10b981;"
                f"background:rgba(16,185,129,0.05);border-radius:3px;'>"
                f"📍 Filtré à <b>{radius} km</b> autour de <b>{city}</b>"
                f"</div>"
            )

    if not docs:
        return geo_badge

    items = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        title = meta.get("title", f"Source {i}")
        location = meta.get("location", "")
        url = meta.get("url", "")
        distance = meta.get("distance_km")
        url_html = f' · <a href="{url}" target="_blank">Voir sur OpenAgenda</a>' if url else ""
        loc_html = f" — <i>{location}</i>" if location else ""
        dist_html = f" · <b>{distance} km</b>" if distance is not None else ""
        items.append(f"<li><b>{title}</b>{loc_html}{dist_html}{url_html}</li>")

    return (
        geo_badge
        + "\n\n<details style='margin-top:0.5em;font-size:0.85em;opacity:0.85;'>"
        + f"<summary>📚 {len(docs)} sources consultées</summary>"
        + f"<ul style='margin-top:0.5em;'>{''.join(items)}</ul>"
        + "</details>"
    )


def _format_web_block(web_response, route_reason: str) -> str:
    """[D3] Bloc HTML pour les réponses agent web — badge + sources web.

    Affiche :
        1. Un badge violet en tête : "🌐 Recherche web active"
        2. Un panneau dépliable avec les sources externes citées
           (icônes par catégorie : 📰 presse, 🎫 billetterie, 🏛️ inst., 🎭 lieu)
    """
    # Badge violet pour distinguer du badge vert/orange D2
    badge = (
        f"\n\n<div style='font-size:0.82em;opacity:0.85;"
        f"padding:0.4em 0.7em;margin:0.4em 0;border-left:2px solid #8b5cf6;"
        f"background:rgba(139,92,246,0.05);border-radius:3px;'>"
        f"🌐 <b>Recherche web temps réel</b> — sources hors catalogue OpenAgenda "
        f"<small style='opacity:0.7;'>({route_reason})</small>"
        f"</div>"
    )

    if not web_response.sources:
        return badge

    items = []
    for s in web_response.sources:
        items.append(
            f"<li>{s.source_icon} <b>{s.title}</b> "
            f"<small style='opacity:0.7;'>({s.source_category})</small> · "
            f'<a href="{s.url}" target="_blank">{s.url}</a></li>'
        )

    panel = (
        f"\n\n<details style='margin-top:0.5em;font-size:0.85em;opacity:0.85;'>"
        f"<summary>🔗 {len(web_response.sources)} sources web citées "
        f"<small style='opacity:0.6;'>(via {web_response.source_search})</small></summary>"
        f"<ul style='margin-top:0.5em;'>{''.join(items)}</ul>"
        f"</details>"
    )

    return badge + panel


def _run_web_agent(message: str, route_reason: str) -> Tuple[str, str]:
    """Exécute l'agent web et formate la réponse.

    Centralise les 3 points de bascule Web de respond() (trigger explicite,
    requête géo sans résultat, routeur LLM) pour qu'une évolution du
    formatage web n'ait qu'un seul site à modifier.

    Returns:
        (response_clean, response_with_extras)
    """
    logger.info(f"D3 routage : web ({route_reason})")
    web_resp = WEB_AGENT.run(message)
    return web_resp.text, web_resp.text + _format_web_block(web_resp, route_reason)


# ============================================================================
# LOGIQUE MÉTIER — RAG + GEO (inchangée)
# ============================================================================

# [TOP-8 — patch β] Détection de questions multi-villes pour bypass du filtre géo
#
# Pourquoi ce mécanisme :
#     L'utilisateur peut demander "Compare les festivals à Marseille et Bordeaux".
#     Si on garde le filtre géo (sur la ville du profil = Paris par exemple),
#     ni Marseille ni Bordeaux ne passent (350+ km), tous les docs sont
#     filtrés, fallback orange déclenché, et Mistral Large n'a plus de
#     contexte pertinent à comparer. Mauvaise réponse garantie.
#
# Solution :
#     Si la question mentionne explicitement >=2 villes du TOP-8, OU si elle
#     contient un mot-clé de comparaison ("compare", "vs", etc.) associé à
#     au moins une ville, on désactive le filtre géo. Le retrieval sémantique
#     sur top-K élargi (=10) donne assez de matière à Mistral Large pour
#     structurer la réponse par ville.
#
# Anti-pattern évité :
#     Pas de logique multi-villes spéciale dans le pipeline. On désactive le
#     filtre, point. C'est Mistral Large qui structure la réponse selon le
#     prompt utilisateur (pas de code-métier de comparaison).
#
# Réf : Test TOP8-3 dans le plan de re-test, bug identifié 2026-05-13.

# Villes connues du corpus TOP-8 (lowercase pour matching)
# Note : dupliqué volontairement de scripts/cities.py pour éviter
#        un import cross-package — la liste est petite et stable.
_COVERED_CITIES_LOWER = frozenset({
    "paris", "lyon", "marseille", "toulouse",
    "nantes", "bordeaux", "lille", "strasbourg",
})

# Mots-clés de comparaison explicite
_COMPARISON_KEYWORDS = re.compile(
    r"(compare|comparer|comparaison|versus|\bvs\b|différence entre|meilleur entre)",
    re.IGNORECASE,
)


def _detect_multi_city_query(message: str) -> tuple[bool, list[str]]:
    """Détecte si une question concerne explicitement plusieurs villes.

    Returns:
        (True, [villes]) si multi-villes détecté, (False, []) sinon.
    """
    msg_low = message.lower()

    # Trouver toutes les villes du TOP-8 mentionnées dans le message
    mentioned = [c.capitalize() for c in _COVERED_CITIES_LOWER if c in msg_low]

    # Cas 1 : >= 2 villes citées → multi-villes
    if len(mentioned) >= 2:
        return (True, mentioned)

    # Cas 2 : 1 ville + mot-clé de comparaison → multi-villes (implicite)
    if mentioned and _COMPARISON_KEYWORDS.search(message):
        return (True, mentioned)

    return (False, [])


def _resolve_geo_target(
    user_message: str, user_id: Optional[int],
) -> Optional[Dict[str, Any]]:
    # [TOP-8 — patch β] Bypass du filtre géo si question multi-villes
    is_multi, cities = _detect_multi_city_query(user_message)
    if is_multi:
        logger.info(
            f"D2 désactivé : question multi-villes détectée "
            f"(villes mentionnées : {cities})"
        )
        return None

    radius = extract_radius_override(user_message) or DEFAULT_RADIUS_KM

    location_override = extract_location_override(user_message)
    if location_override:
        coords = geocode_city(location_override)
        if coords is not None:
            return {
                "city": location_override,
                "lat": coords[0], "lng": coords[1],
                "radius_km": radius, "source": "override",
            }
        else:
            logger.info(f"Override géo '{location_override}' non géocodable, ignoré")

    if user_id is not None and LTM is not None:
        profile_city = LTM.get_user_city(user_id)
        if profile_city and profile_city["lat"] is not None:
            return {
                "city": profile_city["city"],
                "lat": profile_city["lat"], "lng": profile_city["lng"],
                "radius_km": radius, "source": "profile",
            }

    return None


def rag_response(
    user_message: str,
    short_term: ShortTermMemory,
    user_id: Optional[int],
) -> Tuple[str, List, Optional[Dict], Optional[float]]:
    """Pipeline RAG : retrieval → [filtre géo] → prompt → Mistral.

    Retourne (texte, docs_finaux, geo_info, top_l2).
    top_l2 = distance L2 du meilleur document sur le chemin non-géo (None
    sinon) — exposée pour LOG/calibration du fallback routeur (fix D3-2).
    """
    if VECTOR_STORE is None or LLM is None:
        return ("⚠️ Erreur de configuration.", [], None)

    geo_target = _resolve_geo_target(user_message, user_id)
    top_l2: Optional[float] = None  # [fix D3-2] distance L2 top-1 (chemin non-géo)

    if geo_target is not None:
        candidates = VECTOR_STORE.similarity_search(user_message, k=RETRIEVER_K_GEO)
        filtered, fallback, strict_in_radius = filter_by_radius(
            candidates,
            user_lat=geo_target["lat"], user_lng=geo_target["lng"],
            radius_km=geo_target["radius_km"],
        )
        docs = filtered[:RETRIEVER_K]
        # [D3 v2] geo_info expose deux compteurs distincts :
        #   - kept            : docs réellement passés au prompt (post-fallback)
        #   - strict_in_radius : docs STRICTEMENT dans le rayon (pré-fallback)
        # Cas A (bascule Web) se déclenche sur strict_in_radius == 0, ce que
        # kept ne pourrait jamais valoir (fallback non-filtré le remplit).
        geo_info = {
            **geo_target,
            "fallback": fallback,
            "kept": len(docs),
            "strict_in_radius": strict_in_radius,
        }
        logger.info(
            f"D2 actif : filtre {geo_target['radius_km']}km autour de "
            f"{geo_target['city']} (source={geo_target['source']}, "
            f"fallback={fallback}, kept={len(docs)}, strict={strict_in_radius})"
        )
    else:
        # [TOP-8 — patch β] Si désactivation due au multi-villes,
        # on élargit le top-K pour donner plus de matière à Mistral
        # (sinon 5 docs ne suffisent pas à structurer une comparaison)
        is_multi, _ = _detect_multi_city_query(user_message)
        k = RETRIEVER_K * 2 if is_multi else RETRIEVER_K  # 10 vs 5
        # [fix D3-2] Variante _with_score sur le chemin NON-géo (= celui qui
        # passe par le routeur LLM, Cas B de respond()). On récupère la
        # distance L2 du top-1 pour la LOGUER et calibrer empiriquement le
        # seuil de fallback. Comportement docs inchangé (on extrait le [0]).
        scored = VECTOR_STORE.similarity_search_with_score(user_message, k=k)
        docs = [d for d, _ in scored]
        top_l2 = scored[0][1] if scored else None
        geo_info = None
        if is_multi:
            logger.info(f"Multi-villes : top-K élargi à {k} (sans filtre géo)")

    context = "\n\n---\n\n".join(_format_event_with_metadata(d) for d in docs)

    profile_block = ""
    if user_id and LTM is not None:
        profile_block = LTM.get_preference_summary(user_id)

    history_block = short_term.to_prompt_block()

    prompt = build_rag_prompt(
        question=user_message,
        context=context,
        history=history_block,
        profile=profile_block,
    )

    # [D4] Callback Langfuse pour traçage automatique (prompt, tokens, latence, coût)
    response = LLM.invoke(
        prompt,
        config={"callbacks": [LANGFUSE_HANDLER]} if LANGFUSE_HANDLER else {},
    )
    return (response.content, docs, geo_info, top_l2)


def trigger_preference_extraction(short_term: ShortTermMemory, user_id: int, session_id: int) -> int:
    if LTM is None or not MISTRAL_API_KEY:
        return 0

    history = short_term.get_history()
    if not history:
        return 0

    messages = []
    for turn in history:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})

    extracted = extract_preferences(
        messages=messages,
        api_key=MISTRAL_API_KEY,
        model=EXTRACTION_MODEL,
    )

    if extracted:
        count = persist_extracted_preferences(
            user_id=user_id,
            extracted=extracted,
            long_term_memory=LTM,
            session_id=session_id,
        )
        logger.info(f"Préférences extraites : {count} pour user_id={user_id}")
        return count
    return 0


# ============================================================================
# UI GRADIO — Handlers
# ============================================================================

def get_user_list() -> List[str]:
    if LTM is None:
        return []
    users = LTM.list_users()
    return [u["name"] for u in users]


def _format_sessions_for_sidebar(sessions: list) -> list:
    """[SIDEBAR] Convertit list_user_sessions(...) en choices gr.Radio.

    Format Gradio Radio : liste de tuples (label_affiché, value_id).
    Label : "preview tronqué — DD/MM HH:MM"
    """
    if not sessions:
        return []
    formatted = []
    for s in sessions:
        preview = (s.get("preview") or "(sans titre)").strip()
        if len(preview) > 45:
            preview = preview[:42] + "..."
        ts = s.get("last_message_at")
        when = ts.strftime("%d/%m %H:%M") if ts else "?"
        label = f"{preview}  ·  {when}"
        formatted.append((label, s["session_id"]))
    return formatted


def _rebuild_chat_history(messages: list) -> List[Dict[str, str]]:
    """[SIDEBAR] Convertit load_session_history(...) en format Gradio Chatbot.

    Format Gradio messages : [{"role": "user|assistant", "content": str}, ...]
    """
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]


def select_user(name: str, city: str) -> Tuple[Dict, Dict, str, str]:
    if not name or not name.strip():
        return {}, {}, "Aucun utilisateur sélectionné", ""

    if LTM is None:
        return {}, {}, "⚠️ Supabase non connecté", city

    name = name.strip()
    user_id = LTM.get_or_create_user(name)
    session_id = LTM.start_session(user_id)
    profile_summary = LTM.get_preference_summary(user_id)

    city_to_display = city.strip() if city else ""
    if city_to_display:
        coords = geocode_city(city_to_display)
        if coords is not None:
            LTM.set_user_city(user_id, city_to_display, coords[0], coords[1])
            logger.info(f"Profil '{name}' : ville '{city_to_display}' géocodée et persistée")
        else:
            # [fix D2] Dégradation gracieuse : on persiste le NOM de ville même
            # sans coords (lat/lng=NULL), plutôt que de sauter l'UPDATE. Évite le
            # bug « ville figée » quand Nominatim échoue (403 prod). Le filtre géo
            # reste inactif tant qu'on n'a pas de coordonnées.
            LTM.set_user_city(user_id, city_to_display, None, None)
            logger.warning(
                f"Ville {city_to_display} persistée sans coordonnées — "
                f"filtre géographique inactif"
            )
    else:
        stored = LTM.get_user_city(user_id)
        if stored:
            city_to_display = stored["city"]

    user_state = {"id": user_id, "name": name}
    session_state = {"id": session_id}

    profile_display = (
        profile_summary
        if profile_summary
        else f"_Profil vierge pour **{name}** — sera enrichi à mesure des conversations._"
    )

    return user_state, session_state, profile_display, city_to_display


@observe(name="puls_events_respond")
def respond(
    message: str,
    chat_history: List[Dict],
    short_term: ShortTermMemory,
    user_state: Dict,
    session_state: Dict,
    feedback_meta: Dict[int, Dict],
) -> Tuple[str, List[Dict], ShortTermMemory, Dict[int, Dict]]:
    if not message or not message.strip():
        return "", chat_history, short_term, feedback_meta

    if not user_state or "id" not in user_state:
        warning = "⚠️ Active d'abord un profil de démonstration dans la barre latérale."
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": warning},
        ]
        return "", chat_history, short_term, feedback_meta

    user_id = user_state["id"]
    session_id = session_state.get("id") if session_state else None

    # [D3 v2] Routage RAG-first : on tente toujours D1+D2 d'abord,
    # puis on fallback Web si retrieval pauvre OU trigger explicite.
    # Justification : éviter que le routeur LLM court-circuite D2 sur
    # des requêtes géographiques légitimes ("concerts à 15 km de Nantes").

    response_with_extras = ""
    response_clean = ""
    decision = "rag"
    route_reason = "rag_first_default"

    try:
        # Étape 1 — Triggers explicites prioritaires (billets, complet, etc.)
        # → court-circuit direct vers Web, on ne tente même pas le RAG
        from utils.web_agent import _FORCE_WEB_REGEX

        if _FORCE_WEB_REGEX.search(message) and WEB_AGENT is not None:
            decision, route_reason = "web", "trigger_keyword"
            response_clean, response_with_extras = _run_web_agent(message, route_reason)
        else:
            # Étape 2 — Pipeline RAG complet (D1 + D2)
            response_clean, sources, geo_info, top_l2 = rag_response(message, short_term, user_id)

            # Étape 3 — Décider a posteriori si fallback Web nécessaire
            kept_count = (geo_info or {}).get("kept", len(sources) if sources else 0)
            is_geo_query = bool(geo_info)
            strict_count = (geo_info or {}).get("strict_in_radius", -1)
            target_city = (geo_info or {}).get("city", "?")

            # Cas A : requête géo SANS aucun événement réel dans le rayon
            # (strict_in_radius == 0 — le fallback Haversine a rempli kept
            # avec des docs hors zone). Ex : Saint-Nazaire hors corpus top-8
            # → on bascule Web plutôt que de mentir avec des events Nantes.
            if is_geo_query and strict_count == 0 and WEB_AGENT is not None:
                decision, route_reason = "web", f"d2_strict_in_radius=0_city={target_city}"
                response_clean, response_with_extras = _run_web_agent(message, route_reason)

            # Cas B : requête NON-géo → on consulte systématiquement le
            # routeur LLM (peu importe kept_count). Le routeur tranche
            # RAG vs Web selon la nature de la question (actu, billetterie…).
            elif not is_geo_query and LLM_ROUTER is not None:
                # [fix D3-2] On LOGUE la distance L2 top-1 pour calibrer
                # empiriquement le seuil de fallback, mais on ne la passe PAS
                # encore à route_to_rag_or_web (formule de normalisation à
                # valider sur cet index — risque de forcer web sur tout).
                if top_l2 is not None:
                    logger.info(f"D3 retrieval top_l2={top_l2:.4f}")
                llm_decision, llm_reason = route_to_rag_or_web(
                    message,
                    llm_router=LLM_ROUTER,
                    retrieval_count=kept_count,
                )
                if llm_decision == "web" and WEB_AGENT is not None:
                    decision, route_reason = "web", f"llm_router_non_geo (kept={kept_count})"
                    response_clean, response_with_extras = _run_web_agent(message, route_reason)
                else:
                    # Le LLM confirme RAG → on garde la réponse RAG
                    logger.info(f"D3 routage : rag (llm_router_non_geo, kept={kept_count})")
                    response_with_extras = response_clean + _format_sources_html(sources, geo_info)

            # Cas C : requête géo avec résultats réels → RAG nominal (D2)
            else:
                logger.info(
                    f"D3 routage : rag (kept={kept_count}, strict={strict_count}, "
                    f"geo={is_geo_query})"
                )
                response_with_extras = response_clean + _format_sources_html(sources, geo_info)

    except Exception as e:
        logger.error(f"Erreur respond : {e}")
        response_clean = f"⚠️ Erreur lors de la génération : {str(e)[:200]}"
        response_with_extras = response_clean

    # Mémoire courte = version sans HTML (pas de pollution du contexte)
    short_term.add_turn(message, response_clean)

    # [D4] message_id de la réponse assistant — clé d'attribution du vote
    assistant_msg_id = None
    if LTM is not None and session_id:
        try:
            LTM.log_message(session_id, "user", message)
            assistant_msg_id = LTM.log_message(session_id, "assistant", response_clean)
        except Exception as e:
            logger.warning(f"Échec log message : {e}")

    # [D4] trace_id Langfuse de cet échange (None si obs. désactivée/hors contexte)
    trace_id = None
    if LANGFUSE_OK:
        try:
            client = get_client()
            trace_id = client.get_current_trace_id() if client else None
        except Exception as exc:
            logger.warning("Langfuse trace_id indisponible : %s", exc)

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_with_extras},
    ]

    # [D4] On indexe les métadonnées d'attribution par la position du message
    # assistant dans le chatbot (gr.LikeData.index renverra cette position).
    # response_snapshot = version SANS HTML (response_clean), pour l'analyse.
    feedback_meta = dict(feedback_meta or {})
    feedback_meta[len(chat_history) - 1] = {
        "message_id": assistant_msg_id,
        "trace_id": trace_id,
        "question": message,
        "response": response_clean,
    }
    return "", chat_history, short_term, feedback_meta


WELCOME_MESSAGE = [{
    "role": "assistant",
    "content": (
        "Bonjour ! Je suis **Puls**, ton assistant culturel pour 8 grandes "
        "villes françaises (Paris, Lyon, Marseille, Toulouse, Nantes, Bordeaux, "
        "Lille, Strasbourg).\n\n"
        "Je peux t'aider à découvrir concerts, expos, spectacles et festivals "
        "près de chez toi. Pour les questions d'actualité (billetterie, "
        "annonces récentes), je peux aussi consulter la presse locale et "
        "les billetteries officielles.\n\n"
        "_Active un profil de démonstration, puis pose-moi ta question — "
        "ou clique sur une suggestion ci-dessous._ ↓"
    ),
}]


def new_conversation(
    short_term: ShortTermMemory,
    user_state: Dict,
    session_state: Dict,
) -> Tuple[List[Dict], ShortTermMemory, Dict, str, Dict]:
    # [D4] Le 5e retour ({}) réinitialise feedback_meta : le chatbot est
    # reconstruit, les anciens index de message ne sont plus valides.
    if not user_state or "id" not in user_state:
        return list(WELCOME_MESSAGE), ShortTermMemory(window_size=MEMORY_WINDOW_SIZE), {}, "Pas d'utilisateur actif.", {}

    user_id = user_state["id"]
    old_session_id = session_state.get("id") if session_state else None

    n_extracted = 0
    if old_session_id and short_term and len(short_term) > 0:
        n_extracted = trigger_preference_extraction(short_term, user_id, old_session_id)
        if LTM is not None:
            LTM.end_session(old_session_id)

    new_session_id = LTM.start_session(user_id) if LTM is not None else None
    new_state = {"id": new_session_id} if new_session_id else {}

    profile_summary = LTM.get_preference_summary(user_id) if LTM is not None else ""
    profile_display = profile_summary if profile_summary else "_Profil en cours de construction…_"

    info = "✅ Nouvelle conversation démarrée."
    if n_extracted > 0:
        info += f" {n_extracted} préférence(s) extraite(s) de la précédente session."

    return list(WELCOME_MESSAGE), ShortTermMemory(window_size=MEMORY_WINDOW_SIZE), new_state, profile_display + f"\n\n{info}", {}


def _status_line(user_state: Dict, city: str = "") -> str:
    """[D2+D3] Footer avec profil, ville, et état D3 (agent web actif/inactif)."""
    name = user_state.get("name", "—") if user_state else "—"
    city_label = city.strip() if city else "—"
    web_status = "🌐 Web actif" if WEB_AGENT else "🌐 Web inactif"
    return (
        f"<div style='text-align:center;font-size:0.78em;opacity:0.6;"
        f"padding:0.5em 0;border-top:1px solid rgba(255,255,255,0.05);"
        f"margin-top:1em;'>"
        f"Profil : <b>{name}</b> · "
        f"Ville : <b>{city_label}</b> · "
        f"Mémoire conversationnelle active · "
        f"{web_status} · "
        f"Données OpenAgenda · 8 métropoles françaises"
        f"</div>"
    )


# ============================================================================
# [D4] MONITORING CSAT — dashboard satisfaction utilisateur
# ============================================================================

def _format_csat_card(stats: Dict[str, Any], stats_7j: Dict[str, Any]) -> str:
    """Carte Markdown du CSAT global, colorisée selon le score.

    Code couleur métier : 🟢 >=75 %, 🟠 50–75 %, 🔴 <50 %. Affiche aussi la
    tendance sur 7 jours glissants (↗/↘/→) et la date du dernier vote.
    """
    csat = stats.get("csat")
    if csat is None:
        return "### 📊 CSAT — _Aucun vote pour le moment_"

    if csat >= 0.75:
        emoji, statut = "🟢", "Bon"
    elif csat >= 0.5:
        emoji, statut = "🟠", "À surveiller"
    else:
        emoji, statut = "🔴", "Alerte"

    trend = ""
    csat_7j = stats_7j.get("csat")
    if csat_7j is not None:
        delta = csat_7j - csat
        if abs(delta) < 0.05:
            arrow = "→"
        else:
            arrow = "↗" if delta > 0 else "↘"
        trend = f" — Tendance 7j : **{csat_7j:.0%}** {arrow}"

    last = stats.get("last_vote")
    last_str = last.strftime("%d/%m/%Y %H:%M") if last else "—"

    return (
        f"## CSAT global : **{csat:.0%}** {emoji} _{statut}_{trend}\n\n"
        f"- 👍 {stats['thumbs_up']} · 👎 {stats['thumbs_down']} "
        f"· **{stats['total']} votes** au total\n"
        f"- Dernier vote : {last_str}"
    )


def refresh_csat_dashboard() -> Tuple[str, List[List]]:
    """Callback du panneau monitoring : recalcule carte CSAT + top utilisateurs.

    Lit la donnée live depuis Supabase via la façade LTM (un seul pool).
    """
    if LTM is None:
        return "### 📊 CSAT — _Base de données indisponible_", []
    glob = LTM.get_csat()
    week = LTM.get_csat(window_days=7)
    per_user = LTM.get_csat_per_user(limit=10)
    table = [
        [
            u["name"], u["total"], u["thumbs_up"], u["thumbs_down"],
            f"{u['csat']:.0%}" if u["csat"] is not None else "—",
        ]
        for u in per_user
    ]
    return _format_csat_card(glob, week), table


# ============================================================================
# BUILD UI
# ============================================================================

PULS_THEME = gr.themes.Soft(
    primary_hue="rose",
    secondary_hue="indigo",
)

with gr.Blocks(theme=PULS_THEME, title="Puls · Événements culturels en France") as demo:
    gr.Markdown("# 🎭 Puls · Événements culturels")
    gr.Markdown(
        "_Votre guide culturel conversationnel — propulsé par l'IA et OpenAgenda._ "
        "Couverture : **Paris, Lyon, Marseille, Toulouse, Nantes, Bordeaux, Lille, Strasbourg.**"
    )

    short_term_state = gr.State(ShortTermMemory(window_size=MEMORY_WINDOW_SIZE))
    user_state = gr.State({})
    session_state = gr.State({})
    # [D4] Métadonnées d'attribution des votes : {index_message_chatbot ->
    # {message_id, trace_id, question, response}}. Peuplé par respond(), lu
    # par on_like (gr.LikeData.index), réinitialisé à chaque reconstruction
    # du chatbot (nouvelle conversation / session rechargée).
    feedback_meta_state = gr.State({})

    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            with gr.Row():
                # ────────── Colonne historique (gauche, sidebar conversations) ──────────
                # [D1+] Sidebar style Claude.ai / ChatGPT : liste des sessions passées
                # de l'utilisateur connecté, cliquables pour recharger la conversation.
                with gr.Column(scale=1, min_width=220):
                    gr.Markdown("### 💬 Conversations")

                    new_conv_btn = gr.Button(
                        "➕ Nouvelle conversation",
                        variant="primary",
                    )

                    sessions_radio = gr.Radio(
                        choices=[],
                        label="",
                        interactive=True,
                        visible=False,
                        elem_id="sessions-list",
                    )

                    sessions_empty_msg = gr.Markdown(
                        "_Connecte-toi pour voir tes conversations._",
                        visible=True,
                    )

                # ────────── Colonne centre : conversation ──────────
                with gr.Column(scale=5):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        type="messages",
                        value=WELCOME_MESSAGE,
                        allow_tags=True,
                    )

                    msg_input = gr.Textbox(
                        placeholder="Pose ta question sur les événements culturels…",
                        show_label=False,
                    )

                    # [D3] Une chip dédiée web (billetterie) ajoutée
                    # [TOP-8] Chips mises à jour pour démo multi-villes
                    gr.Examples(
                        examples=[
                            ["Quels événements culturels ce week-end ?"],
                            ["Tu te souviens de ce que j'aime ?"],
                            ["Et à Paris, qu'est-ce qu'il y a ?"],
                            ["Compare les festivals à Lyon et Marseille"],
                            ["Y a-t-il encore des billets pour ce soir ?"],
                        ],
                        inputs=[msg_input],
                        label="💡 Suggestions",
                    )

                    send_btn = gr.Button("Envoyer", variant="primary")

                # ────────── Colonne droite : profil + mode démo ──────────
                # [UX] Déplacé de gauche à droite — convention Claude.ai / ChatGPT :
                # historique à gauche (consulté en premier), profil à droite (config
                # ponctuelle).
                with gr.Column(scale=2, min_width=320):
                    gr.Markdown("### 🔧 Mode démo — Simulation utilisateur")
                    gr.Markdown(
                        "<small style='opacity:0.7;line-height:1.4;display:block;"
                        "padding:0.3em 0 0.7em;'>"
                        "En production, cette zone serait remplacée par une "
                        "authentification <b>Supabase Auth</b> (magic link email). "
                        "Pour la démonstration, choisis ou crée un profil ci-dessous "
                        "pour activer la mémoire conversationnelle <i>(défi D1)</i>."
                        "</small>",
                    )

                    user_dropdown = gr.Dropdown(
                        choices=get_user_list(),
                        label="Profil de démonstration",
                        interactive=True,
                        allow_custom_value=True,
                    )

                    new_user_input = gr.Textbox(
                        label="…ou créer un profil de test",
                        placeholder="Ex: Léa, Thomas, …",
                    )

                    city_input = gr.Textbox(
                        label="📍 Ta ville",
                        placeholder="Ex: Nantes, Saint-Nazaire, …",
                        info="Géocodée à la connexion. Override possible en disant "
                             "« à [autre ville] » dans une question.",
                    )

                    select_btn = gr.Button("Simuler la connexion", variant="primary")

                    profile_display = gr.Markdown("_Aucun profil actif._")

                    if SHOW_DEBUG:
                        gr.Markdown("---")
                        web_status = "✅" if (WEB_AGENT and BRAVE_CLIENT.is_available()) else (
                            "🟡 DDG only" if WEB_AGENT else "❌"
                        )
                        gr.Markdown(
                            f"**Modèle** : `{CHAT_MODEL}`\n\n"
                            f"**Routeur D3** : `{EXTRACTION_MODEL}`\n\n"
                            f"**Mémoire courte** : {MEMORY_WINDOW_SIZE} tours\n\n"
                            f"**Index** : `{FAISS_INDEX_PATH}`\n\n"
                            f"**Top-K géo** : {RETRIEVER_K_GEO} → {RETRIEVER_K}\n\n"
                            f"**Rayon défaut** : {DEFAULT_RADIUS_KM} km\n\n"
                            f"**Agent web (D3)** : {web_status}\n\n"
                            f"**Whitelist** : {WHITELIST.count() if WHITELIST else 0} domaines",
                            elem_id="debug-panel",
                        )

            status_md = gr.Markdown(_status_line({}, ""))

        with gr.Tab("📊 Monitoring CSAT"):
            gr.Markdown("### Satisfaction utilisateur — données live Supabase")
            gr.Markdown(
                "<small style='opacity:0.7;'>Votes 👍/👎 collectés sous chaque "
                "réponse de l'assistant (US-704). Clique sur Rafraîchir pour "
                "recharger l'agrégat depuis Supabase.</small>"
            )
            csat_card = gr.Markdown()
            csat_table = gr.Dataframe(
                headers=["utilisateur", "total", "👍", "👎", "csat"],
                datatype=["str", "number", "number", "number", "str"],
                label="Top 10 utilisateurs par volume de votes",
                wrap=True,
                interactive=False,
            )
            csat_refresh_btn = gr.Button("🔄 Rafraîchir", variant="primary")

    # ────────── Wiring ──────────

    def on_select(existing_dropdown: str, new_name: str, city: str):
        name = new_name.strip() if new_name and new_name.strip() else existing_dropdown
        user_s, sess_s, profile, city_resolved = select_user(name, city)

        # [SIDEBAR] Peupler la liste des sessions passées de l'utilisateur
        sessions_choices = []
        sidebar_empty_visible = True
        sidebar_radio_visible = False
        if user_s and user_s.get("id") and LTM is not None:
            try:
                sessions = LTM.list_user_sessions(user_s["id"], limit=20)
                # Exclure la session courante (celle qu'on vient d'ouvrir)
                current_sid = sess_s.get("id") if sess_s else None
                sessions = [s for s in sessions if s["session_id"] != current_sid]
                sessions_choices = _format_sessions_for_sidebar(sessions)
                if sessions_choices:
                    sidebar_empty_visible = False
                    sidebar_radio_visible = True
            except Exception as e:
                logger.warning(f"Sidebar : échec list_user_sessions : {e}")

        return (
            user_s, sess_s, profile,
            gr.update(choices=get_user_list(), value=name),
            "",
            city_resolved,
            # [SIDEBAR] 2 sorties supplémentaires
            gr.update(choices=sessions_choices, value=None, visible=sidebar_radio_visible),
            gr.update(visible=sidebar_empty_visible),
        )

    def on_session_click(
        session_id,           # peut être int ou None
        short_term: ShortTermMemory,
        user_state: Dict,
    ):
        """[SIDEBAR] Recharge une conversation passée depuis Supabase.

        Effets :
        - chatbot : rempli avec l'historique de la session sélectionnée
        - short_term : neuf (cohérent avec new_conversation)
        - session_state : pointe vers la session sélectionnée (devient active)
        """
        if not session_id or LTM is None:
            return gr.update(), short_term, gr.update(), gr.update()

        user_id = user_state.get("id") if user_state else None
        if user_id is None:
            logger.warning("Sidebar : clic session sans utilisateur actif — ignoré")
            return gr.update(), short_term, gr.update(), gr.update()

        # Garde anti-IDOR : les session_id sont des entiers séquentiels
        # devinables — on refuse de charger une session hors périmètre de
        # l'utilisateur actif. Réutilise list_user_sessions (déjà borné).
        owned = {s["session_id"] for s in LTM.list_user_sessions(user_id, limit=20)}
        if int(session_id) not in owned:
            logger.warning(
                f"Sidebar : session {session_id} hors périmètre user "
                f"{user_id} — refus"
            )
            return gr.update(), short_term, gr.update(), gr.update()

        try:
            messages = LTM.load_session_history(int(session_id))
        except Exception as e:
            logger.warning(f"Sidebar : échec load_session_history({session_id}) : {e}")
            return gr.update(), short_term, gr.update(), gr.update()

        chat_history = _rebuild_chat_history(messages)
        new_short_term = ShortTermMemory(window_size=MEMORY_WINDOW_SIZE)
        new_session_state = {"id": int(session_id)}

        logger.info(
            f"[SIDEBAR] Conversation #{session_id} rechargée "
            f"({len(chat_history)} messages)"
        )
        # 4e retour {} : reset feedback_meta — les index repartent de zéro.
        # (Conséquence : voter sur un message d'une session rechargée n'aura
        #  pas de métadonnées d'attribution ; le vote brut est tout de même
        #  persisté pour préserver le signal CSAT — cf. on_like.)
        return chat_history, new_short_term, new_session_state, {}

    def on_like(
        evt: gr.LikeData,
        user_state: Dict,
        session_state: Dict,
        feedback_meta: Dict[int, Dict],
    ):
        """[US-704 / D4] Persiste un vote 👍/👎 avec attribution au message.

        Récupère via `evt.index` (position du message voté dans le chatbot)
        les métadonnées d'attribution stockées par respond() : message_id,
        trace_id Langfuse, et snapshots question/réponse. Fire-and-forget
        (aucun output UI). Sans utilisateur/session actif → on ignore.

        Si le message voté n'a pas de métadonnées (ex. session rechargée
        depuis l'historique), on persiste quand même le vote brut pour ne pas
        perdre le signal CSAT, en loggant un warning.
        """
        if LTM is None or not user_state or "id" not in user_state:
            return
        session_id = session_state.get("id") if session_state else None
        if not session_id:
            return

        rating = 1 if evt.liked else -1
        idx = evt.index
        if isinstance(idx, (list, tuple)):  # robustesse selon type de chatbot
            idx = idx[0] if idx else None
        meta = (feedback_meta or {}).get(idx) if idx is not None else None

        try:
            if meta is None:
                logger.warning(
                    f"[US-704] Vote sans métadonnées (index={evt.index}) — "
                    f"message probablement rechargé ; vote brut persisté."
                )
                fb_id = LTM.add_feedback(session_id, user_state["id"], rating)
            else:
                fb_id = LTM.add_feedback(
                    session_id=session_id,
                    user_id=user_state["id"],
                    rating=rating,
                    message_id=meta.get("message_id"),
                    question_snapshot=meta.get("question"),
                    response_snapshot=meta.get("response"),
                    langfuse_trace_id=meta.get("trace_id"),
                )
            logger.info(
                f"[US-704] Feedback {'👍' if rating == 1 else '👎'} "
                f"(fb={fb_id}, msg={meta.get('message_id') if meta else None}, "
                f"trace={'oui' if (meta and meta.get('trace_id')) else 'non'}, "
                f"session={session_id})"
            )
        except Exception as e:
            logger.warning(f"[US-704] Échec add_feedback : {e}")

    sessions_radio.change(
        fn=on_session_click,
        inputs=[sessions_radio, short_term_state, user_state],
        outputs=[chatbot, short_term_state, session_state, feedback_meta_state],
    )

    chatbot.like(
        fn=on_like,
        inputs=[user_state, session_state, feedback_meta_state],
    )

    select_btn.click(
        fn=on_select,
        inputs=[user_dropdown, new_user_input, city_input],
        outputs=[
            user_state, session_state, profile_display,
            user_dropdown, new_user_input, city_input,
            sessions_radio, sessions_empty_msg,   # [SIDEBAR]
        ],
    )

    send_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot, short_term_state, user_state, session_state, feedback_meta_state],
        outputs=[msg_input, chatbot, short_term_state, feedback_meta_state],
    )

    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot, short_term_state, user_state, session_state, feedback_meta_state],
        outputs=[msg_input, chatbot, short_term_state, feedback_meta_state],
    )

    new_conv_btn.click(
        fn=new_conversation,
        inputs=[short_term_state, user_state, session_state],
        outputs=[chatbot, short_term_state, session_state, profile_display, feedback_meta_state],
    )

    user_state.change(
        fn=_status_line,
        inputs=[user_state, city_input],
        outputs=[status_md],
    )
    city_input.change(
        fn=_status_line,
        inputs=[user_state, city_input],
        outputs=[status_md],
    )

    # [D4] Monitoring CSAT — chargement au boot + bouton rafraîchir
    demo.load(
        fn=refresh_csat_dashboard,
        outputs=[csat_card, csat_table],
    )
    csat_refresh_btn.click(
        fn=refresh_csat_dashboard,
        outputs=[csat_card, csat_table],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )

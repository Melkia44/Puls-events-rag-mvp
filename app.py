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
# Imports
# ============================================================================
import logging
import re
from typing import List, Tuple, Dict, Any, Optional

import gradio as gr
from langchain_mistralai import ChatMistralAI

from utils.config import (
    CHAT_MODEL, EXTRACTION_MODEL, EMBEDDING_MODEL,
    FAISS_INDEX_PATH, RETRIEVER_K, MEMORY_WINDOW_SIZE,
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

RETRIEVER_K_GEO = 20


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
) -> Tuple[str, List, Optional[Dict]]:
    """Pipeline RAG : retrieval → [filtre géo] → prompt → Mistral.

    Retourne (texte, docs_finaux, geo_info).
    """
    if VECTOR_STORE is None or LLM is None:
        return ("⚠️ Erreur de configuration.", [], None)

    geo_target = _resolve_geo_target(user_message, user_id)

    if geo_target is not None:
        candidates = VECTOR_STORE.similarity_search(user_message, k=RETRIEVER_K_GEO)
        filtered, fallback = filter_by_radius(
            candidates,
            user_lat=geo_target["lat"], user_lng=geo_target["lng"],
            radius_km=geo_target["radius_km"],
        )
        docs = filtered[:RETRIEVER_K]
        geo_info = {**geo_target, "fallback": fallback}
        logger.info(
            f"D2 actif : filtre {geo_target['radius_km']}km autour de "
            f"{geo_target['city']} (source={geo_target['source']}, "
            f"fallback={fallback}, kept={len(docs)})"
        )
    else:
        # [TOP-8 — patch β] Si désactivation due au multi-villes,
        # on élargit le top-K pour donner plus de matière à Mistral
        # (sinon 5 docs ne suffisent pas à structurer une comparaison)
        is_multi, _ = _detect_multi_city_query(user_message)
        k = RETRIEVER_K * 2 if is_multi else RETRIEVER_K  # 10 vs 5
        docs = VECTOR_STORE.similarity_search(user_message, k=k)
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

    response = LLM.invoke(prompt)
    return (response.content, docs, geo_info)


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
            logger.warning(f"Ville '{city_to_display}' non géocodable, non persistée")
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


def respond(
    message: str,
    chat_history: List[Dict],
    short_term: ShortTermMemory,
    user_state: Dict,
    session_state: Dict,
) -> Tuple[str, List[Dict], ShortTermMemory]:
    if not message or not message.strip():
        return "", chat_history, short_term

    if not user_state or "id" not in user_state:
        warning = "⚠️ Active d'abord un profil de démonstration dans la barre latérale."
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": warning},
        ]
        return "", chat_history, short_term

    user_id = user_state["id"]
    session_id = session_state.get("id") if session_state else None

    # [D3] Routage RAG vs Web — décision en amont, latence ~500ms si LLM router
    # On utilise les triggers regex + fallback retrieval count si possible
    try:
        decision, route_reason = route_to_rag_or_web(
            message,
            llm_router=LLM_ROUTER,
        )
        logger.info(f"D3 routage : {decision} ({route_reason})")
    except Exception as e:
        logger.warning(f"D3 routage error : {e} → RAG par défaut")
        decision, route_reason = "rag", "routing_error"

    # Exécution selon le routage
    response_with_extras = ""
    response_clean = ""
    try:
        if decision == "web" and WEB_AGENT is not None:
            # [D3] Pipeline agent web
            web_resp = WEB_AGENT.run(message)
            response_clean = web_resp.text
            response_with_extras = response_clean + _format_web_block(web_resp, route_reason)
        else:
            # Pipeline RAG (D1 + D2)
            response_clean, sources, geo_info = rag_response(message, short_term, user_id)
            response_with_extras = response_clean + _format_sources_html(sources, geo_info)
    except Exception as e:
        logger.error(f"Erreur respond : {e}")
        response_clean = f"⚠️ Erreur lors de la génération : {str(e)[:200]}"
        response_with_extras = response_clean

    # Mémoire courte = version sans HTML (pas de pollution du contexte)
    short_term.add_turn(message, response_clean)

    if LTM is not None and session_id:
        try:
            LTM.log_message(session_id, "user", message)
            LTM.log_message(session_id, "assistant", response_clean)
        except Exception as e:
            logger.warning(f"Échec log message : {e}")

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_with_extras},
    ]
    return "", chat_history, short_term


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
) -> Tuple[List[Dict], ShortTermMemory, Dict, str]:
    if not user_state or "id" not in user_state:
        return list(WELCOME_MESSAGE), ShortTermMemory(window_size=MEMORY_WINDOW_SIZE), {}, "Pas d'utilisateur actif."

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

    return list(WELCOME_MESSAGE), ShortTermMemory(window_size=MEMORY_WINDOW_SIZE), new_state, profile_display + f"\n\n{info}"


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

    with gr.Row():
        # ────────── Colonne gauche : mock d'auth + position ──────────
        with gr.Column(scale=1):
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

            new_conv_btn = gr.Button("🔄 Nouvelle conversation", variant="secondary")

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

        # ────────── Colonne droite : conversation ──────────
        with gr.Column(scale=3):
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

    status_md = gr.Markdown(_status_line({}, ""))

    # ────────── Wiring ──────────

    def on_select(existing_dropdown: str, new_name: str, city: str):
        name = new_name.strip() if new_name and new_name.strip() else existing_dropdown
        user_s, sess_s, profile, city_resolved = select_user(name, city)
        return (
            user_s, sess_s, profile,
            gr.update(choices=get_user_list(), value=name),
            "",
            city_resolved,
        )

    select_btn.click(
        fn=on_select,
        inputs=[user_dropdown, new_user_input, city_input],
        outputs=[user_state, session_state, profile_display, user_dropdown, new_user_input, city_input],
    )

    send_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot, short_term_state, user_state, session_state],
        outputs=[msg_input, chatbot, short_term_state],
    )

    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot, short_term_state, user_state, session_state],
        outputs=[msg_input, chatbot, short_term_state],
    )

    new_conv_btn.click(
        fn=new_conversation,
        inputs=[short_term_state, user_state, session_state],
        outputs=[chatbot, short_term_state, session_state, profile_display],
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


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )

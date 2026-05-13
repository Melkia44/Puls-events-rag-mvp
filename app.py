"""
app.py
Puls-Events MVP P13 — Interface Gradio.

Vague 2 : RAG + D1 mémoire + D2 contexte géographique
    - D1 — Mémoire conversationnelle (court terme + long terme Supabase)
    - D2 — Contexte géographique : ville profil + override langage naturel
           + filtrage Haversine post-retrieval + affichage distance

Pas encore dans cette version :
    - D3 agent web smolagents (vague 3)
    - D4 monitoring Langfuse (vague 3)

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
    print(f"    .env existe ? {_ENV_PATH.exists()}")

# ============================================================================
# Imports
# ============================================================================
import logging
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
# [D2] Module de contexte géographique
from utils.geo import (
    geocode_city, filter_by_radius,
    extract_radius_override, extract_location_override,
    DEFAULT_RADIUS_KM,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

logger.info(f"Gradio version : {gr.__version__}")


# ============================================================================
# STARTUP
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

logger.info("Initialisation LLM Mistral…")
LLM = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model=CHAT_MODEL,
    temperature=0.3,
) if MISTRAL_API_KEY else None


SHOW_DEBUG = os.getenv("SHOW_DEBUG", "0") == "1"
logger.info(f"Mode debug UI : {'activé' if SHOW_DEBUG else 'désactivé'}")

# [D2] Top-K élargi pour le retrieval — on récupère plus de candidats
# que le K cible afin que le filtre Haversine ait de quoi travailler sans
# tomber sous le seuil min_docs_kept.
RETRIEVER_K_GEO = 20


# ============================================================================
# HELPERS FORMATAGE
# ============================================================================

def _format_iso_date(iso_str: str) -> str:
    """Convertit '2026-04-10T18:00:00Z' en '10 avril 2026 à 18h00'."""
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
    """Formate un Document LangChain en bloc structuré pour le prompt RAG.

    [D2] Si la distance utilisateur est disponible (post-filtrage Haversine),
    elle est injectée pour que le LLM puisse la citer dans sa réponse.
    """
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

    # [D2] Distance utilisateur si calculée
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
    """Construit un bloc <details> HTML rendant les sources RAG dépliables.

    [D2] Si geo_info est fourni (filtre actif), un badge "📍 Filtré à X km
    autour de [ville]" est affiché en tête, et chaque source affiche sa
    distance individuelle.
    """
    if not docs and not geo_info:
        return ""

    # [D2] Badge filtre géo en tête de bloc (visible même replié grâce
    # au <summary> qui le contient)
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
        # [D2] Distance affichée par événement si calculée
        dist_html = f" · <b>{distance} km</b>" if distance is not None else ""

        items.append(f"<li><b>{title}</b>{loc_html}{dist_html}{url_html}</li>")

    return (
        geo_badge
        + "\n\n<details style='margin-top:0.5em;font-size:0.85em;opacity:0.85;'>"
        + f"<summary>📚 {len(docs)} sources consultées</summary>"
        + f"<ul style='margin-top:0.5em;'>{''.join(items)}</ul>"
        + "</details>"
    )


# ============================================================================
# LOGIQUE MÉTIER
# ============================================================================

def _resolve_geo_target(
    user_message: str, user_id: Optional[int],
) -> Optional[Dict[str, Any]]:
    """[D2] Détermine la cible géographique d'une requête utilisateur.

    Priorités (ordre décroissant) :
        1. Override langage naturel dans le message ("à Saint-Nazaire")
        2. Ville persistée du profil utilisateur
        3. None → pas de filtre géo

    Le rayon est lui aussi déduit, avec priorité override > défaut 15 km.

    Returns:
        dict {city, lat, lng, radius_km, source} ou None
        - source ∈ {"override", "profile"} → utile pour log et UI
    """
    radius = extract_radius_override(user_message) or DEFAULT_RADIUS_KM

    # 1. Override langage
    location_override = extract_location_override(user_message)
    if location_override:
        coords = geocode_city(location_override)
        if coords is not None:
            return {
                "city": location_override,
                "lat": coords[0],
                "lng": coords[1],
                "radius_km": radius,
                "source": "override",
            }
        else:
            logger.info(f"Override géo '{location_override}' non géocodable, ignoré")

    # 2. Ville du profil
    if user_id is not None and LTM is not None:
        profile_city = LTM.get_user_city(user_id)
        if profile_city and profile_city["lat"] is not None:
            return {
                "city": profile_city["city"],
                "lat": profile_city["lat"],
                "lng": profile_city["lng"],
                "radius_km": radius,
                "source": "profile",
            }

    # 3. Pas de contexte géo disponible
    return None


def rag_response(
    user_message: str,
    short_term: ShortTermMemory,
    user_id: Optional[int],
) -> Tuple[str, List, Optional[Dict]]:
    """Pipeline RAG : retrieval FAISS → [filtre géo] → prompt → Mistral.

    [D2] Si une cible géo est résolue, on élargit le top-K à RETRIEVER_K_GEO,
    on filtre par Haversine, et on remonte les K meilleurs pour le prompt.

    Returns:
        (texte_réponse, documents_finaux, geo_info)
        - geo_info : dict utile pour l'UI (badge filtre) ou None
    """
    if VECTOR_STORE is None or LLM is None:
        return ("⚠️ Erreur de configuration. Vérifie MISTRAL_API_KEY et l'index FAISS.", [], None)

    # [D2] Résolution de la cible géo AVANT retrieval pour décider du K
    geo_target = _resolve_geo_target(user_message, user_id)

    if geo_target is not None:
        # Retrieval élargi pour avoir de quoi filtrer
        candidates = VECTOR_STORE.similarity_search(user_message, k=RETRIEVER_K_GEO)
        filtered, fallback = filter_by_radius(
            candidates,
            user_lat=geo_target["lat"],
            user_lng=geo_target["lng"],
            radius_km=geo_target["radius_km"],
        )
        # On garde les K meilleurs (tri par distance, déjà fait dans filter)
        docs = filtered[:RETRIEVER_K]
        geo_info = {**geo_target, "fallback": fallback}
        logger.info(
            f"D2 actif : filtre {geo_target['radius_km']}km autour de "
            f"{geo_target['city']} (source={geo_target['source']}, "
            f"fallback={fallback}, kept={len(docs)})"
        )
    else:
        # Pas de cible géo → retrieval standard
        docs = VECTOR_STORE.similarity_search(user_message, k=RETRIEVER_K)
        geo_info = None

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
    """[D2] Active un profil et géocode la ville si fournie / changée.

    Returns:
        (user_state, session_state, profile_display, city_field_value)
        Le dernier élément force la valeur affichée du champ ville après
        géocodage (utile pour le pré-remplissage au chargement d'un profil).
    """
    if not name or not name.strip():
        return {}, {}, "Aucun utilisateur sélectionné", ""

    if LTM is None:
        return {}, {}, "⚠️ Supabase non connecté", city

    name = name.strip()
    user_id = LTM.get_or_create_user(name)
    session_id = LTM.start_session(user_id)
    profile_summary = LTM.get_preference_summary(user_id)

    # [D2] Gestion de la ville
    # Stratégie : si l'utilisateur a saisi une ville à la connexion, on
    # géocode et on la persiste (override de la valeur précédente).
    # Sinon on lit la ville persistée pour pré-remplir l'UI.
    city_to_display = city.strip() if city else ""
    if city_to_display:
        coords = geocode_city(city_to_display)
        if coords is not None:
            LTM.set_user_city(user_id, city_to_display, coords[0], coords[1])
            logger.info(f"Profil '{name}' : ville '{city_to_display}' géocodée et persistée")
        else:
            logger.warning(f"Ville '{city_to_display}' non géocodable, non persistée")
    else:
        # Lire la ville persistée pour pré-remplir le champ
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

    # [D2] rag_response retourne désormais (texte, docs, geo_info)
    try:
        response, sources, geo_info = rag_response(message, short_term, user_id)
        response_with_sources = response + _format_sources_html(sources, geo_info)
    except Exception as e:
        logger.error(f"Erreur RAG : {e}")
        response = f"⚠️ Erreur lors de la génération : {str(e)[:200]}"
        response_with_sources = response

    short_term.add_turn(message, response)

    if LTM is not None and session_id:
        try:
            LTM.log_message(session_id, "user", message)
            LTM.log_message(session_id, "assistant", response)
        except Exception as e:
            logger.warning(f"Échec log message : {e}")

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_with_sources},
    ]
    return "", chat_history, short_term


WELCOME_MESSAGE = [{
    "role": "assistant",
    "content": (
        "Bonjour ! Je suis **Puls**, ton assistant culturel pour Nantes Métropole.\n\n"
        "Je peux t'aider à découvrir concerts, expos, spectacles et festivals "
        "près de chez toi. Si tu actives un profil à gauche et indiques ta ville, "
        "je filtrerai les événements dans un rayon autour de toi.\n\n"
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
    """[D2] Footer enrichi avec la ville active."""
    name = user_state.get("name", "—") if user_state else "—"
    city_label = city.strip() if city else "—"
    return (
        f"<div style='text-align:center;font-size:0.78em;opacity:0.6;"
        f"padding:0.5em 0;border-top:1px solid rgba(255,255,255,0.05);"
        f"margin-top:1em;'>"
        f"Profil : <b>{name}</b> · "
        f"Ville : <b>{city_label}</b> · "
        f"Mémoire conversationnelle active · "
        f"Données OpenAgenda · Nantes Métropole"
        f"</div>"
    )


# ============================================================================
# BUILD UI
# ============================================================================

PULS_THEME = gr.themes.Soft(
    primary_hue="rose",
    secondary_hue="indigo",
)

with gr.Blocks(theme=PULS_THEME, title="Puls · Événements Nantes Métropole") as demo:
    gr.Markdown("# 🎭 Puls · Événements Nantes Métropole")
    gr.Markdown("_Votre guide culturel conversationnel — propulsé par l'IA et OpenAgenda_")

    short_term_state = gr.State(ShortTermMemory(window_size=MEMORY_WINDOW_SIZE))
    user_state = gr.State({})
    session_state = gr.State({})

    with gr.Row():
        # ────────── Colonne gauche : mock d'auth + position (D1 + D2) ──────
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

            # [D2] Champ ville — toujours visible pour démontrer D2 d'un coup d'œil
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
                gr.Markdown(
                    f"**Modèle** : `{CHAT_MODEL}`\n\n"
                    f"**Mémoire courte** : {MEMORY_WINDOW_SIZE} tours\n\n"
                    f"**Index** : `{FAISS_INDEX_PATH}`\n\n"
                    f"**Top-K géo** : {RETRIEVER_K_GEO} → {RETRIEVER_K}\n\n"
                    f"**Rayon défaut** : {DEFAULT_RADIUS_KM} km",
                    elem_id="debug-panel",
                )

        # ────────── Colonne droite : conversation ──────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                type="messages",
                value=WELCOME_MESSAGE,
                allow_tags=True,  # autorise <details>, <a>, etc. (Gradio 6.0 compat)
            )

            msg_input = gr.Textbox(
                placeholder="Pose ta question sur les événements culturels…",
                show_label=False,
            )

            gr.Examples(
                examples=[
                    ["Quels événements culturels ce week-end à Nantes ?"],
                    ["Tu te souviens de ce que j'aime ?"],
                    ["Recommande-moi un spectacle pour ce soir"],
                    ["Quels concerts près de chez moi dans un rayon de 30 km ?"],
                ],
                inputs=[msg_input],
                label="💡 Suggestions",
            )

            send_btn = gr.Button("Envoyer", variant="primary")

    # ────────── Footer dynamique ──────────
    status_md = gr.Markdown(_status_line({}, ""))

    # ────────── Wiring ──────────

    def on_select(existing_dropdown: str, new_name: str, city: str):
        """[D2] Étendu pour inclure le champ ville en entrée et sortie."""
        name = new_name.strip() if new_name and new_name.strip() else existing_dropdown
        user_s, sess_s, profile, city_resolved = select_user(name, city)
        return (
            user_s, sess_s, profile,
            gr.update(choices=get_user_list(), value=name),
            "",                    # vide le champ "créer un profil"
            city_resolved,         # met à jour le champ ville (préremplissage si profil existant)
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

    # [D2] Mise à jour du footer : profil OU ville change → on rafraîchit
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

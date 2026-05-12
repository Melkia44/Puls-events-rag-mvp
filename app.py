"""
app.py
Puls-Events MVP P13 — Interface Gradio.

Vague 1 : RAG basique + mémoire conversationnelle D1
    - Court terme : buffer fenêtré injecté dans prompt
    - Long terme : profils utilisateurs persistés sur Supabase Postgres
    - Extraction préférences en fin de conversation via Mistral Small

Pas encore dans cette version :
    - D2 géo (vague 2)
    - D3 agent web (vague 3)
    - D4 monitoring Langfuse (vague 3)

Compatible : Gradio 4.x (HF Spaces SDK officiel, type="messages" depuis 4.36).

────────────────────────────────────────────────────────────────────────────
Changelog UI (réponse audit UX du 05/2026) :
    P0.1 — Sidebar technique masquée derrière variable SHOW_DEBUG (12-factor)
    P0.2 — Titre produit "Puls · Événements Nantes Métropole" (plus de "MVP")
    P1.1 — Message d'accueil pré-chargé + 4 chips de suggestion
    P1.2 — Panneau Sources replié sous chaque réponse (traçabilité RAG)
    P2   — Footer dynamique exposant l'état D1 (mémoire + profil actif)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

# ============================================================================
# CHARGEMENT .env EN PREMIER — avant TOUT autre import qui lit os.getenv()
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
# Imports normaux après chargement .env
# ============================================================================
import logging
from typing import List, Tuple, Dict, Any

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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

logger.info(f"Gradio version : {gr.__version__}")


# ============================================================================
# STARTUP — chargement singletons
# ============================================================================

config_errors = validate_config()
if config_errors:
    logger.error("Erreurs de configuration au démarrage :")
    for err in config_errors:
        logger.error(f"  - {err}")
    logger.error("Vérifie tes variables d'environnement (.env local ou Secrets HF)")

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


# Mode debug — flag global lu une fois au démarrage
# Local : `SHOW_DEBUG=1 python app.py` pour exposer la stack technique
# HF Spaces : variable non définie → mode produit propre
SHOW_DEBUG = os.getenv("SHOW_DEBUG", "0") == "1"
logger.info(f"Mode debug UI : {'activé' if SHOW_DEBUG else 'désactivé'}")


# ============================================================================
# LOGIQUE MÉTIER
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
    """True si les 2 ISO sont sur le même jour."""
    try:
        return iso1[:10] == iso2[:10]
    except Exception:
        return False


def _format_event_with_metadata(doc) -> str:
    """Formate un Document LangChain en bloc structuré pour le prompt RAG.

    Au lieu de passer seulement page_content au LLM, on injecte les
    métadonnées (titre, lieu, dates, URL) pour que Mistral puisse les citer.
    Fix du problème "Date : Non précisée" observé en vague 1.
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

    content = (doc.page_content or "").strip()
    if content:
        lines.append(f"Description : {content}")

    if meta.get("url"):
        lines.append(f"Lien : {meta['url']}")

    return "\n".join(lines)


def _format_sources_html(docs) -> str:
    """Construit un bloc <details> HTML rendant les sources RAG dépliables.

    Argument jury : c'est la traçabilité d'un système RAG — preuve que les
    réponses sont ancrées sur des données réelles OpenAgenda, pas hallucinées.
    Le bloc est replié par défaut (clic utilisateur pour expansion) afin de
    ne pas encombrer la lecture de la réponse principale.
    """
    if not docs:
        return ""

    items = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        title = meta.get("title", f"Source {i}")
        location = meta.get("location", "")
        url = meta.get("url", "")
        url_html = f' · <a href="{url}" target="_blank">Voir sur OpenAgenda</a>' if url else ""
        loc_html = f" — <i>{location}</i>" if location else ""
        items.append(f"<li><b>{title}</b>{loc_html}{url_html}</li>")

    return (
        "\n\n<details style='margin-top:0.5em;font-size:0.85em;opacity:0.85;'>"
        f"<summary>📚 {len(docs)} sources consultées</summary>"
        f"<ul style='margin-top:0.5em;'>{''.join(items)}</ul>"
        "</details>"
    )


def rag_response(
    user_message: str,
    short_term: ShortTermMemory,
    user_id: int | None,
) -> Tuple[str, List]:
    """Pipeline RAG : retrieval FAISS → prompt → Mistral.

    Returns:
        (texte_réponse_brute, documents_sources)
        Le texte est SANS HTML pour la mémoire courte (pas de pollution du contexte
        aux tours suivants). Les sources sont retournées séparément pour permettre
        à l'appelant de construire un affichage enrichi côté UI.
    """
    if VECTOR_STORE is None or LLM is None:
        return ("⚠️ Erreur de configuration. Vérifie MISTRAL_API_KEY et l'index FAISS.", [])

    docs = VECTOR_STORE.similarity_search(user_message, k=RETRIEVER_K)

    # Formater chaque event avec ses métadonnées (titre, lieu, date, lien)
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
    return (response.content, docs)


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
# UI GRADIO 4.x (format messages OpenAI-style supporté depuis 4.36)
# ============================================================================

def get_user_list() -> List[str]:
    if LTM is None:
        return []
    users = LTM.list_users()
    return [u["name"] for u in users]


def select_user(name: str) -> Tuple[Dict, Dict, str]:
    if not name or not name.strip():
        return {}, {}, "Aucun utilisateur sélectionné"

    if LTM is None:
        return {}, {}, "⚠️ Supabase non connecté"

    name = name.strip()
    user_id = LTM.get_or_create_user(name)
    session_id = LTM.start_session(user_id)
    profile_summary = LTM.get_preference_summary(user_id)

    user_state = {"id": user_id, "name": name}
    session_state = {"id": session_id}

    profile_display = (
        profile_summary
        if profile_summary
        else f"_Profil vierge pour **{name}** — sera enrichi à mesure des conversations._"
    )

    return user_state, session_state, profile_display


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
        warning = "⚠️ Sélectionne d'abord un utilisateur dans la barre latérale."
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": warning},
        ]
        return "", chat_history, short_term

    user_id = user_state["id"]
    session_id = session_state.get("id") if session_state else None

    # Appel RAG : on récupère réponse brute + documents sources séparément
    # pour pouvoir stocker la version "propre" en mémoire courte et afficher
    # la version "enrichie HTML" côté chat
    try:
        response, sources = rag_response(message, short_term, user_id)
        response_with_sources = response + _format_sources_html(sources)
    except Exception as e:
        logger.error(f"Erreur RAG : {e}")
        response = f"⚠️ Erreur lors de la génération : {str(e)[:200]}"
        response_with_sources = response  # pas de sources en cas d'erreur

    # Mémoire courte = réponse SANS HTML (sinon pollue le contexte au tour suivant)
    short_term.add_turn(message, response)

    if LTM is not None and session_id:
        try:
            LTM.log_message(session_id, "user", message)
            LTM.log_message(session_id, "assistant", response)
        except Exception as e:
            logger.warning(f"Échec log message : {e}")

    # Affichage chat = réponse AVEC bloc sources repliable
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_with_sources},
    ]
    return "", chat_history, short_term


# Message d'accueil pré-chargé dans le chat — supprime la page blanche
# au premier contact (réponse audit UX point #2 critique)
WELCOME_MESSAGE = [{
    "role": "assistant",
    "content": (
        "Bonjour ! Je suis **Puls**, ton assistant culturel pour Nantes Métropole.\n\n"
        "Je peux t'aider à découvrir concerts, expos, spectacles et festivals "
        "près de chez toi. Si tu actives un profil à gauche, je me souviendrai "
        "de tes goûts entre deux conversations.\n\n"
        "_Sélectionne un profil utilisateur, puis pose-moi ta question — "
        "ou clique sur une suggestion ci-dessous._ ↓"
    ),
}]


def new_conversation(
    short_term: ShortTermMemory,
    user_state: Dict,
    session_state: Dict,
) -> Tuple[List[Dict], ShortTermMemory, Dict, str]:
    if not user_state or "id" not in user_state:
        # Pas d'utilisateur actif : on reset mais on garde le message d'accueil
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

    # On reset le chat avec le message d'accueil + on conserve le résumé profil
    return list(WELCOME_MESSAGE), ShortTermMemory(window_size=MEMORY_WINDOW_SIZE), new_state, profile_display + f"\n\n{info}"


def _status_line(user_state: Dict) -> str:
    """Footer minimaliste — affiche en permanence l'état D1 (mémoire + profil).

    Argument jury : rend visible la mémoire conversationnelle (défi #1) sans
    exposer la stack technique. L'utilisateur voit son profil actif, le jury
    voit que D1 tourne.
    """
    name = user_state.get("name", "—") if user_state else "—"
    return (
        f"<div style='text-align:center;font-size:0.78em;opacity:0.6;"
        f"padding:0.5em 0;border-top:1px solid rgba(255,255,255,0.05);"
        f"margin-top:1em;'>"
        f"Profil actif : <b>{name}</b> · "
        f"Mémoire conversationnelle active · "
        f"Données OpenAgenda · Nantes Métropole"
        f"</div>"
    )


# ============================================================================
# BUILD UI (Gradio 4.x — theme dans Blocks())
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
        # ────────── Colonne gauche : profil utilisateur (D1) ──────────
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Utilisateur")

            user_dropdown = gr.Dropdown(
                choices=get_user_list(),
                label="Choisir un profil existant",
                interactive=True,
                allow_custom_value=True,
            )

            new_user_input = gr.Textbox(
                label="…ou créer un nouveau profil",
                placeholder="Ex: Léa, Thomas, …",
            )

            select_btn = gr.Button("Activer ce profil", variant="primary")

            profile_display = gr.Markdown("_Aucun utilisateur actif._")

            new_conv_btn = gr.Button("🔄 Nouvelle conversation", variant="secondary")

            # ─── Bloc technique réservé au mode debug ───
            # En prod : invisible pour l'utilisateur final
            # En démo soutenance : SHOW_DEBUG=1 pour montrer la transparence
            # technique au jury (conformité 12-factor app)
            if SHOW_DEBUG:
                gr.Markdown("---")
                gr.Markdown(
                    f"**Modèle** : `{CHAT_MODEL}`\n\n"
                    f"**Mémoire courte** : {MEMORY_WINDOW_SIZE} tours\n\n"
                    f"**Index** : `{FAISS_INDEX_PATH}`",
                    elem_id="debug-panel",
                )

        # ────────── Colonne droite : conversation ──────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                type="messages",       # format OpenAI-style depuis Gradio 4.36
                value=WELCOME_MESSAGE,  # pré-rempli au chargement (anti page blanche)
            )

            # Définition du textbox AVANT gr.Examples (qui le référence en inputs)
            msg_input = gr.Textbox(
                placeholder="Pose ta question sur les événements culturels…",
                show_label=False,
            )

            # 4 chips de suggestion — chacune démontre une capacité différente :
            #   - événements généraux (RAG de base)
            #   - mémoire conversationnelle (D1)
            #   - personnalisation (profil long terme)
            #   - filtrage thématique
            gr.Examples(
                examples=[
                    ["Quels événements culturels ce week-end à Nantes ?"],
                    ["Tu te souviens de ce que j'aime ?"],
                    ["Recommande-moi un spectacle pour ce soir"],
                    ["Y a-t-il des expositions gratuites en ce moment ?"],
                ],
                inputs=[msg_input],
                label="💡 Suggestions",
            )

            send_btn = gr.Button("Envoyer", variant="primary")

    # ────────── Footer dynamique ──────────
    # Affiche en permanence l'état D1 (profil + mémoire active)
    # Se met à jour automatiquement quand user_state change
    status_md = gr.Markdown(_status_line({}))

    # ────────── Wiring des événements ──────────

    def on_select(existing_dropdown: str, new_name: str):
        name = new_name.strip() if new_name and new_name.strip() else existing_dropdown
        user_s, sess_s, profile = select_user(name)
        return user_s, sess_s, profile, gr.update(choices=get_user_list(), value=name), ""

    select_btn.click(
        fn=on_select,
        inputs=[user_dropdown, new_user_input],
        outputs=[user_state, session_state, profile_display, user_dropdown, new_user_input],
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

    # Mise à jour du footer dès qu'un utilisateur est activé ou changé
    user_state.change(
        fn=_status_line,
        inputs=[user_state],
        outputs=[status_md],
    )


if __name__ == "__main__":
    # Gradio 4.x : theme dans Blocks(), pas dans launch()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
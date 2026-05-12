"""
app.py
Puls-Events MVP P13 - Interface Gradio.

Vague 1 : RAG basique + mémoire conversationnelle D1
    - Court terme : buffer fenêtré injecté dans prompt
    - Long terme : profils utilisateurs persistés sur Supabase Postgres
    - Extraction préférences en fin de conversation via Mistral Small

Pas encore dans cette version :
    - D2 géo (vague 2)
    - D3 agent web (vague 3)
    - D4 monitoring Langfuse (vague 3)
"""

from __future__ import annotations

# ============================================================================
# CHARGEMENT .env EN PREMIER — avant TOUT autre import qui lit os.getenv()
# ============================================================================
import os
from pathlib import Path
from dotenv import load_dotenv

# Chemin absolu vers le .env : à côté de app.py, quel que soit le CWD
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, verbose=True)

# Sanity check immédiat (sans crash, juste avertissement)
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
# STARTUP — chargement singletons (une seule fois au démarrage)
# ============================================================================

config_errors = validate_config()
if config_errors:
    logger.error("Erreurs de configuration au démarrage :")
    for err in config_errors:
        logger.error(f"  - {err}")
    logger.error("Vérifie tes variables d'environnement (.env local ou Secrets HF)")

# Vector store
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

# Long-term memory (Supabase)
logger.info("Connexion à Supabase Postgres…")
try:
    LTM = LongTermMemory(database_url=DATABASE_URL)
    logger.info("Mémoire long terme prête")
except Exception as e:
    logger.error(f"Échec connexion Supabase : {e}")
    LTM = None

# LLM principal
logger.info("Initialisation LLM Mistral…")
LLM = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model=CHAT_MODEL,
    temperature=0.3,
) if MISTRAL_API_KEY else None


# ============================================================================
# LOGIQUE MÉTIER
# ============================================================================

def rag_response(
    user_message: str,
    short_term: ShortTermMemory,
    user_id: int | None,
) -> str:
    """Pipeline RAG simple : retrieval + prompt + génération."""
    if VECTOR_STORE is None or LLM is None:
        return "⚠️ Erreur de configuration. Vérifie MISTRAL_API_KEY et l'index FAISS."

    docs = VECTOR_STORE.similarity_search(user_message, k=RETRIEVER_K)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

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
    return response.content


def trigger_preference_extraction(
    short_term: ShortTermMemory,
    user_id: int,
    session_id: int,
) -> int:
    """Extrait les préférences depuis l'historique et les persiste."""
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
# UI GRADIO 6.x (format messages OpenAI-style)
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
    """
    Handler du chat — format Gradio 6.x messages OpenAI-style.

    chat_history est une liste de {"role": "user|assistant", "content": "..."}
    """
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

    try:
        response = rag_response(message, short_term, user_id)
    except Exception as e:
        logger.error(f"Erreur RAG : {e}")
        response = f"⚠️ Erreur lors de la génération : {str(e)[:200]}"

    # Update mémoires
    short_term.add_turn(message, response)
    if LTM is not None and session_id:
        try:
            LTM.log_message(session_id, "user", message)
            LTM.log_message(session_id, "assistant", response)
        except Exception as e:
            logger.warning(f"Échec log message : {e}")

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return "", chat_history, short_term


def new_conversation(
    short_term: ShortTermMemory,
    user_state: Dict,
    session_state: Dict,
) -> Tuple[List[Dict], ShortTermMemory, Dict, str]:
    """Termine la session : extrait prefs, persiste, démarre nouvelle session, clear UI."""
    if not user_state or "id" not in user_state:
        return [], ShortTermMemory(window_size=MEMORY_WINDOW_SIZE), {}, "Pas d'utilisateur actif."

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

    return [], ShortTermMemory(window_size=MEMORY_WINDOW_SIZE), new_state, profile_display + f"\n\n{info}"


# ============================================================================
# BUILD UI (Gradio 6.x compat)
# ============================================================================

PULS_THEME = gr.themes.Soft(
    primary_hue="rose",
    secondary_hue="indigo",
)

with gr.Blocks(title="Puls-Events MVP") as demo:
    gr.Markdown("# 🎭 Puls-Events MVP")
    gr.Markdown("_Assistant culturel conversationnel avec mémoire personnalisée_")

    short_term_state = gr.State(ShortTermMemory(window_size=MEMORY_WINDOW_SIZE))
    user_state = gr.State({})
    session_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Utilisateur (D1 mémoire)")

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

            gr.Markdown("---")
            gr.Markdown(
                f"**Modèle** : `{CHAT_MODEL}`\n\n"
                f"**Mémoire courte** : {MEMORY_WINDOW_SIZE} tours\n\n"
                f"**Index** : `{FAISS_INDEX_PATH}`",
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Pose ta question sur les événements culturels…",
                    scale=4,
                    show_label=False,
                )
                send_btn = gr.Button("Envoyer", variant="primary", scale=1)

    # Handlers
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


if __name__ == "__main__":
    # Gradio 6.x : theme passé en launch() au lieu de Blocks()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=PULS_THEME,
    )
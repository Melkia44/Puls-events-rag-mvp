"""
utils/memory/profile.py
Extraction automatique des préférences utilisateur via Mistral Small.

Appelé en fin de conversation pour enrichir le profil long terme.
Sortie structurée JSON pour fiabilité.

Defi P13 D1 — extension long terme.
"""

from __future__ import annotations
import json
import logging
from typing import List, Optional, Dict, Any

from langchain_mistralai import ChatMistralAI

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """Tu es un assistant qui analyse une conversation entre un utilisateur et un chatbot d'événements culturels.

Ton objectif : extraire UNIQUEMENT les préférences PERSISTANTES de l'utilisateur (pas les questions ponctuelles).

Tu réponds STRICTEMENT en JSON valide avec ce schéma (pas de texte avant ou après) :
{{
  "thematique": ["..."],
  "lieu": ["..."],
  "moment": ["..."],
  "contrainte": ["..."]
}}

Règles :
- "thematique" : domaines récurrents évoqués (concerts, expositions, théâtre, jazz, jeune public, etc.)
- "lieu" : lieux ou zones géographiques mentionnés comme préférés (Nantes centre, Bouffay, etc.)
- "moment" : créneaux préférés (samedi soir, week-end, après-midi, en semaine, etc.)
- "contrainte" : exigences (gratuit, accessibilité PMR, adapté enfants, etc.)
- Liste vide [] si rien d'extractible pour une catégorie.
- 0 à 5 valeurs par catégorie. Ne sur-extrais pas.
- Mets en minuscules sauf noms propres.

Conversation à analyser :
{conversation}

JSON :"""


def _format_conversation(messages: List[Dict[str, str]]) -> str:
    lines = []
    for m in messages:
        role = "Utilisateur" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role} : {m.get('content', '').strip()}")
    return "\n".join(lines)


def extract_preferences(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str = "mistral-small-latest",
) -> Optional[Dict[str, List[str]]]:
    """
    Extrait les préférences utilisateur d'une conversation via LLM.

    Returns: dict avec clés thematique/lieu/moment/contrainte ou None si échec.
    """
    if not messages:
        return None

    conversation = _format_conversation(messages)
    if len(conversation.strip()) < 20:
        return None

    llm = ChatMistralAI(
        api_key=api_key,
        model=model,
        temperature=0.0,
        max_tokens=400,
    )

    prompt = EXTRACTION_PROMPT.format(conversation=conversation)

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Nettoyage backticks markdown
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()

        prefs = json.loads(raw)

        valid_keys = {"thematique", "lieu", "moment", "contrainte"}
        cleaned = {
            k: v for k, v in prefs.items()
            if k in valid_keys and isinstance(v, list)
        }

        # Suppression doublons + empty strings
        for k in list(cleaned.keys()):
            cleaned[k] = list({
                v.strip().lower() for v in cleaned[k]
                if isinstance(v, str) and v.strip()
            })

        return cleaned

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Extraction préférences : JSON invalide ({e})")
        return None
    except Exception as e:
        logger.error(f"Extraction préférences : erreur LLM ({e})")
        return None


def persist_extracted_preferences(
    user_id: int,
    extracted: Dict[str, List[str]],
    long_term_memory: Any,
    session_id: Optional[int] = None,
) -> int:
    """Persiste les préférences extraites. Retourne le nombre d'upserts."""
    count = 0
    for key, values in extracted.items():
        for value in values:
            if value:
                long_term_memory.upsert_preference(
                    user_id=user_id, key=key, value=value,
                    source_session_id=session_id,
                )
                count += 1
    return count

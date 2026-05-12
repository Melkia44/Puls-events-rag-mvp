"""
utils/memory — Mémoire conversationnelle multi-niveaux (defi P13 D1).

Trois étages :
    - short_term : ConversationBufferWindowMemory minimaliste (fenêtre N tours).
    - long_term : SQLAlchemy + Postgres Supabase (profil utilisateur persisté).
    - profile : extraction LLM des préférences en fin de conversation.
"""

from .short_term import ShortTermMemory, Turn
from .long_term import LongTermMemory
from .profile import extract_preferences, persist_extracted_preferences

__all__ = [
    "ShortTermMemory",
    "Turn",
    "LongTermMemory",
    "extract_preferences",
    "persist_extracted_preferences",
]

"""
utils/config.py
Constantes globales lues depuis les variables d'environnement.
"""

from __future__ import annotations
import os
from pathlib import Path

# ============================================================================
# LLM
# ============================================================================

CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral-large-latest")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "mistral-small-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# ============================================================================
# Vector store FAISS
# ============================================================================

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))

# ============================================================================
# Mémoire
# ============================================================================

MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "5"))
DATABASE_URL = os.getenv("DATABASE_URL")

# ============================================================================
# Validation startup
# ============================================================================

def validate_config() -> list[str]:
    """Retourne la liste des erreurs de config. Vide si tout OK."""
    errors = []
    if not MISTRAL_API_KEY:
        errors.append("MISTRAL_API_KEY manquant")
    if not DATABASE_URL:
        errors.append("DATABASE_URL manquant")
    if not Path(FAISS_INDEX_PATH).exists():
        errors.append(f"Index FAISS introuvable : {FAISS_INDEX_PATH}")
    return errors

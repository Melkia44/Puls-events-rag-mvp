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

# [D3 v2] Pool de candidats pour le filtre Haversine post-retrieval (D2).
# similarity_search est géo-aveugle : un pool trop petit ne contient pas
# assez de docs géo-locaux sur un corpus multi-villes (top-8). Mesure
# empirique (tests/check_strict_distribution.py) : 100 ramène la majorité
# des villes en RAG nominal. Coût nul (haversine Python, zéro appel LLM).
# Source unique de vérité : app.py ET les scripts de mesure la lisent ici.
RETRIEVER_K_GEO = int(os.getenv("RETRIEVER_K_GEO", "100"))

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

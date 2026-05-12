"""
utils/vector_store.py
Chargement du vector store FAISS depuis disque.

L'index est sérialisé en pickle (LangChain FAISS) avec embeddings Mistral.
Nécessite allow_dangerous_deserialization=True (héritage P11, à corriger en v2).
"""

from __future__ import annotations
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings

logger = logging.getLogger(__name__)


def load_vector_store(index_path: str, embedding_model: str = "mistral-embed",
                      api_key: str | None = None) -> FAISS:
    """
    Charge le vector store FAISS depuis le disque.

    Args:
        index_path: chemin du dossier contenant index.faiss + index.pkl
        embedding_model: nom du modèle d'embedding Mistral
        api_key: clé API Mistral (sinon lue depuis env)

    Returns:
        Instance FAISS LangChain prête à l'emploi.

    Raises:
        FileNotFoundError: si l'index n'existe pas
    """
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Index FAISS introuvable : {index_path}")

    if not (path / "index.faiss").exists():
        raise FileNotFoundError(f"Fichier index.faiss manquant dans {index_path}")

    logger.info(f"Chargement de l'index FAISS depuis {index_path}…")

    embeddings = MistralAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
    )

    vector_store = FAISS.load_local(
        folder_path=str(path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,  # héritage P11
    )

    logger.info("Index FAISS chargé avec succès")
    return vector_store

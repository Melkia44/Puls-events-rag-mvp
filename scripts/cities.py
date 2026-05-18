"""
scripts/cities.py
Définition du TOP-8 des grandes métropoles françaises couvertes par
le MVP Puls-Events.

Pourquoi un TOP-8 et pas tout OpenAgenda national :
    - Démontre le pivot multi-villes sans imposer la migration Qdrant
    - Volume estimé ~11-13k vecteurs (cap 30k pour FAISS plat OK)
    - Représentativité géographique Nord/Est/Sud/Ouest + Paris
    - Ingestion gérable en 25-50 min (vs 1-2h pour 15+ villes)

Choix éditorial assumé :
    Sélection sur axes géographiques (couverture France), pas top
    démographique pur. Strasbourg passe devant Nice (couverture Est).
    Bordeaux passe devant Montpellier (axe Sud-Ouest plus représentatif).

Critère de bascule V2 (US-205) :
    Quand on passera au national complet, ce fichier ne servira plus.
    L'index Qdrant sera alimenté par un cron OpenAgenda quotidien
    sans filtre `location_city`.

Données :
    - name : nom usuel (utilisé pour location_city dans la requête ODS)
    - insee_code : code commune INSEE (utile en V2 pour cross-référencer
      avec d'autres datasets publics : INSEE, BAN, etc.)
    - lat, lng : coordonnées du centre-ville pré-calculées (Wikipedia)
      → évite un appel Nominatim au build, accélère l'ingestion
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    name: str
    insee_code: str
    lat: float
    lng: float


# ─── TOP-8 retenu pour le MVP industrialisé ──────────────────────────
# Ordre = ordre d'ingestion (les plus gros d'abord pour échouer vite si problème)
TOP8: list[City] = [
    City("Paris",      "75056", 48.8566,  2.3522),
    City("Lyon",       "69123", 45.7640,  4.8357),
    City("Marseille",  "13055", 43.2965,  5.3698),
    City("Toulouse",   "31555", 43.6043,  1.4437),
    City("Nantes",     "44109", 47.2184, -1.5536),   # ⭐ ancrage initial
    City("Bordeaux",   "33063", 44.8378, -0.5792),
    City("Lille",      "59350", 50.6292,  3.0573),
    City("Strasbourg", "67482", 48.5734,  7.7521),
]


# ─── Helpers ───────────────────────────────────────────────────────────

def get_city(name: str) -> City | None:
    """Retourne la City correspondant au nom (case-insensitive), ou None."""
    name_low = name.strip().lower()
    for c in TOP8:
        if c.name.lower() == name_low:
            return c
    return None


def list_names() -> list[str]:
    """Liste des noms (pour les logs et boucles d'ingestion)."""
    return [c.name for c in TOP8]

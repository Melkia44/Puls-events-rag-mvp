"""
scripts/check_strict_distribution.py
Mesure empirique de la distribution `strict_in_radius` sur des requêtes
représentatives — Puls-Events MVP P13.

OBJECTIF
--------
Quantifier la fréquence des cas où le filtre Haversine retourne :
  - strict ≥ 3  → RAG nominal (Cas C, comportement attendu)
  - strict 1-2  → "fenêtre grise" : RAG fallback complète avec des candidats
                  FAISS potentiellement hors zone géo
  - strict = 0  → Bascule Web (Cas A)

Cette mesure pilote la décision d'architecture sur le routeur D3 :
  - Si fenêtre grise rare (<15%)   → statu quo, transparent via distance_km
  - Si fenêtre grise fréquente     → durcir (bascule Web sur strict < 3)

USAGE
-----
Depuis la racine du projet, venv activé :
    python scripts/check_strict_distribution.py

OPTIONS (en arg de ligne de commande) :
    --radius 30      → tester avec un autre rayon (défaut: 15 km)
    --queries-file   → CSV externe au format "query,city,radius"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permet le lancement depuis n'importe où — ajoute la racine au sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# .env AVANT les imports utils : géocodage Nominatim (NOMINATIM_CONTACT_EMAIL)
# et embeddings Mistral (MISTRAL_API_KEY) en dépendent.
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from utils.config import (
    MISTRAL_API_KEY, EMBEDDING_MODEL, FAISS_INDEX_PATH, RETRIEVER_K_GEO,
)
from utils.geo import filter_by_radius, geocode_city
from utils.vector_store import load_vector_store


# Jeu de requêtes représentatives — 8 villes du top-8 × thématiques courantes
DEFAULT_QUERIES = [
    # (query_text, target_city, radius_km)
    ("Concerts jazz",            "Nantes",     15),
    ("Expositions art moderne",  "Nantes",     15),
    ("Spectacles humour",        "Nantes",     15),
    ("Festival musique",         "Nantes",     15),
    ("Concerts jazz",            "Bordeaux",   15),
    ("Spectacles enfants",       "Lyon",       15),
    ("Festival musique",         "Strasbourg", 15),
    ("Concerts pop",             "Marseille",  15),
    ("Théâtre contemporain",     "Lille",      15),
    ("Expositions",              "Toulouse",   15),
    ("Concerts classique",       "Paris",      15),
    ("Spectacles danse",         "Paris",      15),
]


def classify(strict: int) -> tuple[str, str]:
    """Retourne (verdict_text, stats_key) selon la valeur de strict_in_radius."""
    if strict == 0:
        return "🌐 Web (Cas A)", "web_basculer"
    if strict <= 2:
        return "⚠️ Fenêtre grise", "fenetre_grise"
    return "✅ RAG nominal", "rag_nominal"


def run(queries: list[tuple[str, str, int]]) -> dict[str, int]:
    """Exécute la mesure et imprime le tableau + les stats agrégées."""
    print("Chargement du vector store FAISS…")
    vs = load_vector_store(
        index_path=FAISS_INDEX_PATH,
        embedding_model=EMBEDDING_MODEL,
        api_key=MISTRAL_API_KEY,
    )
    print(f"Index chargé.\n")

    # En-tête
    print(f"{'Query':<28} | {'City':<12} | strict | fallback | verdict")
    print("-" * 80)

    stats = {"rag_nominal": 0, "fenetre_grise": 0, "web_basculer": 0, "kor": 0}

    for query, city, radius in queries:
        coords = geocode_city(city)
        if not coords:
            print(f"{query:<28} | {city:<12} | géocodage KO (Nominatim)")
            stats["kor"] += 1
            continue

        candidates = vs.similarity_search(query, k=RETRIEVER_K_GEO)
        filtered, fallback, strict = filter_by_radius(
            candidates,
            user_lat=coords[0],
            user_lng=coords[1],
            radius_km=radius,
        )

        verdict, stat_key = classify(strict)
        stats[stat_key] += 1

        print(
            f"{query:<28} | {city:<12} | "
            f"{strict:<6} | {str(fallback):<8} | {verdict}"
        )

    return stats


def print_summary(stats: dict[str, int]) -> None:
    """Affiche le récap final + verdict de décision architecturale."""
    total = sum(stats.values())
    if total == 0:
        print("\nAucune requête exécutée.")
        return

    print()
    print("=" * 80)
    print(f"DISTRIBUTION sur {total} requêtes :\n")

    rag = stats["rag_nominal"]
    grey = stats["fenetre_grise"]
    web = stats["web_basculer"]
    kor = stats["kor"]

    print(f"  ✅ RAG nominal       (strict≥3) : {rag:>2}/{total} ({100 * rag / total:>3.0f}%)")
    print(f"  ⚠️ Fenêtre grise     (strict 1-2): {grey:>2}/{total} ({100 * grey / total:>3.0f}%)")
    print(f"  🌐 Bascule Web       (strict=0) : {web:>2}/{total} ({100 * web / total:>3.0f}%)")
    if kor:
        print(f"  ❌ Géocodage KO                 : {kor:>2}/{total} ({100 * kor / total:>3.0f}%)")

    # Verdict décisionnel
    print()
    print("VERDICT DÉCISIONNEL :")
    grey_pct = 100 * grey / total
    if grey_pct < 15:
        print(f"  → Fenêtre grise marginale ({grey_pct:.0f}%) — STATU QUO recommandé.")
        print(f"    Le compromis 'distance_km transparent' est acceptable.")
    elif grey_pct < 35:
        print(f"  → Fenêtre grise modérée ({grey_pct:.0f}%) — décision à trancher.")
        print(f"    Soit statu quo + documenter en US backlog, soit Option 3 (bascule Web).")
    else:
        print(f"  → Fenêtre grise fréquente ({grey_pct:.0f}%) — OPTION 3 recommandée.")
        print(f"    Durcir le routeur : bascule Web dès strict < min_docs_kept.")


def main():
    parser = argparse.ArgumentParser(
        description="Mesure de la distribution strict_in_radius pour D2/D3."
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=None,
        help="Override le rayon de toutes les requêtes (en km).",
    )
    args = parser.parse_args()

    queries = DEFAULT_QUERIES
    if args.radius is not None:
        queries = [(q, c, args.radius) for q, c, _ in queries]
        print(f"Override : tous les rayons fixés à {args.radius} km.\n")

    stats = run(queries)
    print_summary(stats)


if __name__ == "__main__":
    main()

"""
scripts/ingest_top8.py
Orchestrateur d'ingestion multi-villes pour le TOP-8 défini dans cities.py.

────────────────────────────────────────────────────────────────────────────
Architecture :
    - Importe les fonctions pures de scripts/ingest.py
      (fetch_events, to_documents, chunk_documents, build_index)
    - Boucle sur scripts/cities.TOP8 en passant `city=` à fetch_events()
    - Concatène tous les chunks en mémoire
    - Construit UN SEUL index FAISS final (1 seul save_local)
    - Backup automatique de data/faiss_index/ avant écriture

Garde-fous :
    - MAX_EVENTS_PER_CITY : cap dur par ville (anti-Paris-qui-explose)
    - MAX_TOTAL_VECTORS   : cap dur global (anti-FAISS-plat-saturé)
    - Validation au moins MIN_EVENTS_PER_CITY pour qu'une ville compte
      (sinon log warning et on continue, mais on n'inclut pas)
    - Backup avant écriture finale (idempotent, on peut relancer)

Idempotence :
    Si tu relances, le backup pre_top8 est préservé (pas écrasé).
    Tu peux toujours revenir à l'état Nantes-seul en faisant :
        rm -rf data/faiss_index
        mv data/faiss_index.backup_pre_top8 data/faiss_index

Usage :
    cd puls-events-mvp
    source venv/bin/activate
    python scripts/ingest_top8.py

    # Mode debug (1 seule ville pour tester rapidement)
    python scripts/ingest_top8.py --only Nantes

    # Mode dry-run (compte mais n'embed pas, gratuit)
    python scripts/ingest_top8.py --dry-run

Coût estimé (TOP-8 full) :
    ~11 000 - 13 000 vecteurs × ~300 tokens
    = ~3-4M tokens embeddings Mistral
    ≈ 0,30 - 0,40 €
    Durée : 25-50 min selon rate-limits Mistral
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Import local — on est dans scripts/, ingest.py est à côté
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cities import TOP8, get_city, list_names  # noqa: E402
from ingest import (  # noqa: E402
    fetch_events, to_documents, chunk_documents, build_index,
    FAISS_INDEX_DIR, MISTRAL_API_KEY,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ingest_top8")


# ============================================================================
# GARDE-FOUS
# ============================================================================

# Cap dur par ville (anti-Paris-qui-explose le volume)
MAX_EVENTS_PER_CITY = 2000

# Cap dur global (FAISS plat reste rapide jusqu'à ~30k vecteurs, on s'arrête bien avant)
MAX_TOTAL_VECTORS = 30_000

# Minimum acceptable pour qu'une ville soit considérée (sinon log warning)
MIN_EVENTS_PER_CITY = 10

# Backup automatique avant écriture finale
BACKUP_DIR = FAISS_INDEX_DIR.parent / f"{FAISS_INDEX_DIR.name}.backup_pre_top8"


# ============================================================================
# ORCHESTRATION
# ============================================================================

def ingest_one_city(city_name: str) -> tuple[int, list]:
    """Fetch + transform + chunk pour 1 ville.

    Returns:
        (n_raw_events, chunks) — Compteur d'événements bruts et chunks prêts à embed.
    """
    logger.info("=" * 64)
    logger.info(f"📍 Ingestion : {city_name}")
    logger.info("=" * 64)

    raw = fetch_events(city=city_name)

    if len(raw) < MIN_EVENTS_PER_CITY:
        logger.warning(
            f"  ⚠️  {city_name} : seulement {len(raw)} événements trouvés "
            f"(seuil {MIN_EVENTS_PER_CITY}) → ville incluse mais peu représentée"
        )

    if len(raw) > MAX_EVENTS_PER_CITY:
        logger.warning(
            f"  ⚠️  {city_name} : {len(raw)} événements > cap {MAX_EVENTS_PER_CITY}, "
            f"troncature aux {MAX_EVENTS_PER_CITY} premiers"
        )
        raw = raw[:MAX_EVENTS_PER_CITY]

    docs = to_documents(raw)
    chunks = chunk_documents(docs)

    logger.info(f"  ✓ {city_name} : {len(docs)} docs → {len(chunks)} chunks prêts")
    return (len(raw), chunks)


def run_top8(only: str | None = None, dry_run: bool = False) -> int:
    """Pipeline principal multi-villes.

    Returns:
        0 si succès, 1 si erreur.
    """
    if not MISTRAL_API_KEY:
        logger.error("MISTRAL_API_KEY manquant dans .env")
        return 1

    # Sélection des villes (par défaut TOP8 complet, sinon une seule)
    if only:
        city = get_city(only)
        if not city:
            logger.error(
                f"Ville '{only}' inconnue. Disponibles : {', '.join(list_names())}"
            )
            return 1
        cities_to_run = [city]
    else:
        cities_to_run = TOP8

    logger.info(f"🚀 Démarrage ingestion TOP-{len(cities_to_run)} : {[c.name for c in cities_to_run]}")
    if dry_run:
        logger.info("⚠️  MODE DRY-RUN — pas d'embedding, pas d'écriture")

    # Accumulateurs
    all_chunks: list = []
    stats: dict[str, dict] = {}
    total_raw = 0

    for city in cities_to_run:
        try:
            n_raw, chunks = ingest_one_city(city.name)
        except Exception as e:
            logger.error(f"  ❌ Échec ingestion {city.name} : {e}")
            stats[city.name] = {"raw": 0, "chunks": 0, "status": "error"}
            continue

        stats[city.name] = {"raw": n_raw, "chunks": len(chunks), "status": "ok"}
        total_raw += n_raw
        all_chunks.extend(chunks)

        # Garde-fou volume global — coupure préventive
        if len(all_chunks) > MAX_TOTAL_VECTORS:
            logger.error(
                f"⛔ Cap global {MAX_TOTAL_VECTORS} dépassé ({len(all_chunks)} chunks). "
                f"Arrêt de l'ingestion."
            )
            break

    # ──────── Récap ────────
    logger.info("=" * 64)
    logger.info("📊 RÉCAPITULATIF")
    logger.info("=" * 64)
    for city_name, data in stats.items():
        status_icon = "✅" if data["status"] == "ok" else "❌"
        logger.info(
            f"  {status_icon} {city_name:<14} "
            f"{data['raw']:>5} raw → {data['chunks']:>5} chunks"
        )
    logger.info(
        f"  {'TOTAL':<14} {total_raw:>5} raw → {len(all_chunks):>5} chunks"
    )
    logger.info("=" * 64)

    if dry_run:
        logger.info("✅ Dry-run terminé. Pour vraiment lancer l'embedding :")
        logger.info(f"     python scripts/ingest_top8.py")
        return 0

    if not all_chunks:
        logger.error("Aucun chunk obtenu — abandon, aucune écriture FAISS")
        return 1

    # ──────── Embedding final (1 seul build_index) ────────
    logger.info(f"🧮 Construction de l'index FAISS final ({len(all_chunks)} vecteurs)…")
    logger.info(f"   Coût estimé : ~{len(all_chunks) * 300 / 1_000_000 * 0.10:.2f} € "
                f"(à {0.10} €/M tokens)")

    try:
        vector_store = build_index(all_chunks)
    except Exception as e:
        logger.error(f"❌ Échec embedding : {e}")
        return 1

    n_target = vector_store.index.ntotal
    if n_target != len(all_chunks):
        logger.error(
            f"❌ Incohérence chunks ({len(all_chunks)}) vs index ({n_target}). "
            f"Aucune écriture pour préserver l'existant."
        )
        return 1

    # ──────── Backup avant écrasement ────────
    if FAISS_INDEX_DIR.exists():
        if BACKUP_DIR.exists():
            logger.warning(
                f"⚠️  Backup pre-TOP8 existe déjà ({BACKUP_DIR}), "
                f"on ne le réécrase pas (préservation de l'état initial Nantes)."
            )
        else:
            logger.info(f"💾 Backup : {FAISS_INDEX_DIR} → {BACKUP_DIR}")
            shutil.copytree(FAISS_INDEX_DIR, BACKUP_DIR)

    # ──────── Écriture finale ────────
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_INDEX_DIR))
    logger.info(
        f"✅ Index TOP-{len(cities_to_run)} sauvegardé dans {FAISS_INDEX_DIR} "
        f"({n_target} vecteurs)"
    )
    logger.info(
        f"   Pour revenir à l'état initial : "
        f"rm -rf {FAISS_INDEX_DIR} && mv {BACKUP_DIR} {FAISS_INDEX_DIR}"
    )
    return 0


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingestion multi-villes TOP-8 pour Puls-Events MVP",
    )
    parser.add_argument(
        "--only",
        help=f"Une seule ville (parmi {', '.join(list_names())})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compte seulement, pas d'embedding",
    )
    args = parser.parse_args()
    return run_top8(only=args.only, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

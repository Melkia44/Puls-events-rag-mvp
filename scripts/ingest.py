"""
scripts/ingest.py
Pipeline d'ingestion → indexation FAISS pour Puls-Events MVP P13.

Récupère les événements OpenAgenda via l'API publique OpenDataSoft
(dataset `evenements-publics-openagenda`), les transforme en Documents
LangChain enrichis de coordonnées GPS, puis les vectorise via Mistral.

────────────────────────────────────────────────────────────────────────────
Différence clé vs. ingestion P11 :
    Les champs `lat` et `lng` issus de `location_coordinates` ODS sont
    désormais persistés dans la metadata de chaque Document, ce qui permet
    au filtre Haversine de utils/geo.py de faire son travail.

[v2 — TOP-8] : `fetch_events()` accepte désormais un argument `city`
optionnel pour permettre l'orchestration multi-villes via
`scripts/ingest_top8.py` sans dupliquer la logique. Si non fourni,
la variable d'environnement GEO_CITY (défaut "Nantes") est utilisée.

Usage :
    cd puls-events-mvp
    source venv/bin/activate
    python scripts/ingest.py                    # Nantes (rétro-compatible)
    GEO_CITY=Paris python scripts/ingest.py     # autre ville

Variables d'environnement (.env) :
    MISTRAL_API_KEY        — requis
    OPENDATASOFT_API_KEY   — optionnel (rate-limit plus large si fourni)
    GEO_CITY               — défaut "Nantes"
    FRESHNESS_DAYS         — défaut 365
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# .env du projet (parent du dossier scripts/)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENDATASOFT_API_KEY = os.getenv("OPENDATASOFT_API_KEY", "")

GEO_CITY = os.getenv("GEO_CITY", "Nantes").strip()
FRESHNESS_DAYS = int(os.getenv("FRESHNESS_DAYS", "365"))

FAISS_INDEX_DIR = BASE_DIR / os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
BATCH_SIZE = 50

ODS_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1"
    "/catalog/datasets/evenements-publics-openagenda/records"
)


# ============================================================================
# FETCH ODS
# ============================================================================
def fetch_events(city: Optional[str] = None) -> List[dict]:
    """Récupère tous les événements ODS pour `city` avec pagination.

    [v2 — TOP-8] : argument `city` optionnel ajouté. Si None, utilise
    GEO_CITY (variable d'environnement, défaut "Nantes"). Cette signature
    rétro-compatible permet à scripts/ingest_top8.py d'orchestrer plusieurs
    ingestions sans dupliquer le code de fetch.
    """
    city_used = city or GEO_CITY
    start_date = (datetime.now() - timedelta(days=FRESHNESS_DAYS)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    where = (
        f"location_city='{city_used}' AND "
        f"lastdate_end >= '{today}' AND "
        f"firstdate_begin >= '{start_date}'"
    )
    logger.info(f"Fetch ODS — ville={city_used}, depuis {start_date}")
    all_events: List[dict] = []
    offset = 0
    limit = 100  # max ODS
    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "where": where,
            "order_by": "firstdate_begin ASC",
        }
        if OPENDATASOFT_API_KEY:
            params["apikey"] = OPENDATASOFT_API_KEY
        try:
            r = requests.get(ODS_URL, params=params, timeout=15)
            r.raise_for_status()
            results = r.json().get("results", [])
        except requests.RequestException as e:
            logger.error(f"Erreur ODS offset={offset} : {e}")
            break
        if not results:
            break
        all_events.extend(results)
        logger.info(f"  offset {offset} → +{len(results)} (total {len(all_events)})")
        if len(results) < limit:
            break
        offset += limit
    logger.info(f"Récupération terminée : {len(all_events)} événements bruts pour {city_used}")
    return all_events


# ============================================================================
# TRANSFORM
# ============================================================================
def _extract_coords(event: dict) -> tuple[Optional[float], Optional[float]]:
    """Extrait (lat, lng) depuis location_coordinates ODS.

    Format ODS : {"lon": -1.5563, "lat": 47.2128} (peut être None).
    """
    coords = event.get("location_coordinates")
    if not coords or not isinstance(coords, dict):
        return (None, None)
    lat = coords.get("lat")
    lng = coords.get("lon")
    try:
        return (float(lat), float(lng)) if lat is not None and lng is not None else (None, None)
    except (TypeError, ValueError):
        return (None, None)


def to_documents(raw_events: List[dict]) -> List[Document]:
    """Convertit les événements bruts en Documents LangChain enrichis."""
    docs: List[Document] = []
    skipped_empty = 0
    no_coords = 0
    for ev in raw_events:
        title = (ev.get("title_fr") or "").strip()
        if not title:
            continue
        # page_content = description longue ou courte, selon disponibilité
        desc_long = (ev.get("longdescription_fr") or "").strip()
        desc_short = (ev.get("description_fr") or "").strip()
        conditions = (ev.get("conditions_fr") or "").strip()
        # Concaténation lisible (sans HTML), le splitter découpera ensuite
        parts = []
        if desc_long:
            # On retire le HTML basique — pour Mistral, du texte propre vaut mieux
            import re
            desc_long_clean = re.sub(r"<[^>]+>", " ", desc_long)
            desc_long_clean = re.sub(r"\s+", " ", desc_long_clean).strip()
            parts.append(desc_long_clean)
        elif desc_short:
            parts.append(desc_short)
        if conditions:
            parts.append(f"Infos : {conditions}")
        page_content = " — ".join(parts).strip()
        if len(page_content) < 5:
            skipped_empty += 1
            continue
        lat, lng = _extract_coords(ev)
        if lat is None:
            no_coords += 1
        # Location formatée : "Nom (Adresse, Ville)"
        loc_name = (ev.get("location_name") or "").strip()
        loc_addr = (ev.get("location_address") or "").strip()
        loc_city = (ev.get("location_city") or "").strip()
        location = loc_name
        if loc_addr or loc_city:
            location += f" ({loc_addr}{', ' if loc_addr and loc_city else ''}{loc_city})"
        metadata = {
            "title": title,
            "start_date": ev.get("firstdate_begin") or "",
            "end_date": ev.get("lastdate_end") or "",
            "location": location or "Lieu non précisé",
            "url": ev.get("canonicalurl") or "",
            "uid": str(ev.get("uid") or ""),
            # [D2] Coordonnées GPS — la valeur ajoutée de cette ré-ingestion
            "lat": lat,
            "lng": lng,
            "city": loc_city,
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    logger.info(
        f"Transformés : {len(docs)} docs "
        f"(skipped_empty={skipped_empty}, no_coords={no_coords})"
    )
    return docs


# ============================================================================
# CHUNK + EMBED + SAVE
# ============================================================================
def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 5]
    logger.info(f"Chunking : {len(docs)} → {len(chunks)} chunks")
    return chunks


def build_index(chunks: List[Document]) -> FAISS:
    embeddings = MistralAIEmbeddings(
        api_key=MISTRAL_API_KEY,
        model=EMBEDDING_MODEL,
    )
    vector_store: Optional[FAISS] = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        logger.info(f"  Embedding lot {i}–{i + len(batch)} / {len(chunks)}")
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)
            time.sleep(0.5)  # rate-limit Mistral
    assert vector_store is not None
    return vector_store


# ============================================================================
# MAIN
# ============================================================================
def main() -> int:
    if not MISTRAL_API_KEY:
        logger.error("MISTRAL_API_KEY manquant dans .env")
        return 1
    raw = fetch_events()  # utilise GEO_CITY par défaut
    if not raw:
        logger.error("Aucun événement récupéré — abandon")
        return 1
    docs = to_documents(raw)
    if not docs:
        logger.error("Aucun document valide après transformation — abandon")
        return 1
    n_with_coords = sum(1 for d in docs if d.metadata.get("lat") is not None)
    logger.info(f"Coords disponibles sur {n_with_coords}/{len(docs)} documents")
    chunks = chunk_documents(docs)
    if not chunks:
        logger.error("Aucun chunk valide — abandon")
        return 1
    vector_store = build_index(chunks)
    n_target = vector_store.index.ntotal
    if n_target != len(chunks):
        logger.error(
            f"Incohérence source ({len(chunks)}) vs index ({n_target}) — "
            f"index NON sauvegardé"
        )
        return 1
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_INDEX_DIR))
    logger.info(f"Index sauvegardé dans {FAISS_INDEX_DIR} ({n_target} vecteurs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
utils/geo.py
Contexte géographique D2 — Puls-Events MVP P13.

Responsabilités :
    - Géocodage de noms de villes via Nominatim (OpenStreetMap)
    - Cache disque SQLite local pour respecter le rate-limit OSM (1 req/s)
    - Distance Haversine entre deux coordonnées
    - Extraction d'override géo en langage naturel ("à Saint-Nazaire",
      "dans un rayon de 30 km")
    - Filtrage post-retrieval d'une liste de Documents LangChain par rayon
      autour d'un point utilisateur

Choix techniques justifiés :
    - Nominatim plutôt que Google Geocoding : gratuit, souveraineté EU,
      qualité suffisante pour des noms de villes français
    - Cache SQLite plutôt que Redis : zéro infra, TTL 30 jours, ré-hydratation
      transparente entre sessions HF Spaces
    - Haversine post-retrieval plutôt que pré-filtrage spatial : préserve la
      pertinence sémantique du retrieval, latence acceptable (~10ms pour 20 docs)

Anti-patterns évités :
    - Pas d'appel Nominatim en boucle sans throttle (banni par OSM)
    - Pas de filtrage avant retrieval (perd la sémantique)
    - Pas de fallback silencieux si géocodage échoue (log + propage le None)

Réf. backlog : US-501 (recueil position), US-502 (filtre spatial),
               US-503 (tri par distance + affichage)
"""

from __future__ import annotations
import logging
import math
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================

# Endpoint public Nominatim — conformité aux conditions d'usage OSM :
# https://operations.osmfoundation.org/policies/nominatim/
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# User-Agent identifiant le projet (exigé par la politique Nominatim).
# Depuis 2025, OSM filtre agressivement les User-Agents contenant des emails
# manifestement bidon (@example.com, @test.com…) et renvoie 403. Le contact
# doit être un email réel et joignable, fourni via NOMINATIM_CONTACT_EMAIL
# dans .env. À défaut, on tombe sur un UA générique qui peut être 403.
_CONTACT_EMAIL = os.getenv("NOMINATIM_CONTACT_EMAIL", "").strip()
if _CONTACT_EMAIL:
    USER_AGENT = f"Puls-Events-MVP/1.0 ({_CONTACT_EMAIL})"
else:
    USER_AGENT = "Puls-Events-MVP/1.0"
    logger.warning(
        "NOMINATIM_CONTACT_EMAIL non défini — Nominatim peut refuser (403). "
        "Ajoute un email réel dans .env pour activer le géocodage D2."
    )

# Rate-limit Nominatim : 1 requête/seconde maximum (politique OSM)
RATE_LIMIT_DELAY_SECONDS = 1.1

# TTL du cache de géocodage en secondes (30 jours)
CACHE_TTL_SECONDS = 30 * 24 * 3600

# Rayon par défaut si aucun override n'est détecté (en kilomètres)
DEFAULT_RADIUS_KM = 15

# Rayon maximum acceptable par override (sanity check anti-abus)
MAX_RADIUS_KM = 200

# Rayon terrestre moyen utilisé par Haversine (en kilomètres)
EARTH_RADIUS_KM = 6371.0


# ============================================================================
# CACHE GÉOCODAGE — SQLite local
# ============================================================================

class GeocodingCache:
    """Cache disque pour les réponses Nominatim.

    On utilise SQLite plutôt qu'un dict en mémoire pour deux raisons :
      1. La cache survit au redémarrage du Space HF (cold start fréquent
         sur le tier free CPU Basic).
      2. Une seule connexion SQLite supporte plusieurs threads en lecture,
         ce qui colle au modèle Gradio (handlers concurrents).

    Le TTL évite que des coordonnées obsolètes restent indéfiniment.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS geocoding_cache (
                    query TEXT PRIMARY KEY,
                    lat REAL NOT NULL,
                    lng REAL NOT NULL,
                    cached_at INTEGER NOT NULL
                )
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, query: str) -> Optional[Tuple[float, float]]:
        """Retourne (lat, lng) si cache hit ET non expiré, sinon None."""
        norm = query.strip().lower()
        with self._conn() as c:
            row = c.execute(
                "SELECT lat, lng, cached_at FROM geocoding_cache WHERE query = ?",
                (norm,),
            ).fetchone()

        if row is None:
            return None

        lat, lng, cached_at = row
        if time.time() - cached_at > CACHE_TTL_SECONDS:
            logger.debug(f"Cache expiré pour '{query}'")
            return None

        return (lat, lng)

    def set(self, query: str, lat: float, lng: float) -> None:
        norm = query.strip().lower()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO geocoding_cache (query, lat, lng, cached_at) "
                "VALUES (?, ?, ?, ?)",
                (norm, lat, lng, int(time.time())),
            )


# Cache global module-level (singleton léger)
_CACHE: Optional[GeocodingCache] = None
_LAST_NOMINATIM_CALL: float = 0.0


def _get_cache() -> GeocodingCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = GeocodingCache("data/geocoding_cache.db")
    return _CACHE


# ============================================================================
# GEOCODAGE
# ============================================================================

def geocode_city(city: str, country_code: str = "fr") -> Optional[Tuple[float, float]]:
    """Convertit un nom de ville en (latitude, longitude) via Nominatim.

    Args:
        city: nom de ville libre (ex. "Nantes", "Saint-Nazaire", "44000")
        country_code: code pays ISO 3166-1 alpha-2 pour borner la recherche.
            Par défaut "fr" — adapter si Puls-Events s'étend hors France.

    Returns:
        (lat, lng) en degrés décimaux, ou None si introuvable / erreur API.

    Implémente :
        - Cache disque first (zéro appel réseau si déjà connu)
        - Throttling 1.1s entre deux appels Nominatim live
        - User-Agent conforme à la politique OSM
        - Timeout 5s pour ne pas bloquer Gradio
    """
    if not city or not city.strip():
        return None

    # 1. Cache lookup
    cache = _get_cache()
    cached = cache.get(city)
    if cached is not None:
        logger.debug(f"Cache hit géocodage : {city} → {cached}")
        return cached

    # 2. Throttling — respect du rate-limit OSM (1 req/s)
    global _LAST_NOMINATIM_CALL
    elapsed = time.time() - _LAST_NOMINATIM_CALL
    if elapsed < RATE_LIMIT_DELAY_SECONDS:
        time.sleep(RATE_LIMIT_DELAY_SECONDS - elapsed)

    # 3. Appel Nominatim
    params = {
        "q": city,
        "format": "json",
        "countrycodes": country_code,
        "limit": 1,
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=5)
        _LAST_NOMINATIM_CALL = time.time()
        resp.raise_for_status()
        results = resp.json()
    except requests.RequestException as e:
        logger.warning(f"Échec Nominatim pour '{city}' : {e}")
        return None

    if not results:
        logger.info(f"Aucun résultat Nominatim pour '{city}'")
        return None

    try:
        lat = float(results[0]["lat"])
        lng = float(results[0]["lon"])
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Réponse Nominatim mal formée pour '{city}' : {e}")
        return None

    # 4. Mise en cache et retour
    cache.set(city, lat, lng)
    logger.info(f"Géocodage '{city}' → ({lat:.4f}, {lng:.4f})")
    return (lat, lng)


# ============================================================================
# DISTANCE — Haversine
# ============================================================================

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Distance en km entre deux points sur la surface terrestre.

    Formule Haversine — approximation sphérique, erreur < 0.5 % sur Nantes
    Métropole (suffisant pour filtrer des événements culturels).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)

    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_KM * c


# ============================================================================
# EXTRACTION OVERRIDE LANGAGE NATUREL
# ============================================================================

# Regex pour capturer "à 25 km", "dans un rayon de 30 km", "10km", etc.
_RADIUS_REGEX = re.compile(
    r"(?:rayon\s+de\s+|à\s+|dans\s+|\b)(\d{1,4})\s*km",
    re.IGNORECASE,
)

# Regex pour capturer la ville cible dans le message utilisateur.
# Deux familles de patterns sont supportées :
#   1. Avec rayon explicite : "à 15 km de Nantes", "dans un rayon de 30 km de Lyon", "à 5km d'Angers"
#   2. Direct : "à Nantes", "vers Saint-Nazaire", "près de Rennes", "autour de Bordeaux"
# Le groupe de capture (1) reste le nom de la ville dans les deux cas.
_LOCATION_REGEX = re.compile(
    r"\b(?:"
        # Famille 1 — rayon explicite : "à 15 km de", "à 30 km d'", "à 5km d Angers".
        # Le connecteur élidé gère "de␣", "d'", "d'" (apostrophe typo) et "d␣".
        r"(?:à|dans(?:\s+un\s+rayon\s+de)?)\s+\d{1,3}\s*km\s+(?:de\s+|d['’ ]\s*)"
        # Famille 2 — prépositions directes (historique, espace géré ici).
        r"|(?:à|vers|sur|près\s+de|autour\s+de|aux?\s+alentours?\s+de)\s+"
    r")"
    r"([A-ZÀ-Ÿ][A-Za-zÀ-ÿ'\-]+(?:[\s\-][A-ZÀ-Ÿ][A-Za-zÀ-ÿ'\-]+){0,3})",
)

# Stop-list : verbes/mots à l'impératif fréquents qui ne sont pas des villes
# (le regex peut matcher "à Recommande-moi" → on filtre).
_LOCATION_STOPWORDS = {
    "recommande", "propose", "trouve", "cherche", "donne", "montre",
    "dis", "explique", "raconte", "moi", "nous", "vous",
}


def extract_radius_override(text: str) -> Optional[int]:
    """Extrait un rayon explicite depuis un message utilisateur.

    Exemples reconnus :
        "concerts dans un rayon de 30 km" → 30
        "à 5 km de chez moi" → 5
        "spectacles à 50km" → 50

    Returns:
        Rayon en km borné à [1, MAX_RADIUS_KM], ou None si rien trouvé.
    """
    if not text:
        return None

    match = _RADIUS_REGEX.search(text)
    if not match:
        return None

    try:
        radius = int(match.group(1))
    except (ValueError, IndexError):
        return None

    # Bornage anti-abus et anti-bug
    if radius < 1:
        return None
    return min(radius, MAX_RADIUS_KM)


def extract_location_override(text: str) -> Optional[str]:
    """Extrait un nom de ville depuis un message utilisateur.

    Exemples reconnus :
        "des concerts à Saint-Nazaire" → "Saint-Nazaire"
        "et vers Rennes ?" → "Rennes"
        "près de Nantes ce soir" → "Nantes"

    Returns:
        Nom de ville (chaîne brute, à géocoder ensuite), ou None.
    """
    if not text:
        return None

    match = _LOCATION_REGEX.search(text)
    if not match:
        return None

    candidate = match.group(1).strip()

    # Filtrage des faux positifs (impératifs en début de phrase)
    first_word = candidate.split()[0].lower() if candidate else ""
    if first_word in _LOCATION_STOPWORDS:
        return None

    # Si le candidat fait moins de 3 caractères, c'est probablement parasite
    if len(candidate) < 3:
        return None

    return candidate


# ============================================================================
# FILTRAGE POST-RETRIEVAL
# ============================================================================

def filter_by_radius(
    docs: list,
    user_lat: float,
    user_lng: float,
    radius_km: float = DEFAULT_RADIUS_KM,
    min_docs_kept: int = 3,
) -> Tuple[list, bool, int]:
    """Filtre une liste de Documents LangChain par distance Haversine.

    Chaque Document doit avoir `metadata['lat']` et `metadata['lng']` —
    sinon il est exclu silencieusement (et logué en debug).

    Args:
        docs: liste de Documents issus de FAISS (top-K élargi à 20 typiquement)
        user_lat, user_lng: position de l'utilisateur (en degrés décimaux)
        radius_km: rayon de filtrage
        min_docs_kept: seuil de garde-fou. Si après filtrage il reste moins
            que ce nombre de docs, on renvoie la liste NON filtrée (avec
            le flag fallback=True) pour ne pas frustrer l'utilisateur avec
            une réponse "rien trouvé".

    Returns:
        (documents_filtrés, fallback_activé, strict_in_radius)
        - documents_filtrés : trié par distance ascendante, avec
          metadata['distance_km'] ajouté (float arrondi à 1 décimale)
        - fallback_activé : True si le filtre a été désactivé faute de
          résultats suffisants. Sert à informer l'UI ("rayon trop strict").
        - strict_in_radius : nb de docs STRICTEMENT dans le rayon (avant
          tout fallback). 0 = aucun événement réel autour de la ville cible
          → permet à respond() de basculer vers le Web (Cas A, ville hors
          corpus top-8).
    """
    if not docs:
        return [], False, 0

    enriched = []
    skipped_no_coords = 0

    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        lat = meta.get("lat")
        lng = meta.get("lng")

        if lat is None or lng is None:
            skipped_no_coords += 1
            continue

        try:
            distance = haversine_km(user_lat, user_lng, float(lat), float(lng))
        except (TypeError, ValueError):
            skipped_no_coords += 1
            continue

        if distance <= radius_km:
            # On enrichit la metadata in-place pour que l'UI puisse afficher
            # la distance dans le panneau Sources. Effet de bord conscient :
            # le Document FAISS sous-jacent est partagé entre threads, mais
            # on n'écrit que dans son dict de metadata local.
            doc.metadata = {**meta, "distance_km": round(distance, 1)}
            enriched.append(doc)

    if skipped_no_coords:
        logger.debug(f"{skipped_no_coords} docs sans coordonnées exploitables")

    # Tri par distance ascendante (le plus proche en premier)
    enriched.sort(key=lambda d: d.metadata.get("distance_km", float("inf")))

    # [D3 v2] strict_count = nb de docs STRICTEMENT dans le rayon (post-Haversine),
    # exposé en 3e valeur de retour pour piloter le fallback Web (Cas A).
    strict_count = len(enriched)

    if strict_count >= min_docs_kept:
        return enriched, False, strict_count

    # Garde-fou — fallback si filtre trop strict : on renvoie la liste
    # candidate FAISS d'entrée NON filtrée (top-K élargi, ~20 docs — PAS
    # le vector store entier) pour l'UX "éviter rien trouvé", mais on
    # expose strict_count pour que respond() distingue "0 réel autour"
    # (→ Cas A Web) de "peu de résultats" (→ Cas C RAG dégradé).
    logger.warning(
        f"Filtre géo a gardé seulement {strict_count} docs "
        f"(seuil {min_docs_kept}), fallback activé"
    )
    return docs, True, strict_count

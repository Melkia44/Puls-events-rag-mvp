"""
tests/test_geo.py
Tests unitaires pour utils.geo — couvre Haversine, extraction override,
filtrage post-retrieval. Les tests réseau (Nominatim) sont marqués
@pytest.mark.network pour pouvoir les skipper en CI sans connexion.

Run :
    pytest tests/test_geo.py -v
    pytest tests/test_geo.py -v -m "not network"   # offline only
"""

from __future__ import annotations
import pytest
from types import SimpleNamespace

from utils.geo import (
    haversine_km,
    extract_radius_override,
    extract_location_override,
    filter_by_radius,
    geocode_city,
    DEFAULT_RADIUS_KM,
)


# ============================================================================
# Test 1 — Haversine : distances connues
# ============================================================================

class TestHaversine:
    """Valide la formule sur des couples connus."""

    def test_zero_distance(self):
        """Deux points identiques → 0 km."""
        d = haversine_km(47.2184, -1.5536, 47.2184, -1.5536)
        assert d == pytest.approx(0.0, abs=0.001)

    def test_nantes_saint_nazaire(self):
        """Nantes ↔ Saint-Nazaire ≈ 50 km à vol d'oiseau."""
        # Coordonnées approximatives des centres-villes
        nantes = (47.2184, -1.5536)
        saint_nazaire = (47.2735, -2.2138)
        d = haversine_km(*nantes, *saint_nazaire)
        # Tolérance large car centres-villes approximatifs
        assert 45 < d < 55, f"Distance Nantes-St-Nazaire = {d:.1f} km, attendu ~50"

    def test_nantes_paris(self):
        """Nantes ↔ Paris ≈ 343 km à vol d'oiseau."""
        d = haversine_km(47.2184, -1.5536, 48.8566, 2.3522)
        assert 340 < d < 350, f"Distance Nantes-Paris = {d:.1f} km, attendu ~343"

    def test_symmetric(self):
        """d(A, B) == d(B, A)."""
        d1 = haversine_km(47.2184, -1.5536, 48.8566, 2.3522)
        d2 = haversine_km(48.8566, 2.3522, 47.2184, -1.5536)
        assert d1 == pytest.approx(d2, abs=0.001)


# ============================================================================
# Test 2 — Extraction d'override de rayon
# ============================================================================

class TestRadiusOverride:
    """Couvre les formulations naturelles fréquentes."""

    @pytest.mark.parametrize("text,expected", [
        ("concerts dans un rayon de 30 km", 30),
        ("à 5 km de chez moi", 5),
        ("spectacles à 50km", 50),
        ("Trouve-moi des événements dans 10 km", 10),
        ("rayon de 25 km autour de Nantes", 25),
    ])
    def test_extracts_radius(self, text, expected):
        assert extract_radius_override(text) == expected

    @pytest.mark.parametrize("text", [
        "",
        None,
        "concerts à Nantes",                 # pas de km mentionné
        "spectacles ce soir",
        "à 0 km",                            # zéro est rejeté
    ])
    def test_no_override(self, text):
        assert extract_radius_override(text) is None

    def test_bornage_maximum(self):
        """Un rayon délirant est borné à MAX_RADIUS_KM."""
        result = extract_radius_override("dans un rayon de 9999 km")
        assert result is not None
        assert result <= 200  # MAX_RADIUS_KM


# ============================================================================
# Test 3 — Extraction de ville (override langage)
# ============================================================================

class TestLocationOverride:
    """Couvre les prépositions usuelles."""

    @pytest.mark.parametrize("text,expected", [
        ("Des concerts à Saint-Nazaire ce soir", "Saint-Nazaire"),
        ("et vers Rennes ?", "Rennes"),
        ("près de Nantes ce soir", "Nantes"),
        ("autour de Vannes le week-end", "Vannes"),
        ("aux alentours de La Baule", "La Baule"),
    ])
    def test_extracts_location(self, text, expected):
        assert extract_location_override(text) == expected

    @pytest.mark.parametrize("text", [
        "",
        None,
        "Recommande-moi un concert",          # impératif filtré
        "Trouve quelque chose ce soir",       # pas de préposition géo
        "à 5 km",                              # match numérique, pas ville
    ])
    def test_no_location(self, text):
        result = extract_location_override(text)
        assert result is None, f"Faux positif sur '{text}' → '{result}'"


# ============================================================================
# Test 4 — Filtrage post-retrieval Haversine
# ============================================================================

def _make_doc(title: str, lat: float | None, lng: float | None):
    """Construit un faux Document LangChain léger pour les tests."""
    return SimpleNamespace(
        page_content=f"Description de {title}",
        metadata={"title": title, "lat": lat, "lng": lng},
    )


class TestFilterByRadius:
    """Vérifie le filtrage, le tri par distance et le fallback."""

    def setup_method(self):
        # Point utilisateur = Nantes
        self.user_lat = 47.2184
        self.user_lng = -1.5536

        # Faux dataset : 4 événements à différentes distances de Nantes
        self.docs = [
            _make_doc("Pannonica (Nantes)",       47.2200, -1.5500),   # ~0.5 km
            _make_doc("Stereolux (Nantes)",       47.2050, -1.5450),   # ~1.7 km
            _make_doc("Salle VIP (St-Nazaire)",   47.2735, -2.2138),   # ~50 km
            _make_doc("Olympia (Paris)",          48.8700, 2.3300),    # ~343 km
        ]

    def test_filters_within_radius(self):
        """À 15 km autour de Nantes, on garde Pannonica + Stereolux."""
        kept, fallback = filter_by_radius(
            self.docs, self.user_lat, self.user_lng, radius_km=15,
        )
        # Min_docs_kept=3 par défaut → fallback car seulement 2 docs dans 15km
        assert fallback is True
        # Mais les docs filtrés sont quand même les bons
        titles = [d.metadata["title"] for d in kept]
        assert "Pannonica (Nantes)" in titles
        assert "Stereolux (Nantes)" in titles
        assert "Olympia (Paris)" not in titles

    def test_keeps_3_at_60km(self):
        """À 60 km, on garde 3 docs (Nantes x2 + St-Nazaire) sans fallback."""
        kept, fallback = filter_by_radius(
            self.docs, self.user_lat, self.user_lng,
            radius_km=60, min_docs_kept=3,
        )
        assert fallback is False
        assert len(kept) == 3
        # Tri par distance ascendante
        distances = [d.metadata["distance_km"] for d in kept]
        assert distances == sorted(distances)

    def test_distance_metadata_added(self):
        """Chaque doc gardé doit avoir un metadata['distance_km'] arrondi."""
        kept, _ = filter_by_radius(
            self.docs, self.user_lat, self.user_lng, radius_km=400,
        )
        for d in kept:
            assert "distance_km" in d.metadata
            assert isinstance(d.metadata["distance_km"], float)
            # Arrondi à 1 décimale
            assert d.metadata["distance_km"] == round(d.metadata["distance_km"], 1)

    def test_skips_docs_without_coords(self):
        """Un doc sans lat/lng est ignoré silencieusement."""
        docs = [
            _make_doc("Sans coords", None, None),
            _make_doc("Pannonica", 47.2200, -1.5500),
        ]
        kept, _ = filter_by_radius(
            docs, self.user_lat, self.user_lng, radius_km=10, min_docs_kept=1,
        )
        titles = [d.metadata["title"] for d in kept]
        assert "Sans coords" not in titles
        assert "Pannonica" in titles

    def test_fallback_when_zero_match(self):
        """Si aucun doc ne passe le filtre, on retourne TOUT en fallback."""
        kept, fallback = filter_by_radius(
            self.docs, self.user_lat, self.user_lng,
            radius_km=0.01,  # absurde
        )
        assert fallback is True
        assert len(kept) == len(self.docs)


# ============================================================================
# Test bonus — Géocodage live (skippable hors CI réseau)
# ============================================================================

@pytest.mark.network
class TestGeocodingLive:
    """Tests d'intégration Nominatim — nécessitent un accès réseau."""

    def test_geocode_nantes(self):
        result = geocode_city("Nantes")
        assert result is not None
        lat, lng = result
        # Centre Nantes : ~47.22 lat, ~-1.55 lng
        assert 47.0 < lat < 47.4
        assert -1.7 < lng < -1.4

    def test_geocode_unknown(self):
        result = geocode_city("XyzzzInexistantVille12345")
        assert result is None

    def test_geocode_cache_hit(self):
        """Un 2e appel sur la même ville doit être instantané (cache)."""
        import time
        # 1er appel — peut être long si pas en cache
        geocode_city("Saint-Nazaire")
        # 2e appel — doit toucher le cache, donc rapide
        t0 = time.time()
        result = geocode_city("Saint-Nazaire")
        elapsed = time.time() - t0
        assert result is not None
        assert elapsed < 0.1, f"Cache miss probable, {elapsed:.2f}s"

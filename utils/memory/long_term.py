"""
utils/memory/long_term.py
Mémoire conversationnelle LONG TERME — Postgres Supabase via SQLAlchemy.

Stocke utilisateurs, sessions, messages, préférences extraites.
Persisté entre sessions Gradio pour démontrer la mémoire continue.

Defi P13 D1 — niveau 2/2.

────────────────────────────────────────────────────────────────────────────
Changelog :
    [D2] Ajout des colonnes city / city_lat / city_lng dans la table users
         pour persister la position de l'utilisateur entre sessions.
         Migration faite via ALTER TABLE idempotent au démarrage (Postgres-only).
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, Float, ForeignKey,
    UniqueConstraint, Index, CheckConstraint, func, text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

logger = logging.getLogger(__name__)

Base = declarative_base()


# ============================================================================
# MODÈLES ORM
# ============================================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(80), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # [D2] Position utilisateur — saisie au profil, géocodée une fois
    # Nullable : un utilisateur peut exister sans ville renseignée
    city = Column(String(120), nullable=True)
    city_lat = Column(Float, nullable=True)
    city_lng = Column(Float, nullable=True)

    sessions = relationship(
        "ConversationSession", back_populates="user",
        cascade="all, delete-orphan",
    )
    preferences = relationship(
        "Preference", back_populates="user",
        cascade="all, delete-orphan",
    )


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="sessions")
    messages = relationship(
        "Message", back_populates="session",
        cascade="all, delete-orphan",
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("conversation_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    ts = Column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("ConversationSession", back_populates="messages")


class Preference(Base):
    __tablename__ = "preferences"
    __table_args__ = (
        UniqueConstraint("user_id", "key", "value", name="uniq_user_pref"),
        Index("idx_user_pref", "user_id", "weight"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String(40), nullable=False)
    value = Column(String(160), nullable=False)
    weight = Column(Integer, default=1, nullable=False)
    source_session_id = Column(Integer, nullable=True)
    ts = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="preferences")


class Feedback(Base):
    """Vote 👍/👎 d'un utilisateur sur une réponse (US-704, D4 CSAT).

    rating ∈ {1, -1} : 1 = pouce haut, -1 = pouce bas. Contrainte CHECK
    en base pour rejeter toute autre valeur même hors ORM.
    """
    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("rating IN (1, -1)", name="ck_feedback_rating"),
        Index("idx_feedback_session", "session_id"),
        Index("idx_feedback_user", "user_id"),
        Index("idx_feedback_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        Integer,
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    rating = Column(Integer, nullable=False)  # 1 = 👍, -1 = 👎
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# ============================================================================
# REPOSITORY (façade haut-niveau)
# ============================================================================

class LongTermMemory:
    """
    Façade simple pour l'accès aux profils utilisateur persistés sur Postgres.
    """

    def __init__(self, database_url: str):
        """
        Args:
            database_url: postgresql+psycopg2://... (Supabase Session Pooler)
        """
        if not database_url:
            raise ValueError("DATABASE_URL est requise pour LongTermMemory")

        # Config recommandée HF Spaces + Supabase Session Pooler
        self.engine = create_engine(
            database_url,
            pool_size=5,
            max_overflow=5,
            pool_pre_ping=True,    # détecte les connexions zombies après sleep HF
            pool_recycle=300,      # recycle après 5 min
            echo=False,
        )

        # Création du schéma au premier démarrage (idempotent)
        Base.metadata.create_all(self.engine)
        logger.info("Schéma DB vérifié/créé sur Supabase")

        # [D2] Migration idempotente — ajoute les colonnes ville si absentes
        # Postgres-spécifique (ADD COLUMN IF NOT EXISTS). SQLAlchemy ne fait
        # pas d'ALTER TABLE automatique, c'est une limite connue hors Alembic.
        # Coût démarrage : 3 requêtes catalogue Postgres = négligeable.
        self._migrate_user_geo_columns()

        self.SessionLocal = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False,
        )

    def _migrate_user_geo_columns(self) -> None:
        """Ajoute les colonnes city/city_lat/city_lng à users si elles manquent.

        Postgres 9.6+ supporte ADD COLUMN IF NOT EXISTS, donc l'opération
        est strictement idempotente et sans risque.
        """
        migrations = [
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS city VARCHAR(120)",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS city_lat DOUBLE PRECISION",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS city_lng DOUBLE PRECISION",
        ]
        try:
            with self.engine.begin() as conn:
                for sql in migrations:
                    conn.execute(text(sql))
            logger.info("Migration colonnes géo users : OK (idempotent)")
        except Exception as e:
            logger.error(f"Échec migration colonnes géo users : {e}")
            # On ne lève pas : si la migration échoue, l'app peut tourner
            # en dégradé (D1 ok, D2 inactif). Les setters/getters géo
            # géreront le cas "colonne absente" en levant proprement.

    @contextmanager
    def _session(self) -> Session:
        s = self.SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # --- Users ---

    def get_or_create_user(self, name: str) -> int:
        with self._session() as s:
            user = s.query(User).filter(User.name == name).one_or_none()
            if user is None:
                user = User(name=name)
                s.add(user)
                s.flush()
            return user.id

    def list_users(self) -> List[Dict]:
        with self._session() as s:
            users = s.query(User).order_by(User.name).all()
            return [{
                "id": u.id, "name": u.name,
                "created_at": u.created_at.isoformat(),
            } for u in users]

    # --- [D2] Géolocalisation utilisateur ---

    def set_user_city(
        self, user_id: int, city: str, lat: float, lng: float,
    ) -> None:
        """Persiste la ville et ses coordonnées géocodées pour un utilisateur.

        Args:
            user_id: identifiant interne
            city: nom de ville saisi par l'utilisateur (conservé tel quel pour
                l'affichage UI, pas normalisé)
            lat, lng: coordonnées issues du géocodage Nominatim
        """
        with self._session() as s:
            user = s.get(User, user_id)
            if user is None:
                logger.warning(f"set_user_city : user_id={user_id} introuvable")
                return
            user.city = city
            user.city_lat = lat
            user.city_lng = lng
            logger.info(f"Ville '{city}' enregistrée pour user_id={user_id}")

    def get_user_city(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retourne le dict {city, lat, lng} ou None si non renseigné."""
        with self._session() as s:
            user = s.get(User, user_id)
            if user is None or user.city is None:
                return None
            return {
                "city": user.city,
                "lat": user.city_lat,
                "lng": user.city_lng,
            }

    # --- Sessions ---

    def start_session(self, user_id: int) -> int:
        with self._session() as s:
            sess = ConversationSession(user_id=user_id)
            s.add(sess)
            s.flush()
            return sess.id

    def end_session(self, session_id: int) -> None:
        with self._session() as s:
            sess = s.get(ConversationSession, session_id)
            if sess:
                sess.ended_at = datetime.utcnow()

    # --- Messages ---

    def log_message(self, session_id: int, role: str, content: str) -> None:
        with self._session() as s:
            msg = Message(session_id=session_id, role=role, content=content)
            s.add(msg)

    # --- Lecture sessions (sidebar historique) ---

    def list_user_sessions(
        self,
        user_id: int,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Liste les dernières sessions d'un utilisateur pour la sidebar UI.

        Filtre les sessions vides (zéro message) — on n'affiche que celles
        qui contiennent au moins une vraie interaction.

        Args:
            user_id: identifiant de l'utilisateur
            limit: nombre max de sessions retournées (défaut 20)

        Returns:
            Liste de dicts triée par dernier message DESC (plus récente d'abord) :
                [{
                    "session_id": int,
                    "started_at": datetime,
                    "last_message_at": datetime,
                    "preview": str (1er message user, tronqué à 80 chars),
                    "msg_count": int,
                }, ...]
        """
        with self._session() as s:
            # Sessions de cet utilisateur — borne la sous-requête preview
            # à ses propres messages (sinon DISTINCT ON scanne la table
            # messages de TOUS les utilisateurs).
            user_session_ids = (
                s.query(ConversationSession.id)
                .filter(ConversationSession.user_id == user_id)
                .subquery()
            )

            # Sous-requête : 1er message user de chaque session (preview)
            first_msg_subq = (
                s.query(Message.session_id, Message.content)
                .filter(
                    Message.role == "user",
                    Message.session_id.in_(s.query(user_session_ids.c.id)),
                )
                .order_by(Message.session_id, Message.ts.asc())
                .distinct(Message.session_id)
                .subquery()
            )

            # Requête principale : sessions + agrégats messages
            results = (
                s.query(
                    ConversationSession.id,
                    ConversationSession.started_at,
                    func.max(Message.ts).label("last_message_at"),
                    func.count(Message.id).label("msg_count"),
                    first_msg_subq.c.content.label("preview"),
                )
                .join(Message, Message.session_id == ConversationSession.id)
                .outerjoin(
                    first_msg_subq,
                    first_msg_subq.c.session_id == ConversationSession.id,
                )
                .filter(ConversationSession.user_id == user_id)
                .group_by(
                    ConversationSession.id,
                    ConversationSession.started_at,
                    first_msg_subq.c.content,
                )
                .having(func.count(Message.id) > 0)
                .order_by(func.max(Message.ts).desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "session_id": r.id,
                    "started_at": r.started_at,
                    "last_message_at": r.last_message_at,
                    "preview": (r.preview or "")[:80].strip(),
                    "msg_count": r.msg_count,
                }
                for r in results
            ]

    def load_session_history(
        self,
        session_id: int,
    ) -> List[Dict[str, str]]:
        """Recharge l'historique complet d'une session pour réinjection UI.

        Args:
            session_id: identifiant de la session à charger

        Returns:
            Liste de messages dans l'ordre chronologique :
                [{"role": "user"|"assistant", "content": str, "ts": datetime}, ...]
        """
        with self._session() as s:
            msgs = (
                s.query(Message)
                .filter(Message.session_id == session_id)
                .order_by(Message.ts.asc())
                .all()
            )
            return [
                {"role": m.role, "content": m.content, "ts": m.ts}
                for m in msgs
            ]

    # --- Feedback (US-704, D4 CSAT) ---

    def add_feedback(self, session_id: int, user_id: int, rating: int) -> None:
        """Enregistre un vote 👍 (rating=1) ou 👎 (rating=-1) sur une réponse.

        Idempotence volontairement non gérée : on archive chaque clic
        (un utilisateur peut changer d'avis ; l'agrégat CSAT prendra le
        dernier état via la fenêtre temporelle si besoin). rating hors
        {1,-1} est rejeté en amont ET par la contrainte CHECK en base.
        """
        if rating not in (1, -1):
            logger.warning(f"add_feedback : rating invalide {rating!r}, ignoré")
            return
        with self._session() as s:
            s.add(Feedback(
                session_id=session_id, user_id=user_id, rating=rating,
            ))

    def get_csat(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Agrégat CSAT : 👍, 👎, total et score (% de 👍).

        Args:
            user_id: si fourni, restreint au périmètre d'un utilisateur ;
                sinon agrégat global (tous utilisateurs).

        Returns:
            {"thumbs_up": int, "thumbs_down": int, "total": int,
             "csat": float|None}  — csat = up / total (0..1), None si total=0.
        """
        with self._session() as s:
            q = s.query(
                func.count().filter(Feedback.rating == 1).label("up"),
                func.count().filter(Feedback.rating == -1).label("down"),
                func.count(Feedback.id).label("total"),
            )
            if user_id is not None:
                q = q.filter(Feedback.user_id == user_id)
            row = q.one()
            total = row.total or 0
            return {
                "thumbs_up": row.up or 0,
                "thumbs_down": row.down or 0,
                "total": total,
                "csat": round((row.up or 0) / total, 3) if total else None,
            }

    # --- Preferences ---

    def upsert_preference(
        self, user_id: int, key: str, value: str,
        source_session_id: Optional[int] = None,
    ) -> None:
        """Ajoute la préférence ou incrémente son poids si déjà existante."""
        with self._session() as s:
            pref = (s.query(Preference)
                    .filter(Preference.user_id == user_id,
                            Preference.key == key,
                            Preference.value == value)
                    .one_or_none())
            if pref:
                pref.weight += 1
                pref.ts = datetime.utcnow()
            else:
                s.add(Preference(
                    user_id=user_id, key=key, value=value,
                    source_session_id=source_session_id, weight=1,
                ))

    def get_user_preferences(
        self, user_id: int, min_weight: int = 1,
    ) -> Dict[str, List[Dict]]:
        """Retourne les préférences groupées par clé."""
        with self._session() as s:
            prefs = (s.query(Preference)
                     .filter(Preference.user_id == user_id,
                             Preference.weight >= min_weight)
                     .order_by(Preference.weight.desc()).all())

            grouped: Dict[str, List[Dict]] = {}
            for p in prefs:
                grouped.setdefault(p.key, []).append({
                    "value": p.value, "weight": p.weight,
                })
            return grouped

    def get_preference_summary(self, user_id: int) -> str:
        """Format texte lisible pour injection dans le prompt système."""
        prefs = self.get_user_preferences(user_id)
        if not prefs:
            return ""

        labels = {
            "thematique": "Thématiques préférées",
            "lieu": "Lieux fréquents",
            "moment": "Moments préférés",
            "contrainte": "Contraintes",
        }
        lines = []
        for key, items in prefs.items():
            label = labels.get(key, key)
            values = ", ".join(f"{i['value']} (×{i['weight']})" for i in items[:5])
            lines.append(f"- {label} : {values}")

        return "Profil de l'utilisateur :\n" + "\n".join(lines)

"""
utils/memory/long_term.py
Mémoire conversationnelle LONG TERME — Postgres Supabase via SQLAlchemy.

Stocke utilisateurs, sessions, messages, préférences extraites.
Persisté entre sessions Gradio pour démontrer la mémoire continue.

Defi P13 D1 — niveau 2/2.
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey,
    UniqueConstraint, Index,
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

        self.SessionLocal = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False,
        )

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

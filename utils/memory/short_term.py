"""
utils/memory/short_term.py
Mémoire conversationnelle COURT TERME — fenêtre des N derniers tours.
Injectée dans le prompt RAG pour résoudre la coréférence.

Defi P13 D1 — niveau 1/2.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import List


@dataclass
class Turn:
    """Un tour de conversation : question utilisateur + réponse assistant."""
    user: str
    assistant: str

    def to_prompt_block(self) -> str:
        return f"Utilisateur : {self.user}\nAssistant : {self.assistant}"


@dataclass
class ShortTermMemory:
    """
    Buffer fenêtré des N derniers tours de conversation active.

    Volontairement minimaliste : pas de dépendance LangChain pour rester
    indépendant des évolutions API, et pour faciliter les tests.
    """
    window_size: int = 5
    _turns: deque = field(default_factory=deque)

    def __post_init__(self):
        if not isinstance(self._turns, deque):
            self._turns = deque(self._turns, maxlen=self.window_size)
        else:
            self._turns = deque(self._turns, maxlen=self.window_size)

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Ajoute un tour. Le plus ancien est éjecté si fenêtre pleine."""
        self._turns.append(Turn(user=user_msg, assistant=assistant_msg))

    def get_history(self) -> List[Turn]:
        """Retourne la liste ordonnée des tours stockés (chronologique)."""
        return list(self._turns)

    def to_prompt_block(self) -> str:
        """
        Sérialise l'historique pour injection dans le prompt RAG.
        Vide si aucun tour.
        """
        if not self._turns:
            return ""
        blocks = [t.to_prompt_block() for t in self._turns]
        return (
            "=== Historique de la conversation ===\n"
            + "\n\n".join(blocks)
            + "\n=== Question actuelle ==="
        )

    def clear(self) -> None:
        """Vide la mémoire courte (bouton 'Nouvelle conversation')."""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def is_empty(self) -> bool:
        return len(self._turns) == 0

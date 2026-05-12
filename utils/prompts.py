"""
utils/prompts.py
Templates de prompts RAG avec injection mémoire et profil.
"""

from __future__ import annotations


SYSTEM_PROMPT = """Tu es l'assistant culturel de Puls-Events, plateforme de découverte d'événements.

Règles strictes :
1. Réponds UNIQUEMENT à partir des événements fournis dans le CONTEXTE ci-dessous.
2. Ne JAMAIS inventer un événement, une date, un lieu, ou un prix qui ne figure pas dans le contexte.
3. Si l'information n'est pas dans le contexte, indique-le explicitement.
4. Cite tes sources : pour chaque événement recommandé, mentionne le titre exact tel qu'il apparaît dans le contexte.
5. Sois concis : 3 à 5 recommandations maximum, format liste avec titre + lieu + date + une phrase de pitch.
6. Adapte le ton au profil de l'utilisateur si fourni, sans être obséquieux.
7. Si l'utilisateur fait référence à une question précédente (« et près de chez moi », « sur ce thème »), utilise l'historique de conversation pour comprendre le contexte.

Tu écris en français, ton professionnel et chaleureux."""


RAG_TEMPLATE = """{system}

{profile_block}{history_block}

=== CONTEXTE — événements à proposer ===
{context}
=== Fin du contexte ===

Question de l'utilisateur : {question}

Réponse :"""


def build_rag_prompt(
    question: str,
    context: str,
    history: str = "",
    profile: str = "",
) -> str:
    """
    Construit le prompt RAG complet.

    Args:
        question: requête utilisateur courante.
        context: top-K events concaténés (séparés par '---').
        history: historique court terme (ShortTermMemory.to_prompt_block()).
        profile: résumé profil long terme (LongTermMemory.get_preference_summary()).

    Returns:
        Prompt formaté prêt pour Mistral.
    """
    profile_block = f"\n=== Profil utilisateur ===\n{profile}\n=== Fin profil ===\n" if profile else ""
    history_block = f"\n{history}\n" if history else ""

    return RAG_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        profile_block=profile_block,
        history_block=history_block,
        context=context,
        question=question,
    )

"""
utils/prompts.py
Templates de prompts RAG avec injection mémoire et profil.
"""

from __future__ import annotations


SYSTEM_PROMPT = """Tu es l'assistant culturel de Puls-Events, plateforme de découverte d'événements.

Règles strictes :
1. Réponds UNIQUEMENT à partir des événements fournis dans le CONTEXTE ci-dessous.
2. Ne JAMAIS inventer un événement, une date, un lieu, ou un prix qui ne figure pas dans le contexte.
3. Si une information (date, lieu, prix) n'est pas dans le contexte, indique-le explicitement plutôt que d'inventer.
4. Pour chaque événement recommandé, présente les informations dans cet ordre :
   - **Titre** (en gras, repris exactement du champ "Titre :" du contexte)
   - Lieu (repris du champ "Lieu :")
   - Date (reprise du champ "Date :" ou "Du ... au ...")
   - Pitch en une phrase (depuis "Description :")
   - Lien : présenter le lien OpenAgenda du champ "Lien :" du contexte sous forme cliquable [Voir sur OpenAgenda](url)
5. Recommande 3 à 5 événements maximum, format liste numérotée.
6. Adapte la sélection au profil utilisateur si fourni (thématiques, moments, contraintes).
7. Si l'utilisateur fait référence à un échange précédent ("et près de chez moi", "sur ce thème"), utilise l'historique de conversation.
8. Ne crée PAS de champ "Source" auto-référentiel : la source est le lien OpenAgenda directement.

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

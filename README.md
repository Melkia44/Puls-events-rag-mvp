---
title: Puls-Events MVP
emoji: 🎭
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.36.0
app_file: app.py
pinned: false
license: mit
short_description: Chatbot RAG de découverte d'événements culturels
---

# 🎭 Puls-Events MVP

Assistant culturel conversationnel basé sur du RAG (Retrieval-Augmented Generation),
qui aide à découvrir des événements à proximité avec mémoire conversationnelle persistée.

**Projet P13 · Parcours Data Engineer · OpenClassrooms 2026**

## Stack

- **UI** : Gradio
- **LLM** : Mistral La Plateforme (API)
- **Vector DB** : FAISS local
- **Persistance** : PostgreSQL Supabase (Session Pooler)
- **Embeddings** : Mistral `mistral-embed`

## Périmètre vague 1

- ✅ D1 — Mémoire conversationnelle (court + long terme + extraction préférences)
- ⏳ D2 — Contexte géographique (vague 2)
- ⏳ D3 — Recherche web smolagents (vague 3)
- ⏳ D4 — Monitoring Langfuse + RAGAS (vague 3)

## Variables d'environnement requises

À configurer dans **Settings → Variables and secrets** :

- `MISTRAL_API_KEY` (secret) — clé API Mistral La Plateforme
- `DATABASE_URL` (secret) — connection string Supabase Session Pooler

Voir `.env.example` pour le template complet.

## Lien projet

Repo POC P11 (base technique) : [github.com/Melkia44/P11](https://github.com/Melkia44/P11)

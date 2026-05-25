---
title: Puls-Events MVP
emoji: 🎭
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Chatbot RAG de découverte d'événements culturels
---

# 🎭 Puls-Events MVP

> Assistant culturel conversationnel basé sur du **RAG (Retrieval-Augmented Generation)**
> qui aide à découvrir des événements culturels dans les 8 plus grandes métropoles françaises.

**Projet P13 · Parcours Data Engineer · OpenClassrooms · 2026**

🚀 **Démo live :** [huggingface.co/spaces/Melkia44/puls-events-mvp](https://huggingface.co/spaces/Melkia44/puls-events-mvp)

---

## ✨ Fonctionnalités

- 🧠 **Mémoire conversationnelle** : court terme (5 derniers tours) + long terme (préférences extraites + persistées)
- 📍 **Filtrage géographique** : Haversine + détection ville utilisateur (8 métropoles : Paris, Lyon, Marseille, Toulouse, Nantes, Bordeaux, Lille, Strasbourg)
- 🌐 **Agent web temps réel** : Brave Search + whitelist de 33 domaines de confiance (presse culturelle, billetteries, agendas)
- 💬 **Sidebar historique** : navigation entre conversations passées (style Claude.ai / ChatGPT)
- 📊 **Observabilité LLM** : traces, latence, coût $ par requête via Langfuse
- ⚖️ **Évaluation qualité** : métriques RAGAS (faithfulness + answer relevancy)

---

## 🏗️ Stack technique

| Couche | Outils |
|--------|--------|
| UI | Gradio 5.50 |
| LLM | Mistral La Plateforme (Large pour synthèse, Small pour routing) |
| Vector DB | FAISS local (7024 vecteurs / 5622 événements, 8 métropoles, 100% géolocalisés) |
| Persistance | PostgreSQL Supabase (Session Pooler, eu-west-3 Paris) |
| Embeddings | Mistral `mistral-embed` |
| Recherche web | Brave Search API + smolagents |
| Observabilité | Langfuse Cloud (traces LLM + coût) |
| Évaluation | RAGAS (faithfulness + answer relevancy) |
| Source données | OpenAgenda API (événements culturels) |

---

## 🎯 Les 4 défis P13

### D1 — Mémoire conversationnelle ✅
- Court terme : buffer 5 tours injecté dans le prompt RAG
- Long terme : extraction préférences via Mistral Small (LLM-as-extractor) + persistance Supabase
- Sidebar UI : navigation entre conversations passées

### D2 — Contexte géographique ✅
- Géocodage Nominatim avec batch + cache
- Filtre Haversine post-retrieval
- Logique `strict_in_radius` → bascule Web si zéro source locale (Cas A)
- Tuning empirique `RETRIEVER_K_GEO=100` (fenêtre grise 33% → 8%)

### D3 — Agent web temps réel ✅
- Routeur 3 cascades : trigger_keyword regex → strict_in_radius → LLM Mistral Small
- Brave Search + whitelist YAML 33 domaines (5 catégories curées)
- Déduplication par domaine post-filtrage (diversité éditoriale forcée)
- Fallback DuckDuckGo si Brave indispo

### D4 — Monitoring & évaluation ✅
- **Langfuse** : traces des 3 sites `.invoke()` LLM (RAG + routeur + agent web)
  - Prompt + completion + tokens + latence + coût $ par requête
  - Pricing Mistral mappé (Large $2/M in, Small $0.20/M in)
- **RAGAS** : évaluation automatique sur 10 cas de test (4 chemins du pipeline)
  - Faithfulness (anti-hallucination)
  - Answer relevancy (pertinence)
  - Script `scripts/evaluate_ragas.py` reproductible

---

## 🚀 Démarrage rapide (local)

### Pré-requis
- Python 3.12
- Comptes : Mistral La Plateforme, Supabase, Brave Search (optionnel), Langfuse (optionnel)

### Installation

```bash
# Cloner
git clone https://github.com/Melkia44/Puls-events-rag-mvp.git
cd Puls-events-rag-mvp

# Venv + dépendances
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configurer .env (voir .env.example pour le template)
cp .env.example .env
# → édite .env avec tes clés Mistral + Supabase + (Langfuse + Brave)

# Lancer l'app
python app.py
# → http://localhost:7860
```

### Évaluation RAGAS

```bash
python scripts/evaluate_ragas.py
# → résultats dans evaluation/ragas_results_YYYYMMDD_HHMMSS.csv
```

📖 **Pour le déploiement complet** : voir [DEPLOY.md](./DEPLOY.md)

---

## 📊 Tests & qualité

- **Tests unitaires** : 74/74 verts (hors network)
  ```bash
  python -m pytest tests/ -q -m "not network"
  ```
- **Évaluation RAGAS** : baseline 0.41 Faithfulness (run du 19/05/2026)
- **Monitoring continu** : dashboard Langfuse Cloud

---

## 🗺️ Roadmap V2

| Item | Pourquoi |
|------|----------|
| **Langfuse self-hosted VPC AWS Paris** | Souveraineté RGPD (actuellement Cloud US) |
| **PostHog** | Funnel d'usage utilisateur + A/B testing |
| **Index DB `messages(role, session_id, ts)`** | Perf SQL > 10k messages (US-DB-001) |
| **Retry exponentiel rate limit Mistral** | RAGAS éval complète (US-RAGAS-001/002) |
| **Élargir dataset RAGAS à 30+ questions + ground_truth** | Couvrir `context_precision` et `context_recall` (US-RAGAS-003) |
| **Whitelist sites officiels festivals** | Rio Loco, Beauregard, Hellfest, etc. |
| **Refresh sidebar au new_conv_btn** | UX nice-to-have |
| **Migration Gradio 6** | `theme` dans `launch()` au lieu de `Blocks()` |

---

## 📂 Structure du repo

```
.
├── app.py                       # Point d'entrée Gradio
├── requirements.txt
├── .env.example
├── README.md                    # Ce fichier
├── DEPLOY.md                    # Guide de déploiement complet
├── config/
│   └── domain_whitelist.yaml    # 33 domaines whitelistés D3
├── utils/
│   ├── memory/
│   │   ├── short_term.py        # Buffer fenêtré D1
│   │   ├── long_term.py         # LTM Supabase + sidebar sessions
│   │   └── profile.py           # Extraction préférences
│   ├── geo.py                   # Géocodage + Haversine D2
│   └── web_agent.py             # Routeur D3 + Brave + smolagents
├── data/
│   └── faiss_index/             # Index FAISS (LFS)
├── scripts/
│   ├── ingest.py                # Ingestion OpenAgenda → FAISS
│   └── evaluate_ragas.py        # Évaluation qualité RAG
├── evaluation/
│   ├── eval_dataset.json        # 10 cas de test RAGAS
│   └── ragas_results_*.csv      # Scores horodatés
└── tests/                       # 74 tests unitaires
```

---

## 🎓 Genèse du projet

Évolution **POC → MVP scalable** :
- **P11** (POC validé) : moteur de recherche sémantique + chatbot RAG simple
  → [github.com/Melkia44/P11](https://github.com/Melkia44/P11)
- **P13** (MVP industrialisé) : ajout des 4 défis techniques (mémoire, géo, web, monitoring)
  + observabilité production-ready

---

## 👤 Auteur

**Mathieu Lowagie** — Parcours Data Engineer OpenClassrooms 2026
🔗 [github.com/Melkia44](https://github.com/Melkia44)

---


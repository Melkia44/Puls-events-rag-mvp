# 🚀 Guide de déploiement — Puls-Events MVP

Documentation opérationnelle complète pour reproduire le déploiement local ou HF Spaces.

---

## 📋 Pré-requis

### Comptes externes requis

| Service | Usage | Obligatoire |
|---------|-------|-------------|
| [Mistral La Plateforme](https://console.mistral.ai/) | LLM (Large + Small) + Embeddings | ✅ |
| [Supabase](https://supabase.com/) | Persistance Postgres (D1 long terme) | ✅ |
| [Brave Search API](https://api.search.brave.com/) | Recherche web temps réel (D3) | 🟡 Optionnel (fallback DuckDuckGo) |
| [Langfuse Cloud](https://cloud.langfuse.com/) | Observabilité LLM (D4) | 🟡 Optionnel (dégradation gracieuse) |
| [Hugging Face](https://huggingface.co/) | Hosting Spaces (déploiement public) | 🟡 Optionnel (sinon local) |

### Environnement local

- Python 3.12
- Git LFS (`git lfs version` doit répondre)
- ~3 Go d'espace disque (deps + FAISS + venv)

---

## 🔐 Variables d'environnement

Copier `.env.example` vers `.env` et compléter :

```bash
# === LLM Mistral (obligatoire) ===
MISTRAL_API_KEY=ta_clé_mistral

# === Persistance Supabase (obligatoire) ===
DATABASE_URL=postgresql+psycopg2://postgres.PROJECT_ID:PWD@aws-1-eu-west-3.pooler.supabase.com:5432/postgres

# === Mémoire courte ===
MEMORY_WINDOW_SIZE=5

# === Vector store ===
FAISS_INDEX_PATH=data/faiss_index
RETRIEVER_K=5
RETRIEVER_K_GEO=100
DEFAULT_RADIUS_KM=15

# === Brave Search (optionnel D3) ===
BRAVE_API_KEY=ta_clé_brave

# === Géocodage Nominatim (recommandé : email contact requis depuis 2025) ===
NOMINATIM_CONTACT_EMAIL=ton@email.fr

# === Langfuse (optionnel D4) ===
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

⚠️ **Critique** : Pour Supabase, utiliser **Session Pooler port 5432**, pas Transaction Pooler 6543.

---

## 🛠️ Setup Supabase

1. Créer un projet Supabase (région **eu-west-3 Paris** pour cohérence RGPD)
2. Récupérer la connection string : Settings → Database → Connection string → **Session Pooler** (port 5432)
3. Les tables sont **créées automatiquement** au 1er lancement de l'app (`Base.metadata.create_all()`) :
   - `users` (id, name, city, city_lat, city_lng)
   - `conversation_sessions` (id, user_id, started_at, ended_at)
   - `messages` (id, session_id, role, content, ts)
   - `preferences` (id, user_id, key, value, weight, source_session_id, ts)

### Migration colonnes géo (D2)

Si la table `users` existe sans colonnes géo (déploiement antérieur), elles sont ajoutées via `ALTER TABLE` idempotent au démarrage.

---

## 🚀 Déploiement local

```bash
# 1. Cloner
git clone https://github.com/Melkia44/Puls-events-rag-mvp.git
cd Puls-events-rag-mvp

# 2. Git LFS pour récupérer l'index FAISS
git lfs install
git lfs pull

# 3. Venv + dépendances
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. .env (cf. section "Variables d'environnement")
cp .env.example .env
nano .env

# 5. Lancer
python app.py
# → http://localhost:7860
```

### Validation locale

- **Connexion** : créer un profil test, vérifier la sidebar gauche
- **D1 mémoire** : 2 tours coréférencés ("J'aime le jazz" → "Recommande-moi quelque chose")
- **D2 géographique** : "Concerts à 15 km de Nantes" → badge vert "Filtré à 15 km autour de Nantes"
- **D2 hors-couverture** : "Concerts à 15 km de Saint-Nazaire" → badge violet "bascule Web"
- **D3 trigger** : "Réserver des billets pour Beauregard" → badge violet "trigger_keyword"
- **D4 Langfuse** : vérifier l'apparition des traces sur `cloud.langfuse.com`

---

## 🌐 Déploiement HF Spaces

### 1. Configurer les secrets

Settings → **Variables and secrets** → **New secret**.

Ajouter en tant que **Secret** (pas Variable) :
- `MISTRAL_API_KEY`
- `DATABASE_URL`
- `BRAVE_API_KEY` (si D3)
- `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` + `LANGFUSE_HOST` (si D4)

### 2. Push

Le remote `origin` pointe vers HF Spaces :

```bash
git remote -v
# origin   https://huggingface.co/spaces/Melkia44/puls-events-mvp

# Push code + tags
git push origin main
git push origin --tags
```

### 3. Authentification

- Username : `Melkia44`
- Password : token HF (commence par `hf_...`) — généré sur https://huggingface.co/settings/tokens

### 4. Surveillance build

- URL Space → onglet **Logs**
- Build 3-5 min (téléchargement deps + LFS)
- Bannière verte "Running" = OK
- Bannière rouge → vérifier les logs

---

## 📊 Évaluation RAGAS

```bash
# 1. Vérifier que le pipeline répond
python -c "from app import LLM, rag_response; print('OK')"

# 2. Lancer l'évaluation
python scripts/evaluate_ragas.py
# Durée : 5-10 min (limite Mistral ~30 req/min, sleep 4s entre questions)
# Coût : ~$0.05 sur 10 questions

# 3. Voir les résultats
ls evaluation/ragas_results_*.csv
```

Le script génère un CSV horodaté avec faithfulness + answer_relevancy par question.

---

## 🔧 Troubleshooting

### `Connection refused` ou timeout Supabase

- Vérifier port **5432** (Session Pooler), pas 6543
- Vérifier `aws-1-eu-west-3.pooler.supabase.com` dans la connection string
- Tester depuis psql : `psql "<DATABASE_URL>"`

### `405 Method Not Allowed` Langfuse

- **Symptôme** : `Failed to export span batch code: 405`
- **Cause** : `LANGFUSE_HOST` mal configuré ou redirection EU→US
- **Fix** : utiliser `LANGFUSE_HOST=https://cloud.langfuse.com` (pas `eu.cloud.langfuse.com`)

### `Rate limit exceeded` Mistral pendant RAGAS

- **Symptôme** : `429 Rate limit` sur questions 5-10
- **Cause** : tier gratuit Mistral ~30 req/min, RAGAS judge fait 2-3 appels par question
- **Fix** : augmenter `RATE_LIMIT_SLEEP` dans `scripts/evaluate_ragas.py` (4s → 15s)

### `FileNotFoundError: index.faiss`

- **Symptôme** : l'index FAISS n'est pas trouvé au boot
- **Cause** : LFS pas pull
- **Fix** :
  ```bash
  git lfs pull
  ls -lh data/faiss_index/   # doit afficher index.faiss ~5 Mo
  ```

### `Nominatim 403 Forbidden`

- **Symptôme** : géocodage des villes échoue
- **Cause** : User-Agent invalide (OSM rejette `@example.com` depuis 2025)
- **Fix** : configurer `NOMINATIM_CONTACT_EMAIL=ton@email.fr` dans `.env`

### `Authentication error: Langfuse client initialized without public_key`

- **Symptôme** : warning au boot, traces non envoyées
- **Cause** : `.env` pas chargé OU vars d'env mal nommées
- **Fix** : vérifier `LANGFUSE_PUBLIC_KEY` (pas `LANGFUSE_API_KEY`) dans `.env`

---

## 📦 Données — Ingestion FAISS (optionnelle)

Si tu veux **reconstruire l'index FAISS** (au lieu d'utiliser celui livré via LFS) :

```bash
python scripts/ingest_top8.py
# Source : OpenAgenda API (gratuite, pas de clé requise)
# Couvre 8 métropoles (Paris, Lyon, Marseille, Toulouse, Nantes, Bordeaux, Lille, Strasbourg)
# Sortie : ~7024 vecteurs (~5622 événements uniques), 100% géolocalisés
# Durée : ~5-10 min (rate-limited côté OpenAgenda)
#
# (scripts/ingest.py seul = Nantes uniquement, défaut GEO_CITY ; rétro-compat P11)
```

---

## 🎓 Architecture globale

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         Pipeline Puls (app.py)          │
│                                         │
│  1. _resolve_geo_target() → ville       │
│  2. FAISS retrieval (k=100)             │
│  3. filter_by_radius (Haversine 15 km)  │
│  4. Routeur 3 cascades (D3) :           │
│     - trigger_keyword regex             │
│     - strict_in_radius == 0             │
│     - LLM Mistral Small                 │
│  5. Si RAG : Mistral Large + sources    │
│     Si Web : Brave + whitelist + agent  │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Response   │
└─────────────┘

Persistance :     Supabase Postgres (sessions, messages, prefs)
Observabilité :   Langfuse Cloud (traces, coût, latence)
Évaluation :      RAGAS (faithfulness, relevancy) - run manuel
```

---

## 📜 Versions principales

| Tag | Contenu |
|-----|---------|
| `v0.2-d2-stable` | D1 mémoire + D2 géo fonctionnels |
| `v0.3-d3-routed` | Routeur 3 cascades + bascule Cas A |
| `v0.4-sidebar-history` | Sidebar conversations gauche |
| `v0.4.1-simplify` | DRY + perf SQL + IDOR garde |
| `v0.5-d4-langfuse` | Observabilité LLM |
| `v0.6-d4-ragas` | Éval qualité RAG |

Voir les notes annotées : `git show <tag>`.

---



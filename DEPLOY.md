# 🚀 Mode d'emploi — Vague 1 (D1 mémoire fonctionnelle)

> Objectif : à la fin, ton Space HF affiche un chatbot Puls-Events avec mémoire conversationnelle (court + long terme persisté Supabase), accessible via URL publique.

---

## Pré-requis vérifiés

- ✅ VM Ubuntu fonctionnelle avec Python 3.12
- ✅ Git LFS installé (`git lfs version` répond)
- ✅ Clone HF Space dans `~/projets/p13/puls-events-mvp/`
- ✅ Token HF Write configuré
- ✅ Index FAISS copié dans `~/projets/p13/puls-events-mvp/data/faiss_index/`
- ✅ Clé Mistral API notée quelque part de sûr
- ✅ Projet Supabase actif avec connection string Session Pooler 5432

---

## Étape 1 — Dézipper le code dans le repo MVP (3 min)

Dans ta VM Ubuntu, depuis ton dossier de téléchargement :

```bash
# Aller dans le repo MVP
cd ~/projets/p13/puls-events-mvp

# Vérifier qu'on est bien dans le clone HF (doit avoir .git/, app.py Hello World)
ls -la

# Dézipper le pack vague 1 (chemin à adapter selon où tu télécharges)
unzip ~/Téléchargements/mvp-v1.zip -d /tmp/mvp-v1
ls /tmp/mvp-v1
```

Tu dois voir l'arborescence du pack :
```
app.py
requirements.txt
README.md
.env.example
.gitignore
utils/
DEPLOY.md
```

Puis copier tous les fichiers vers le repo MVP (en remplaçant les existants comme app.py Hello World et README.md HF) :

```bash
cp -r /tmp/mvp-v1/* ~/projets/p13/puls-events-mvp/
cp /tmp/mvp-v1/.env.example ~/projets/p13/puls-events-mvp/
cp /tmp/mvp-v1/.gitignore ~/projets/p13/puls-events-mvp/

# Vérifier
ls -la ~/projets/p13/puls-events-mvp/
```

Tu dois maintenant voir : `app.py`, `requirements.txt`, `README.md`, `.env.example`, `.gitignore`, `utils/`, `data/faiss_index/` (avec les fichiers copiés du P11), `.git/`, `.gitattributes`.

---

## Étape 2 — Créer le `.env` local pour les tests (2 min)

```bash
cd ~/projets/p13/puls-events-mvp
cp .env.example .env
nano .env
```

Édite et remplace les placeholders par tes vraies valeurs :

```env
MISTRAL_API_KEY=ta-vraie-cle-mistral-ici
DATABASE_URL=postgresql+psycopg2://postgres.<PROJECT_ID>:<MDP>@aws-1-eu-west-3.pooler.supabase.com:5432/postgres
MEMORY_WINDOW_SIZE=5
FAISS_INDEX_PATH=data/faiss_index
```

Sauve avec `Ctrl+O`, `Enter`, `Ctrl+X`.

---

## Étape 3 — Tester le MVP en local (10 min)

Avant de pousser sur HF (qui prend 3-5 min à chaque build), **on valide en local** que tout marche.

### Créer un venv dédié

```bash
cd ~/projets/p13/puls-events-mvp
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

L'installation prend ~2-3 min (téléchargement Gradio, LangChain, FAISS, SQLAlchemy, psycopg2-binary).

### Lancer l'app en local

```bash
python app.py
```

Tu dois voir dans les logs :
```
... | INFO | Chargement du vector store FAISS…
... | INFO | Vector store prêt
... | INFO | Connexion à Supabase Postgres…
... | INFO | Schéma DB vérifié/créé sur Supabase
... | INFO | Mémoire long terme prête
... | INFO | Initialisation LLM Mistral…
Running on local URL:  http://0.0.0.0:7860
```

Ouvre **http://localhost:7860** dans ton navigateur (depuis la VM ou depuis l'hôte si bridge réseau).

### Tester la chaîne complète

1. Dans le champ "ou créer un nouveau profil" : tape `Léa`, clique **Activer ce profil**
2. Dans le chat : tape `Quels concerts à Nantes ce week-end ?`
3. L'app répond avec des recommandations issues de l'index FAISS
4. Tape `Et plutôt en après-midi ?` — l'app doit comprendre la coréférence (mémoire courte)
5. Clique **🔄 Nouvelle conversation** — les préférences sont extraites et persistées
6. Le profil de Léa apparaît dans la sidebar avec ses préférences

### Vérifier dans Supabase Table Editor

Va sur ton dashboard Supabase → **Table Editor**. Tu dois voir maintenant 4 tables :
- `users` avec 1 ligne (Léa)
- `conversation_sessions` avec 1-2 lignes
- `messages` avec plusieurs lignes
- `preferences` avec quelques lignes selon ce que Mistral a extrait

**Si tout marche en local, on pousse sur HF.**

### Arrêter le serveur local

`Ctrl+C` dans le terminal.

---

## Étape 4 — Configurer les secrets HF Spaces (3 min)

Va sur :
**https://huggingface.co/spaces/Melkia44/puls-events-mvp/settings**

Section **"Variables and secrets"** → **"New secret"**.

Ajoute 2 secrets :

| Nom | Type | Valeur |
|---|---|---|
| `MISTRAL_API_KEY` | **Secret** | Ta clé Mistral |
| `DATABASE_URL` | **Secret** | Connection string Supabase complète |

⚠️ Choisis bien **Secret**, pas Variable. Les variables sont visibles dans les logs build, pas les secrets.

---

## Étape 5 — Pousser sur HF Spaces (5 min)

### Préparer Git LFS pour l'index FAISS

```bash
cd ~/projets/p13/puls-events-mvp

# Activer LFS pour les binaires (déjà fait avant mais on revérifie)
git lfs install
git lfs track "*.faiss" "*.pkl" "*.bin"

# Vérifier .gitattributes
cat .gitattributes
```

Tu dois voir au moins :
```
*.faiss filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
```

### Vérifier ce qui va être committé

```bash
git status
```

Tu dois voir comme "fichiers modifiés" : `app.py`, `README.md`, `requirements.txt`, et comme "fichiers nouveaux" : `.env.example`, `.gitignore`, `utils/`, `data/`, etc.

**Le `.env` ne doit PAS apparaître** (il est dans le `.gitignore`). Vérifie deux fois.

### Commit et push

```bash
git add .
git status   # revérifier que .env n'est pas listé
git commit -m "Vague 1 : RAG + mémoire D1 fonctionnelle"
git push
```

Au premier push avec LFS :
- Username : `Melkia44`
- Password : ton token HF (commencer par `hf_...`)

Le push prend 1-2 min (l'index FAISS de 5.3 Mo passe par LFS).

---

## Étape 6 — Surveiller le build HF (3-5 min)

Va sur :
**https://huggingface.co/spaces/Melkia44/puls-events-mvp**

Tu vois en haut une bannière jaune **"Building"**. Clique sur l'onglet **Logs** pour voir le build en temps réel.

Tu verras :
1. Téléchargement des dépendances (1-2 min)
2. `Application startup complete` quand l'app est lancée
3. Bannière verte **"Running"**

Si la bannière passe verte → ouvre l'URL, teste avec un utilisateur, vérifie que ça marche.

---

## Étape 7 — En cas de bug

### "ImportError: No module named X"

Vérifier que la dépendance est bien dans `requirements.txt`. Si oui et que ça plante quand même, fais :
```bash
echo "" >> requirements.txt  # force un retrigger
git commit -am "Force rebuild"
git push
```

### "Connection refused" sur Supabase

Vérifier que `DATABASE_URL` dans les secrets HF utilise bien le **Session Pooler port 5432**, pas Transaction Pooler 6543.

### "FileNotFoundError: index.faiss"

Vérifier que l'index a bien été poussé via LFS :
```bash
git lfs ls-files
```
Tu dois voir `data/faiss_index/index.faiss` et `data/faiss_index/index.pkl`.

Si non, tu as oublié l'étape `git lfs track` avant le `git add`. Refais :
```bash
git lfs track "*.faiss" "*.pkl"
git rm --cached data/faiss_index/index.faiss data/faiss_index/index.pkl
git add data/faiss_index/index.faiss data/faiss_index/index.pkl
git commit -m "Move FAISS index to LFS"
git push
```

### "MISTRAL_API_KEY manquant" dans les logs HF

Vérifier dans Settings → Variables and secrets que le secret est bien nommé `MISTRAL_API_KEY` (sensible à la casse).

---

## Étape 8 — Validation finale

Une fois le Space en **Running** :

1. Ouvre l'URL publique du Space
2. Crée 2 utilisateurs : `Léa` et `Thomas`
3. Sur Léa : 3-4 questions sur des concerts ou expos
4. Clique "Nouvelle conversation" pour déclencher l'extraction préférences
5. Switche sur Thomas, pose des questions différentes (jeunesse, famille)
6. Vérifie dans Supabase Table Editor que les 4 tables se sont remplies
7. Vérifie que le profil de Léa contient des préférences distinctes de Thomas

**Si ces 7 étapes sont OK → vague 1 validée, on passe à la vague 2 (D2 géo + re-ranker).**

---

## Récap final vague 1

Tu as :
- ✅ Un Space HF en ligne avec chatbot Puls-Events
- ✅ Mémoire conversationnelle 3 étages fonctionnelle
- ✅ Persistance Postgres Supabase opérationnelle
- ✅ Index FAISS du P11 ré-utilisé (pas de réindexation)
- ✅ URL démontrable au mentor

Tu peux montrer **dès aujourd'hui** au mentor un MVP fonctionnel sur D1.

Les défis D2, D3, D4 arrivent dans les vagues suivantes.

"""
scripts/evaluate_ragas.py
─────────────────────────
Évaluation automatique du pipeline RAG Puls via RAGAS.

Boucle sur evaluation/eval_dataset.json, appelle le vrai pipeline RAG
pour chaque question, puis calcule 2 métriques RAGAS :
  - faithfulness     : anti-hallucination (réponse supportée par sources)
  - answer_relevancy : pertinence (répond à la question)

Mistral Large est utilisé comme LLM-as-judge.
Rate limit Mistral : sleep 4s entre questions (~6 req/min sustainable).
Résultats exportés en CSV horodaté.

Usage : python scripts/evaluate_ragas.py
Coût estimé : ~$0.05 sur 10 questions.
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("ragas_eval")

for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "ragas"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# === CONFIG ===
DATASET_PATH = ROOT / "evaluation" / "eval_dataset.json"
OUTPUT_DIR = ROOT / "evaluation"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = OUTPUT_DIR / f"ragas_results_{TIMESTAMP}.csv"

# Rate limit : Mistral La Plateforme ~1 req/sec en pratique
RATE_LIMIT_SLEEP = 4.0


def load_eval_dataset():
    logger.info(f"📋 Chargement du dataset : {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"   → {len(data)} questions chargées")
    return data


def load_pipeline():
    logger.info("🔧 Chargement du pipeline RAG (import app.py)...")
    from app import LLM, rag_response
    from utils.memory.short_term import ShortTermMemory
    logger.info("   → Pipeline chargé (LLM Mistral Large + rag_response)")
    return LLM, rag_response, ShortTermMemory


def run_pipeline_on_questions(questions, rag_response, ShortTermMemory):
    results = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        category = item.get("category", "?")
        logger.info(f"[{i}/{len(questions)}] {category} | {q[:60]}...")

        # [Rate limit] Pause entre questions (sauf la 1re)
        if i > 1:
            time.sleep(RATE_LIMIT_SLEEP)

        short_term = ShortTermMemory(window_size=5)
        user_id = None

        try:
            t0 = time.time()
            answer, docs, geo_info = rag_response(q, short_term, user_id)
            latency = time.time() - t0
            contexts = [d.page_content for d in (docs or [])][:5]
            results.append({
                "question": q,
                "category": category,
                "answer": answer,
                "contexts": contexts,
                "latency_s": round(latency, 2),
                "n_contexts": len(contexts),
            })
            logger.info(f"   ✅ {len(contexts)} contextes, {latency:.1f}s, "
                        f"{len(answer)} chars")
        except Exception as e:
            logger.error(f"   ❌ Erreur sur question {i} : {str(e)[:200]}")
            results.append({
                "question": q,
                "category": category,
                "answer": f"[ERREUR PIPELINE] {str(e)[:200]}",
                "contexts": [],
                "latency_s": 0.0,
                "n_contexts": 0,
            })
    return results


def evaluate_with_ragas(results, LLM):
    logger.info("⚖️  Évaluation RAGAS (faithfulness + answer_relevancy)...")
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from langchain_mistralai import MistralAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    eval_data = [
        r for r in results
        if r["contexts"] and not r["answer"].startswith("[ERREUR")
    ]
    skipped = len(results) - len(eval_data)
    if skipped:
        logger.warning(
            f"   ⚠️  {skipped} questions exclues (sans contextes ou erreur pipeline)"
        )

    if not eval_data:
        logger.error("   ❌ Aucune donnée valide à évaluer")
        return None, []

    ds = Dataset.from_dict({
        "question": [r["question"] for r in eval_data],
        "answer":   [r["answer"]   for r in eval_data],
        "contexts": [r["contexts"] for r in eval_data],
    })

    judge_llm = LangchainLLMWrapper(LLM)
    embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(model="mistral-embed")
    )

    result = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )
    logger.info(f"   ✅ Éval RAGAS terminée")
    return result, eval_data


def export_results(ragas_result, eval_data, all_results):
    import pandas as pd

    df_ragas = ragas_result.to_pandas()
    logger.info(f"   Colonnes RAGAS : {list(df_ragas.columns)}")

    df_meta = pd.DataFrame([{
        "question": r["question"],
        "category": r["category"],
        "latency_s": r["latency_s"],
        "n_contexts": r["n_contexts"],
    } for r in eval_data])

    # Détection dynamique de la colonne question (varie selon version RAGAS)
    question_col = None
    for candidate in ["question", "user_input", "Question"]:
        if candidate in df_ragas.columns:
            question_col = candidate
            break

    if question_col and question_col != "question":
        df_ragas = df_ragas.rename(columns={question_col: "question"})

    if "question" in df_ragas.columns:
        df_final = pd.merge(df_meta, df_ragas, on="question", how="left")
    else:
        logger.warning("   Pas de colonne 'question' commune — concat par index")
        df_final = pd.concat(
            [df_meta.reset_index(drop=True), df_ragas.reset_index(drop=True)],
            axis=1,
        )

    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info(f"💾 Résultats exportés : {OUTPUT_CSV}")

    # Résumé console
    print("\n" + "═" * 70)
    print("📊 RÉSULTATS RAGAS — Évaluation Puls MVP P13")
    print("═" * 70)
    print(f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Dataset  : {len(all_results)} questions")
    print(f"Évaluées : {len(eval_data)} (avec contextes valides)")
    print(f"Skippées : {len(all_results) - len(eval_data)} (sans contextes ou erreur)")
    print("─" * 70)

    score_cols = {
        "faithfulness": ["faithfulness", "Faithfulness"],
        "answer_relevancy": ["answer_relevancy", "AnswerRelevancy"],
    }
    for label, candidates in score_cols.items():
        for col in candidates:
            if col in df_final.columns:
                values = df_final[col].dropna()
                if len(values) > 0:
                    mean = values.mean()
                    if label == "faithfulness":
                        print(f"Faithfulness     (anti-hallucination) : {mean:.3f}  (sur {len(values)}/{len(df_final)} évalués)")
                    else:
                        print(f"Answer relevancy (pertinence)         : {mean:.3f}  (sur {len(values)}/{len(df_final)} évalués)")
                break

    print("─" * 70)
    print(f"📁 Détails par question : {OUTPUT_CSV.name}")
    print("═" * 70 + "\n")


def main():
    logger.info("🚀 Évaluation RAGAS — démarrage")
    logger.info(f"   Sortie : {OUTPUT_CSV}")
    logger.info(f"   Sleep entre questions : {RATE_LIMIT_SLEEP}s (anti rate-limit)")

    questions = load_eval_dataset()
    LLM, rag_response, ShortTermMemory = load_pipeline()
    all_results = run_pipeline_on_questions(questions, rag_response, ShortTermMemory)

    ragas_result, eval_data = evaluate_with_ragas(all_results, LLM)
    if ragas_result is None:
        logger.error("Échec éval RAGAS")
        sys.exit(1)

    export_results(ragas_result, eval_data, all_results)
    logger.info("✅ Terminé")


if __name__ == "__main__":
    main()

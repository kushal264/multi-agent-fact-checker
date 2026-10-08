"""
benchmark.py
============
Runs the debate pipeline over a labeled claims dataset and reports real
accuracy numbers -- exactly the kind of quantitative result you put on a
resume line, the same way you reported F1/ROC-AUC for Fraud Detection and
accuracy for Plant Disease.

Usage:
    python benchmark.py                # uses DummyLLM, no API key needed
    python benchmark.py --real         # uses GeminiLLM, needs GEMINI_API_KEY

LEARNING NOTE: "UNCERTAIN" verdicts are counted as incorrect in accuracy,
but reported separately -- in a real fact-checking system, an UNCERTAIN
verdict that correctly flags an ambiguous/under-evidenced claim is often
more valuable than a confident wrong answer. We report both numbers so
you can talk about that nuance in an interview.
"""

import argparse
import json

from debate_agents.retriever import TfidfRetriever
from debate_agents.debate import run_debate
from debate_agents.llm import DummyLLM, GeminiLLM


def load_json(path):
    with open(path) as f:
        return json.load(f)


def run_benchmark(use_real_llm: bool = False, rebuttal_round: bool = True):
    corpus = load_json("data/evidence_corpus.json")
    claims = load_json("data/claims.json")

    retriever = TfidfRetriever(corpus)
    llm = GeminiLLM() if use_real_llm else DummyLLM()

    correct = 0
    uncertain = 0
    results = []

    for item in claims:
        result = run_debate(item["claim"], retriever, llm, rebuttal_round=rebuttal_round)
        predicted = result.verdict.verdict
        is_correct = predicted == item["label"]
        correct += int(is_correct)
        uncertain += int(predicted == "UNCERTAIN")

        results.append({
            "claim": item["claim"],
            "gold_label": item["label"],
            "predicted": predicted,
            "confidence": result.verdict.confidence,
            "correct": is_correct,
        })

        print(f"[{'OK ' if is_correct else 'ERR'}] {item['claim'][:60]:<60} "
              f"gold={item['label']:<10} pred={predicted:<10} conf={result.verdict.confidence}")

    n = len(claims)
    accuracy = correct / n
    print("\n--- Benchmark Summary ---")
    print(f"Claims evaluated : {n}")
    print(f"Correct          : {correct}")
    print(f"Accuracy         : {accuracy:.2%}")
    print(f"Uncertain verdicts: {uncertain}")

    return {"accuracy": accuracy, "n": n, "correct": correct, "uncertain": uncertain, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real Gemini API instead of DummyLLM")
    parser.add_argument("--no-rebuttal", action="store_true", help="Disable the rebuttal round")
    args = parser.parse_args()

    run_benchmark(use_real_llm=args.real, rebuttal_round=not args.no_rebuttal)

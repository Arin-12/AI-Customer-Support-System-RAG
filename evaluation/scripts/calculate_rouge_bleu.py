
import csv
import json
import sys
from pathlib import Path

import requests
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILE = PROJECT_ROOT / "evaluation" / "reference_answers.json"
OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "results" / "rouge_bleu_results.csv"

API_URL = "http://127.0.0.1:8000/chat"
USER_ID = 1


def main():
    with open(REFERENCE_FILE, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    smoothing = SmoothingFunction().method1

    results = []

    for index, case in enumerate(test_cases, start=1):
        query = case["query"]
        reference = case["reference"]

        print(f"\n[{index}/{len(test_cases)}] {query}")

        try:
            response = requests.post(
                API_URL,
                json={
                    "user_id": USER_ID,
                    "message": query,
                },
                timeout=120,
            )
            response.raise_for_status()
            actual = response.json()["answer"]

        except (requests.RequestException, KeyError, ValueError) as exc:
            print(f"Request failed: {exc}")
            print("Stopping so incomplete results aren't mistaken for a full evaluation.")
            break

        rouge = scorer.score(reference, actual)

        reference_tokens = reference.lower().split()
        actual_tokens = actual.lower().split()

        bleu = sentence_bleu(
            [reference_tokens],
            actual_tokens,
            smoothing_function=smoothing,
        )

        row = {
            "query": query,
            "reference_answer": reference,
            "actual_answer": actual,
            "rouge1_f1": rouge["rouge1"].fmeasure,
            "rouge2_f1": rouge["rouge2"].fmeasure,
            "rougeL_f1": rouge["rougeL"].fmeasure,
            "bleu": bleu,
        }
        results.append(row)

        print(f"ROUGE-1 F1: {row['rouge1_f1']:.4f}")
        print(f"ROUGE-2 F1: {row['rouge2_f1']:.4f}")
        print(f"ROUGE-L F1: {row['rougeL_f1']:.4f}")
        print(f"BLEU:      {row['bleu']:.4f}")

    if not results:
        print("\nNo results were generated. Check that the API is running.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n--- Mean metrics ---")
    print(f"Questions evaluated: {len(results)}/{len(test_cases)}")
    print(f"Mean ROUGE-1 F1: {sum(r['rouge1_f1'] for r in results) / len(results):.4f}")
    print(f"Mean ROUGE-2 F1: {sum(r['rouge2_f1'] for r in results) / len(results):.4f}")
    print(f"Mean ROUGE-L F1: {sum(r['rougeL_f1'] for r in results) / len(results):.4f}")
    print(f"Mean BLEU:        {sum(r['bleu'] for r in results) / len(results):.4f}")
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

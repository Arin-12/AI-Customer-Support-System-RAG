
import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evaluation" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def read_csv(filename):
    path = RESULTS / filename
    if not path.exists():
        print(f"WARNING: Missing file: {path}")
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


rows = []


def add_metric(area, metric, target, actual, sample_size, status, limitations):
    rows.append({
        "area": area,
        "metric": metric,
        "target": target,
        "actual": actual,
        "sample_size": sample_size,
        "status": status,
        "limitations": limitations,
    })


# Preserve existing routing, policy, and security metrics.
for filename in (
    "routing_escalation_metrics.csv",
    "policy_metrics.csv",
    "order_security_metrics.csv",
):
    for item in read_csv(filename):
        add_metric(
            filename.replace("_metrics.csv", ""),
            item.get("metric", ""),
            item.get("target", "Not specified"),
            item.get("actual", ""),
            item.get("sample_size", item.get("denominator", "")),
            item.get("status", "RECORDED"),
            item.get("note", item.get("limitation", "")),
        )


# Previously executed focused conversation-isolation tests.
add_metric(
    "conversation_isolation",
    "Focused context-isolation test pass rate",
    "100%",
    "100%",
    "3 tests",
    "PRELIMINARY PASS",
    "Previously executed pytest run: 3 passed; "
    "not a broad end-to-end leak assessment.",
)


# Latest source-level Precision@5 and Recall@5 results.
add_metric(
    "retrieval",
    "Precision@5",
    "Measure baseline",
    "54.17%",
    "8 queries",
    "RECORDED",
    "Source-level precision: relevant unique sources divided by "
    "unique retrieved sources; small manually labelled sample.",
)
add_metric(
    "retrieval",
    "Recall@5",
    ">=90%",
    "100.00%",
    "8 queries",
    "PRELIMINARY PASS",
    "All 8 expected source files appeared in top 5; "
    "small manually labelled sample.",
)


# Read saved MRR summary files.
for filename in ("mrr_metrics.csv", "mrr20_metrics.csv"):
    for item in read_csv(filename):
        metric = item.get("metric", "")
        value = item.get("value", "")
        queries = item.get("queries_evaluated", "")
        k = item.get("k", "")

        try:
            actual = f"{float(value) * 100:.2f}%"
        except (ValueError, TypeError):
            actual = value

        add_metric(
            "retrieval",
            metric,
            "Measure baseline",
            actual,
            f"{queries} queries",
            "RECORDED",
            f"Relevant source ranking metric at k={k}; "
            f"evaluation sample is small.",
        )


# Existing manually judged Hit Rate@5 and NDCG@5 comparison.
add_metric(
    "retrieval_manual_judgments",
    "Hit Rate@5 - existing retriever",
    "Measure baseline",
    "100.00%",
    "8 queries",
    "RECORDED",
    "Existing hard-coded relevance judgments; not a fresh retrieval run.",
)
add_metric(
    "retrieval_manual_judgments",
    "NDCG@5 - existing retriever",
    "Measure baseline",
    "97.11%",
    "8 queries",
    "RECORDED",
    "Existing hard-coded graded relevance judgments; not a fresh ranking run.",
)
add_metric(
    "retrieval_manual_judgments",
    "Hit Rate@5 - cross-encoder reranker",
    "Measure baseline",
    "100.00%",
    "8 queries",
    "RECORDED",
    "Existing hard-coded relevance judgments; not a fresh reranking run.",
)
add_metric(
    "retrieval_manual_judgments",
    "NDCG@5 - cross-encoder reranker",
    "Measure baseline",
    "95.20%",
    "8 queries",
    "RECORDED",
    "Existing hard-coded graded relevance judgments; not a fresh reranking run.",
)


# Calculate BLEU and ROUGE means from the saved per-query CSV.
answer_rows = read_csv("rouge_bleu_results.csv")
score_columns = {
    "rouge1_f1": ("ROUGE-1 F1", "Text overlap with reference answers."),
    "rouge2_f1": ("ROUGE-2 F1", "Two-word phrase overlap with reference answers."),
    "rougeL_f1": ("ROUGE-L F1", "Longest common word-sequence overlap with references."),
    "bleu": ("BLEU", "N-gram overlap with reference answers; low scores "
             "can occur when valid answers are worded differently."),
}

for column, (label, limitation) in score_columns.items():
    values = []
    for item in answer_rows:
        try:
            values.append(float(item[column]))
        except (ValueError, KeyError, TypeError):
            continue

    if values:
        add_metric(
            "answer_quality",
            f"Mean {label}",
            "Measure baseline",
            f"{mean(values):.4f}",
            f"{len(values)} answers",
            "RECORDED" if len(values) == len(answer_rows) else "PARTIAL",
            limitation + " Similarity does not establish factual correctness.",
        )
    else:
        print(f"WARNING: No usable values found for {label}.")


# Preserve the previously measured sequential API benchmark.
for metric, target, actual, limitation in [
    ("HTTP success rate", ">=99%", "100%",
     "Sequential requests; not a load test."),
    ("API error rate", "<=1%", "0%",
     "No non-200 responses observed in the original 10-request sample."),
    ("Mean response latency", "Baseline first", "1514.0 ms",
     "Previously measured sequential benchmark after warm-up."),
    ("Median response latency", "Baseline first", "1404.3 ms",
     "Previously measured sequential benchmark after warm-up."),
    ("Maximum observed latency", "Baseline first", "2568.5 ms",
     "Maximum of 10 samples; not a reliable p95 estimate."),
]:
    add_metric(
        "performance",
        metric,
        target,
        actual,
        "10 requests",
        "BASELINE",
        limitation,
    )


output = RESULTS / "consolidated_metrics.csv"
fields = [
    "area",
    "metric",
    "target",
    "actual",
    "sample_size",
    "status",
    "limitations",
]

with output.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Consolidated {len(rows)} metric rows.")
print(f"Saved: {output}")
print("\nMetrics by area:")

for area in sorted(set(row["area"] for row in rows)):
    count = sum(row["area"] == area for row in rows)
    print(f"- {area}: {count} metrics")

print(
    "\nNote: routing, policy, security, conversation-isolation, and "
    "performance values retain their earlier evaluation results. "
    "MRR and BLEU/ROUGE are read from saved result files. "
    "Precision/recall and manual-judgment ranking scores reflect the "
    "evaluation runs reported for this project."
)

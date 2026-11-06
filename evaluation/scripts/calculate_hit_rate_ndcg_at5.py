
import math

# Relevance judgments in rank order for each system.
# 1.0 = directly relevant, 0.5 = partially relevant, 0.0 = irrelevant.
# These are preliminary manual judgments for the eight test queries.

EXISTING_SCORES = [
    [1.0, 0.5, 1.0, 0.5, 0.5],
    [1.0, 0.0, 0.5, 1.0, 1.0],
    [1.0, 1.0, 0.5, 1.0, 0.5],
    [1.0, 1.0, 0.5, 0.5, 0.0],
    [1.0, 1.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 0.5, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 0.0],
    [1.0, 0.5, 0.0, 0.5, 0.0],
]

RERANKER_SCORES = [
    [0.5, 0.5, 0.5, 0.5, 1.0],
    [1.0, 0.5, 0.0, 1.0, 0.5],
    [1.0, 0.5, 1.0, 0.5, 0.5],
    [1.0, 1.0, 0.5, 0.5, 0.0],
    [1.0, 1.0, 0.0, 0.5, 1.0],
    [1.0, 1.0, 0.5, 0.5, 0.5],
    [1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 0.5, 0.0, 0.5, 0.0],
]


def hit_rate_at_k(scores, k=5):
    return float(any(score > 0 for score in scores[:k]))


def ndcg_at_k(scores, k=5):
    def dcg(values):
        return sum(
            (2 ** score - 1) / math.log2(rank + 1)
            for rank, score in enumerate(values[:k], start=1)
        )

    actual_dcg = dcg(scores)
    ideal_dcg = dcg(sorted(scores, reverse=True))

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate(label, judgments):
    hit_scores = [hit_rate_at_k(scores) for scores in judgments]
    ndcg_scores = [ndcg_at_k(scores) for scores in judgments]

    print(f"\n--- {label} ---")
    print(f"Queries evaluated: {len(judgments)}")
    print(f"Hit Rate@5: {sum(hit_scores) / len(hit_scores):.2%}")
    print(f"Mean nDCG@5: {sum(ndcg_scores) / len(ndcg_scores):.2%}")


if __name__ == "__main__":
    evaluate("EXISTING RETRIEVER", EXISTING_SCORES)
    evaluate("CROSS-ENCODER RERANKER", RERANKER_SCORES)

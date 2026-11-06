
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.agent.router import Route, route_query
from ai.ticket.escalation import should_escalate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# These are the 11 scenarios from tests/test_routing_evaluation.py.
ROUTING_CASES = [
    ("return policy", "What is your return policy?", Route.RAG),
    ("order lookup", "Where is order ORD1002?", Route.SQL),
    (
        "combined order and policy",
        "My order ORD1005 contains a damaged product. Can I get a refund?",
        Route.SQL_AND_RAG,
    ),
    ("human support", "I want to talk to a human agent", Route.TICKET),
    ("vague help", "Can you help me?", Route.UNKNOWN),
    ("product specifications", "What are the laptop specifications?", Route.RAG),
    ("unknown order", "Where is order ORD9999?", Route.SQL),
    ("cancellation policy", "What is your cancellation policy?", Route.RAG),
    ("order cancellation", "Can I cancel order ORD1005?", Route.SQL_AND_RAG),
    ("order tracking", "Can you track order ORD1005?", Route.SQL),
    (
        "human with order ID",
        "I want a human agent to help me with order ORD1005",
        Route.TICKET,
    ),
]

# These are the 4 labelled cases from tests/test_escalation_evaluation.py.
ESCALATION_CASES = [
    {
        "name": "human requested",
        "query": "I want to speak to a human agent.",
        "confidence": 0.85,
        "verification_failed": False,
        "human_requested": True,
        "expected": True,
    },
    {
        "name": "verification failed",
        "query": "Where is my order ORD1005?",
        "confidence": 0.20,
        "verification_failed": True,
        "human_requested": False,
        "expected": True,
    },
    {
        "name": "low confidence",
        "query": "What happened to my order?",
        "confidence": 0.30,
        "verification_failed": False,
        "human_requested": False,
        "expected": True,
    },
    {
        "name": "verified high confidence",
        "query": "Where is my order ORD1005?",
        "confidence": 0.95,
        "verification_failed": False,
        "human_requested": False,
        "expected": False,
    },
]


def percent(value):
    return round(value * 100, 2)


def main():
    rows = []

    # Routing accuracy
    routing_correct = 0

    for name, query, expected in ROUTING_CASES:
        actual = route_query(query)
        correct = actual == expected
        routing_correct += int(correct)

        print(
            f"ROUTING | {name} | expected={expected} "
            f"| actual={actual} | correct={correct}"
        )

    routing_accuracy = routing_correct / len(ROUTING_CASES)

    rows.append({
        "metric": "Routing accuracy",
        "target": ">=95%",
        "actual": f"{percent(routing_accuracy):.2f}%",
        "correct": routing_correct,
        "sample_size": len(ROUTING_CASES),
        "status": "MEETS TARGET" if routing_accuracy >= 0.95 else "BELOW TARGET",
        "note": "11 labelled routing scenarios; small sample",
    })

    # Escalation classification metrics
    tp = tn = fp = fn = 0

    for case in ESCALATION_CASES:
        actual = should_escalate(
            user_query=case["query"],
            confidence=case["confidence"],
            verification_failed=case["verification_failed"],
            human_requested=case["human_requested"],
        )
        expected = case["expected"]

        if expected and actual:
            tp += 1
        elif not expected and not actual:
            tn += 1
        elif not expected and actual:
            fp += 1
        else:
            fn += 1

        print(
            f"ESCALATION | {case['name']} | expected={expected} "
            f"| actual={actual} | correct={actual == expected}"
        )

    total = len(ESCALATION_CASES)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    rows.extend([
        {
            "metric": "Escalation accuracy",
            "target": ">=95%",
            "actual": f"{percent(accuracy):.2f}%",
            "correct": tp + tn,
            "sample_size": total,
            "status": "MEETS TARGET" if accuracy >= 0.95 else "BELOW TARGET",
            "note": "4 labelled escalation scenarios; small sample",
        },
        {
            "metric": "Escalation precision",
            "target": "Measure baseline",
            "actual": f"{percent(precision):.2f}%",
            "correct": tp,
            "sample_size": tp + fp,
            "status": "RECORDED",
            "note": f"TP={tp}; FP={fp}",
        },
        {
            "metric": "Escalation recall",
            "target": "Measure baseline",
            "actual": f"{percent(recall):.2f}%",
            "correct": tp,
            "sample_size": tp + fn,
            "status": "RECORDED",
            "note": f"TP={tp}; FN={fn}",
        },
        {
            "metric": "Escalation F1",
            "target": "Measure baseline",
            "actual": f"{percent(f1):.2f}%",
            "correct": "",
            "sample_size": total,
            "status": "RECORDED",
            "note": "Harmonic mean of precision and recall",
        },
    ])

    output_file = RESULTS_DIR / "routing_escalation_metrics.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric", "target", "actual", "correct",
                "sample_size", "status", "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nMETRICS SUMMARY")
    for row in rows:
        print(
            f"{row['metric']}: {row['actual']} "
            f"(n={row['sample_size']}) - {row['status']}"
        )

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

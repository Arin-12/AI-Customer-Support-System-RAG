
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.verification.response_verifier import verify_response


CASES = [
    {
        "name": "supported return window",
        "context": "Eligible products may generally be returned within 7 days of delivery, subject to product condition and category rules.",
        "answer": "Eligible products may generally be returned within 7 days.",
        "expected_verified": True,
    },
    {
        "name": "unsupported return window",
        "context": "Eligible products may generally be returned within 7 days of delivery, subject to product condition and category rules.",
        "answer": "You can return any product within 14 days.",
        "expected_verified": False,
    },
    {
        "name": "unsupported unopened restriction",
        "context": "Eligible products may generally be returned within 7 days of delivery, subject to product condition and category rules.",
        "answer": "Every product has to be returned unopened.",
        "expected_verified": False,
    },
    {
        "name": "unsupported warranty duration",
        "context": "Warranty terms depend on manufacturer, model, seller, region, and purchase conditions.",
        "answer": "The laptop has a 2-year warranty.",
        "expected_verified": False,
    },
    {
        "name": "unsupported warranty guarantee",
        "context": "Warranty terms depend on manufacturer, model, seller, region, and purchase conditions.",
        "answer": "Your laptop is definitely covered under warranty.",
        "expected_verified": False,
    },
    {
        "name": "unsupported guaranteed refund",
        "context": "Refund eligibility must be determined after verifying the order and applicable policy.",
        "answer": "Your refund is guaranteed.",
        "expected_verified": False,
    },
    {
        "name": "unconfirmed callback",
        "context": "Customers may contact support for assistance with their case.",
        "answer": "A specialist will call you shortly.",
        "expected_verified": False,
    },
    {
        "name": "qualified refund",
        "context": "Eligible cases may include approved returns and damaged products. Refund eligibility must be determined after verifying the order and policy.",
        "answer": "A damaged product may qualify for a refund after the order and applicable policy have been verified.",
        "expected_verified": True,
    },
    {
        "name": "empty answer",
        "context": "Some valid context",
        "answer": "",
        "expected_verified": False,
    },
    {
        "name": "answer without context",
        "context": "",
        "answer": "This product is eligible for a refund.",
        "expected_verified": False,
    },
]


def main():
    results = []

    for case in CASES:
        verification = verify_response(
            answer=case["answer"],
            context=case["context"],
        )

        actual = bool(verification.verified)
        expected = case["expected_verified"]
        correct = actual == expected

        results.append({
            "case": case["name"],
            "expected_verified": expected,
            "actual_verified": actual,
            "correct_decision": correct,
            "confidence": verification.confidence,
            "reason": verification.reason,
        })

        print(
            f"{case['name']}: expected={expected}, "
            f"actual={actual}, correct={correct}"
        )

    total = len(results)
    correct_count = sum(row["correct_decision"] for row in results)
    accuracy = correct_count / total if total else 0

    # A false acceptance means the verifier accepts a labelled
    # unsupported answer. A false rejection means it rejects a
    # labelled supported answer.
    unsupported_cases = [
        row for row in results if row["expected_verified"] is False
    ]
    supported_cases = [
        row for row in results if row["expected_verified"] is True
    ]

    false_acceptances = sum(
        row["actual_verified"] for row in unsupported_cases
    )
    false_rejections = sum(
        not row["actual_verified"] for row in supported_cases
    )

    unsupported_acceptance_rate = (
        false_acceptances / len(unsupported_cases)
        if unsupported_cases else 0
    )
    supported_rejection_rate = (
        false_rejections / len(supported_cases)
        if supported_cases else 0
    )

    results_dir = PROJECT_ROOT / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    detail_path = results_dir / "policy_case_results.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary = [
        {
            "metric": "Verifier decision accuracy",
            "actual": f"{accuracy * 100:.2f}%",
            "numerator": correct_count,
            "denominator": total,
            "note": "Agreement with 10 labelled verifier cases; not end-to-end factual accuracy",
        },
        {
            "metric": "Unsupported-claim acceptance rate",
            "actual": f"{unsupported_acceptance_rate * 100:.2f}%",
            "numerator": false_acceptances,
            "denominator": len(unsupported_cases),
            "note": "Lower is better; measured on labelled unsupported cases only",
        },
        {
            "metric": "Supported-answer rejection rate",
            "actual": f"{supported_rejection_rate * 100:.2f}%",
            "numerator": false_rejections,
            "denominator": len(supported_cases),
            "note": "Lower is better; measured on labelled supported cases only",
        },
    ]

    summary_path = results_dir / "policy_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\nPOLICY METRICS")
    for metric in summary:
        print(
            f"{metric['metric']}: {metric['actual']} "
            f"({metric['numerator']}/{metric['denominator']})"
        )

    print(f"\nDetailed results: {detail_path}")
    print(f"Summary results: {summary_path}")


if __name__ == "__main__":
    main()

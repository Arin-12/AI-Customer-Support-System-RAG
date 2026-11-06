
import csv
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.sql import sql_tools


def main():
    results = []

    # Deterministic helper-level cases: no database changes required.
    scenarios = [
        {
            "name": "owner can access own order",
            "order_id": "ORD1005",
            "requester": "USR005",
            "mock_result": {
                "found": True,
                "order": {
                    "order_id": "ORD1005",
                    "user_id": "USR005",
                },
                "items": [{"product_name": "Laptop"}],
            },
            "expected": "allow",
        },
        {
            "name": "other customer is denied",
            "order_id": "ORD1005",
            "requester": "USR001",
            "mock_result": {
                "found": True,
                "order": {
                    "order_id": "ORD1005",
                    "user_id": "USR005",
                },
                "items": [{"product_name": "Private laptop"}],
            },
            "expected": "deny",
        },
        {
            "name": "unknown order returns no details",
            "order_id": "ORD9999",
            "requester": "USR001",
            "mock_result": {
                "found": False,
                "order": None,
                "items": [],
            },
            "expected": "not_found",
        },
    ]

    for case in scenarios:
        with patch.object(
            sql_tools,
            "lookup_order_context",
            return_value=case["mock_result"],
        ):
            result = sql_tools.lookup_order_context_for_user(
                case["order_id"],
                case["requester"],
            )

        if result.get("access_denied") is True:
            actual = "deny"
        elif result.get("found") is True:
            actual = "allow"
        else:
            actual = "not_found"

        # A denial must not contain the private order or item fields.
        no_private_details = (
            "order" not in result and "items" not in result
        )
        safe = (
            actual == case["expected"]
            and (
                case["expected"] != "deny"
                or no_private_details
            )
        )

        results.append({
            "case": case["name"],
            "test_type": "mocked helper",
            "expected": case["expected"],
            "actual": actual,
            "no_private_details_on_denial": (
                no_private_details if actual == "deny" else "N/A"
            ),
            "passed": safe,
            "note": "Deterministic helper-level test; not an API test",
        })

    # Live database case: ensure another customer's order is denied.
    try:
        result = sql_tools.lookup_order_context_for_user(
            "ORD1005",
            "USR001",
        )

        passed = (
            result.get("found") is False
            and result.get("access_denied") is True
            and "order" not in result
            and "items" not in result
        )

        results.append({
            "case": "live database denies another customer",
            "test_type": "live database helper",
            "expected": "deny without details",
            "actual": (
                "deny without details" if passed
                else "unexpected result"
            ),
            "no_private_details_on_denial": passed,
            "passed": passed,
            "note": "Tests helper against configured database",
        })
    except Exception as exc:
        results.append({
            "case": "live database denies another customer",
            "test_type": "live database helper",
            "expected": "deny without details",
            "actual": "ERROR",
            "no_private_details_on_denial": "N/A",
            "passed": False,
            "note": f"Execution failed: {type(exc).__name__}",
        })

    output_dir = PROJECT_ROOT / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "order_security_case_results.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    passed_count = sum(row["passed"] for row in results)
    total = len(results)
    pass_rate = passed_count / total * 100 if total else 0

    summary = [
        {
            "metric": "Ownership scenario pass rate",
            "actual": f"{pass_rate:.2f}%",
            "numerator": passed_count,
            "denominator": total,
            "target": "100%",
            "status": "PASS" if passed_count == total else "FAIL",
            "limitation": "Includes mocked helper cases; not end-to-end API security",
        },
        {
            "metric": "Unauthorized disclosures in tested helper cases",
            "actual": str(sum(
                row["test_type"] == "mocked helper"
                and row["expected"] == "deny"
                and not row["no_private_details_on_denial"]
                for row in results
            )),
            "numerator": 0,
            "denominator": 1,
            "target": "0",
            "status": (
                "PASS" if any(
                    row["case"] == "other customer is denied"
                    and row["passed"]
                    for row in results
                ) else "FAIL"
            ),
            "limitation": "One mocked unauthorized-access case; API not tested",
        },
    ]

    summary_path = output_dir / "order_security_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\nORDER SECURITY RESULTS")
    for row in results:
        print(
            f"{row['case']}: passed={row['passed']} "
            f"| actual={row['actual']}"
        )

    print("\nSUMMARY")
    for row in summary:
        print(
            f"{row['metric']}: {row['actual']} "
            f"| target={row['target']} | {row['status']}"
        )

    print(f"\nDetailed results: {detail_path}")
    print(f"Summary results: {summary_path}")


if __name__ == "__main__":
    main()

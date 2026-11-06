
import pytest

from ai.verification.response_verifier import verify_response


CASES = [
    {
        "name": "supported 7-day return window",
        "answer": (
            "Eligible products may generally be returned within "
            "7 days of delivery, subject to product condition "
            "and category rules."
        ),
        "context": (
            "Eligible products may generally be returned within "
            "7 days of delivery, subject to product condition "
            "and category rules."
        ),
        "expected": True,
    },
    {
        "name": "invented 14-day return window",
        "answer": "You can return any product within 14 days.",
        "context": (
            "Eligible products may generally be returned within "
            "7 days of delivery, subject to product condition "
            "and category rules."
        ),
        "expected": False,
    },
    {
        "name": "unsupported warranty duration",
        "answer": "The laptop has a 2-year warranty.",
        "context": (
            "Warranty terms depend on manufacturer, model, seller, "
            "region, and purchase conditions."
        ),
        "expected": False,
    },
    {
        "name": "unsupported warranty guarantee",
        "answer": "Your laptop is definitely covered under warranty.",
        "context": (
            "Warranty terms depend on manufacturer, model, seller, "
            "region, and purchase conditions."
        ),
        "expected": False,
    },
    {
        "name": "guaranteed refund",
        "answer": "Your refund is guaranteed.",
        "context": (
            "Refund eligibility must be determined after verifying "
            "the order and applicable policy."
        ),
        "expected": False,
    },
    {
        "name": "unconfirmed callback",
        "answer": "A specialist will call you shortly.",
        "context": (
            "Customers may contact support for assistance with their case."
        ),
        "expected": False,
    },
    {
        "name": "qualified refund statement",
        "answer": (
            "A damaged product may qualify for a refund after "
            "the order and applicable policy have been verified."
        ),
        "context": (
            "Eligible cases may include approved returns and damaged "
            "products. Refund eligibility must be determined after "
            "verifying the order and policy."
        ),
        "expected": True,
    },
    {
        "name": "unsupported 30-day return window",
        "answer": "You can return the product within 30 days.",
        "context": (
            "Eligible products may generally be returned within "
            "7 days of delivery, subject to product condition "
            "and category rules."
        ),
        "expected": False,
    },
    {
        "name": "unsupported completed refund claim",
        "answer": "Your money has been refunded.",
        "context": (
            "Refund eligibility must be determined after verifying "
            "the order and applicable policy."
        ),
        "expected": False,
    },
    {
        "name": "unsupported warranty repair eligibility",
        "answer": "Your device is eligible for warranty repair.",
        "context": (
            "Warranty terms depend on manufacturer, model, seller, "
            "region, and purchase conditions."
        ),
        "expected": False,
    },
    {
        "name": "unconfirmed team contact promise",
        "answer": "Our team will contact you shortly.",
        "context": (
            "Customers may contact support for assistance with their case."
        ),
        "expected": False,
    },
    {
        "name": "documented callback promise",
        "answer": "A specialist will call you.",
        "context": (
            "A specialist will call you after the support case is reviewed."
        ),
        "expected": True,
    },
]


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[case["name"] for case in CASES],
)
def test_policy_grounding_evaluation(case):
    result = verify_response(
        answer=case["answer"],
        context=case["context"],
    )

    assert result.verified is case["expected"], (
        f"{case['name']}: expected verified="
        f"{case['expected']}, got {result.verified}; "
        f"reason={result.reason}"
    )

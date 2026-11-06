
import pytest

from ai.ticket.escalation import should_escalate


CASES = [
    {
        "name": "human explicitly requested",
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
        "name": "verified answer with high confidence",
        "query": "Where is my order ORD1005?",
        "confidence": 0.95,
        "verification_failed": False,
        "human_requested": False,
        "expected": False,
    },
]


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[case["name"] for case in CASES],
)
def test_escalation_evaluation(case):
    actual = should_escalate(
        user_query=case["query"],
        confidence=case["confidence"],
        verification_failed=case["verification_failed"],
        human_requested=case["human_requested"],
    )

    assert actual is case["expected"], (
        f"{case['name']}: expected escalation="
        f"{case['expected']}, got {actual}"
    )

from ai.ticket.escalation import should_escalate
from ai.verification.confidence import calculate_confidence
from ai.verification.response_verifier import verify_response


def test_verified_sql_and_rag_does_not_escalate():

    verification = verify_response(
        answer="Your order has been delivered.",
        context="Order ORD1005 has status delivered.",
    )

    confidence = calculate_confidence(
        context_available=True,
        sql_verified=True,
        rag_supported=True,
    )

    escalated = should_escalate(
        user_query="Where is my order ORD1005?",
        confidence=confidence,
        verification_failed=not verification.verified,
    )

    assert verification.verified is True
    assert confidence == 0.95
    assert escalated is False


def test_no_context_escalates():

    verification = verify_response(
        answer="I cannot verify this information.",
        context="",
    )

    confidence = calculate_confidence(
        context_available=False,
    )

    escalated = should_escalate(
        user_query="What happened to my order?",
        confidence=confidence,
        verification_failed=not verification.verified,
    )

    assert verification.verified is False
    assert confidence == 0.20
    assert escalated is True


def test_human_request_escalates():

    verification = verify_response(
        answer="I will connect you with support.",
        context="Customer requested human assistance.",
    )

    confidence = calculate_confidence(
        context_available=True,
        rag_supported=True,
    )

    escalated = should_escalate(
        user_query="I want to speak to a human agent.",
        confidence=confidence,
        human_requested=True,
        verification_failed=not verification.verified,
    )

    assert verification.verified is True
    assert confidence == 0.85
    assert escalated is True
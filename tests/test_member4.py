from ai.ticket.escalation import should_escalate
from ai.verification.confidence import calculate_confidence
from ai.verification.response_verifier import (
    verify_response,
    VerificationResult,
)


def test_escalation_human_requested():
    assert should_escalate(
        user_query="I want to speak to a human",
        confidence=0.95,
        human_requested=True,
    ) is True


def test_escalation_verification_failed():
    assert should_escalate(
        user_query="Where is my order?",
        confidence=0.95,
        verification_failed=True,
    ) is True


def test_escalation_low_confidence():
    assert should_escalate(
        user_query="I have a question",
        confidence=0.50,
    ) is True


def test_escalation_acceptable_confidence():
    assert should_escalate(
        user_query="What is your return policy?",
        confidence=0.85,
    ) is False


def test_confidence_sql_and_rag():
    assert calculate_confidence(
        context_available=True,
        sql_verified=True,
        rag_supported=True,
    ) == 0.95


def test_confidence_sql_only():
    assert calculate_confidence(
        context_available=True,
        sql_verified=True,
        rag_supported=False,
    ) == 0.90


def test_confidence_rag_only():
    assert calculate_confidence(
        context_available=True,
        sql_verified=False,
        rag_supported=True,
    ) == 0.85


def test_confidence_context_only():
    assert calculate_confidence(
        context_available=True,
    ) == 0.60


def test_confidence_no_context():
    assert calculate_confidence(
        context_available=False,
    ) == 0.20


def test_verifier_empty_answer():
    result = verify_response(
        answer="",
        context="Some valid context",
    )

    assert isinstance(result, VerificationResult)
    assert result.verified is False
    assert result.confidence == 0.0


def test_verifier_no_context():
    result = verify_response(
        answer="This is an answer.",
        context="",
    )

    assert result.verified is False
    assert result.confidence == 0.2


def test_verifier_with_context():
    result = verify_response(
        answer="A damaged product may qualify for a refund after verification.",
        context=(
            "A damaged product may qualify for replacement or refund "
            "after verification."
        ),
    )

    assert result.verified is True
    assert result.confidence == 0.8




def test_verifier_rejects_unsupported_return_restrictions():
    context = (
        "Eligible products may generally be returned within 7 days "
        "of delivery, subject to product condition and category rules. "
        "Electronics may require inspection for activation status, "
        "physical damage, missing accessories, and serial-number "
        "verification."
    )

    result = verify_response(
        answer=(
            "All products must be unused and in their original packaging. "
            "Apple, Samsung, and Sony have the same return rules."
        ),
        context=context,
    )

    assert result.verified is False


def test_verifier_accepts_supported_return_window():
    context = (
        "Eligible products may generally be returned within 7 days "
        "of delivery, subject to product condition and category rules."
    )

    result = verify_response(
        answer="Eligible products may generally be returned within 7 days.",
        context=context,
    )

    assert result.verified is True


def test_verifier_rejects_reworded_unsupported_return_claim():
    context = (
        "Eligible products may generally be returned within 7 days "
        "of delivery, subject to product condition and category rules."
    )

    result = verify_response(
        answer="Every product has to be returned unopened.",
        context=context,
    )

    assert result.verified is False


def test_verifier_accepts_supported_answer_with_extra_detail():
    context = (
        "Eligible products may generally be returned within 7 days "
        "of delivery, subject to product condition and category rules."
    )

    result = verify_response(
        answer=(
            "Eligible products may generally be returned within 7 days "
            "of delivery, subject to product condition and category rules."
        ),
        context=context,
    )

    assert result.verified is True


def test_order_id_followup_preserves_previous_issue():
    from ai.agent.orchestrator import AgentOrchestrator
    from ai.agent.router import Route

    class FakeLLM:
        def generate(self, prompt):
            return "I can help check the order for your hardware issue."

    def fake_sql(query, user_id=None):
        assert "ORD1048" in query.upper()
        return {
            "found": True,
            "order": {
                "order_id": "ORD1048",
                "order_status": "delivered",
                "user_id": "USR001",
            },
        }

    orchestrator = AgentOrchestrator(
        llm=FakeLLM(),
        sql_tool=fake_sql,
    )

    first = orchestrator.handle(
        "My laptop has a hardware problem.",
        user_id="USR001",
        conversation_id="test-conversation",
    )

    second = orchestrator.handle(
        "ORD1048",
        user_id="USR001",
        conversation_id="test-conversation",
    )

    assert second.route == Route.SQL
    assert "hardware problem" in second.query.lower()
    assert "ORD1048" in second.query.upper()



def test_verifier_rejects_invented_return_window():
    context = (
        "Eligible products may generally be returned within 7 days "
        "of delivery, subject to product condition and category rules."
    )

    result = verify_response(
        answer="You can return any product within 14 days.",
        context=context,
    )

    assert result.verified is False


def test_verifier_rejects_unsupported_warranty_duration():
    context = (
        "Warranty terms depend on manufacturer, model, seller, "
        "region, and purchase conditions."
    )

    result = verify_response(
        answer="The laptop has a 2-year warranty.",
        context=context,
    )

    assert result.verified is False


def test_verifier_rejects_unsupported_warranty_guarantee():
    context = (
        "Warranty terms depend on manufacturer, model, seller, "
        "region, and purchase conditions."
    )

    result = verify_response(
        answer="Your laptop is definitely covered under warranty.",
        context=context,
    )

    assert result.verified is False


def test_verifier_rejects_guaranteed_refund():
    context = (
        "Refund eligibility must be determined after verifying "
        "the order and applicable policy."
    )

    result = verify_response(
        answer="Your refund is guaranteed.",
        context=context,
    )

    assert result.verified is False


def test_verifier_rejects_unconfirmed_callback():
    context = (
        "Customers may contact support for assistance with their case."
    )

    result = verify_response(
        answer="A specialist will call you shortly.",
        context=context,
    )

    assert result.verified is False


def test_verifier_accepts_supported_refund_qualification():
    context = (
        "Eligible cases may include approved returns and damaged "
        "products. Refund eligibility must be determined after "
        "verifying the order and policy."
    )

    result = verify_response(
        answer=(
            "A damaged product may qualify for a refund after "
            "the order and applicable policy have been verified."
        ),
        context=context,
    )

    assert result.verified is True



def test_orchestrator_clears_unverified_response():
    from ai.agent.orchestrator import AgentOrchestrator
    from ai.agent.router import Route

    class FakeLLM:
        def generate(self, prompt):
            return "You can return any product within 14 days."

    def fake_rag(query):
        return [
            "Eligible products may generally be returned within 7 days "
            "of delivery, subject to product condition and category rules."
        ]

    orchestrator = AgentOrchestrator(
        llm=FakeLLM(),
        rag_tool=fake_rag,
    )

    state = orchestrator.handle(
        "What is your return policy?"
    )

    assert state.route == Route.RAG
    assert state.requires_escalation is True
    assert state.response is None
    assert state.confidence <= 0.20



def test_issue_context_does_not_leak_between_conversations():
    from ai.agent.orchestrator import AgentOrchestrator

    class FakeLLM:
        def generate(self, prompt):
            return "I can help with your question."

    orchestrator = AgentOrchestrator(llm=FakeLLM())

    orchestrator.handle(
        "My laptop has a hardware problem.",
        user_id="USR001",
        conversation_id="conversation-A",
    )

    second = orchestrator.handle(
        "ORD1048",
        user_id="USR001",
        conversation_id="conversation-B",
    )

    assert "hardware problem" not in second.query.lower()
    assert second.query.strip().upper() == "ORD1048"




def test_issue_context_does_not_leak_between_users():
    from ai.agent.orchestrator import AgentOrchestrator

    class FakeLLM:
        def generate(self, prompt):
            return "I can help with your question."

    orchestrator = AgentOrchestrator(llm=FakeLLM())

    orchestrator.handle(
        "My laptop has a hardware problem.",
        user_id="USR001",
        conversation_id="shared-conversation",
    )

    second = orchestrator.handle(
        "ORD1048",
        user_id="USR002",
        conversation_id="shared-conversation",
    )

    assert "hardware problem" not in second.query.lower()
    assert second.query.strip().upper() == "ORD1048"

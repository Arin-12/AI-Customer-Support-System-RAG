
import pytest

from ai.agent.router import Route, route_query


@pytest.mark.parametrize(
    ("scenario", "query", "expected_route"),
    [
        (
            "return policy",
            "What is your return policy?",
            Route.RAG,
        ),
        (
            "order lookup",
            "Where is order ORD1002?",
            Route.SQL,
        ),
        (
            "combined order and policy",
            "My order ORD1005 contains a damaged product. "
            "Can I get a refund?",
            Route.SQL_AND_RAG,
        ),
        (
            "human support request",
            "I want to talk to a human agent",
            Route.TICKET,
        ),
        (
            "vague help request",
            "Can you help me?",
            Route.UNKNOWN,
        ),
        (
            "product specifications",
            "What are the laptop specifications?",
            Route.RAG,
        ),
        (
            "unknown order ID",
            "Where is order ORD9999?",
            Route.SQL,
        ),

        
        (
            "general cancellation policy",
            "What is your cancellation policy?",
            Route.RAG,
        ),
        (
            "order cancellation",
            "Can I cancel order ORD1005?",
            Route.SQL_AND_RAG,
        ),
        (
            "specific order tracking",
            "Can you track order ORD1005?",
            Route.SQL,
        ),
        (
            "human request with order ID",
            "I want a human agent to help me with order ORD1005",
            Route.TICKET,
        ),

                (
            "general exchange question",
            "Can I exchange a product I purchased?",
            Route.RAG,
        ),
        (
            "general order cancellation policy",
            "How can I cancel my order?",
            Route.RAG,
        ),
        (
            "general shipment tracking question",
            "How can I track my shipment and delivery?",
            Route.RAG,
        ),

    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_routing_evaluation(scenario, query, expected_route):
    actual_route = route_query(query)

    assert actual_route == expected_route, (
        f"Scenario: {scenario}; "
        f"expected {expected_route}, got {actual_route}"
    )

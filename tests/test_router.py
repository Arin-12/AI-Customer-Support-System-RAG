from ai.agent.router import Route, route_query


def test_policy_query():
    result = route_query("What is your return policy?")
    assert result == Route.RAG


def test_order_query():
    result = route_query("Where is order ORD1002?")
    assert result == Route.SQL


def test_combined_query():
    result = route_query(
        "My order ORD1005 contains a damaged product. Can I get a refund?"
    )
    assert result == Route.SQL_AND_RAG


def test_human_request():
    result = route_query("I want to talk to a human agent")
    assert result == Route.TICKET
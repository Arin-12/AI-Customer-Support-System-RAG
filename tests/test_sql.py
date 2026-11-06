from ai.sql.sql_tools import (
    lookup_order,
    lookup_order_context,
    lookup_user_orders,
    lookup_user_tickets,
)
from ai.sql.sql_tools import lookup_product
from ai.sql.pipeline import execute_sql_operation


def test_lookup_order():
    result = lookup_order("ORD1005")

    assert result is not None
    assert result["order_id"] == "ORD1005"
    assert result["user_id"] == "USR005"


def test_lookup_order_context():
    result = lookup_order_context("ORD1005")

    assert result is not None
    assert result["found"] is True
    assert result["order"]["order_id"] == "ORD1005"
    assert len(result["items"]) > 0


def test_lookup_user_orders():
    result = lookup_user_orders("USR005")

    assert len(result) > 0
    assert result[0]["user_id"] == "USR005"


def test_lookup_user_tickets():
    result = lookup_user_tickets("USR005")

    assert len(result) > 0
    assert result[0]["user_id"] == "USR005"


def test_lookup_product():
    result = lookup_product("PROD035")

    assert result is not None
    assert result["product_id"] == "PROD035"
    assert result["product_name"] == "IdeaPad Slim 5"


def test_execute_sql_operation():
    result = execute_sql_operation("order", "ORD1005")

    assert result is not None
    assert result["order_id"] == "ORD1005"


def test_execute_sql_operation_invalid():
    try:
        execute_sql_operation("unknown", "TEST")
        assert False
    except ValueError as error:
        assert "Unsupported SQL operation" in str(error)

def test_lookup_product_not_found():
    result = lookup_product("PROD999")

    assert result is None

def test_lookup_order_context_not_found():
    result = lookup_order_context("ORD9999")

    assert result["found"] is False
    assert result["order"] is None
    assert result["items"] == []


    
def test_order_ownership_allows_owner(monkeypatch):
    import ai.sql.sql_tools as sql_tools

    def fake_lookup_order_context(order_id):
        return {
            "found": True,
            "order": {
                "order_id": order_id,
                "user_id": "USR005",
            },
            "items": [{"product_name": "Laptop"}],
        }

    monkeypatch.setattr(
        sql_tools,
        "lookup_order_context",
        fake_lookup_order_context,
    )

    result = sql_tools.lookup_order_context_for_user(
        "ORD1005",
        "USR005",
    )

    assert result["found"] is True
    assert result["order"]["order_id"] == "ORD1005"
    assert result["order"]["user_id"] == "USR005"


def test_order_ownership_denies_other_customer(monkeypatch):
    import ai.sql.sql_tools as sql_tools

    def fake_lookup_order_context(order_id):
        return {
            "found": True,
            "order": {
                "order_id": order_id,
                "user_id": "USR005",
            },
            "items": [{"product_name": "Private laptop"}],
        }

    monkeypatch.setattr(
        sql_tools,
        "lookup_order_context",
        fake_lookup_order_context,
    )

    result = sql_tools.lookup_order_context_for_user(
        "ORD1005",
        "USR001",
    )

    assert result["found"] is False
    assert result["access_denied"] is True
    assert "order" not in result
    assert "items" not in result


def test_order_ownership_handles_unknown_order(monkeypatch):
    import ai.sql.sql_tools as sql_tools

    def fake_lookup_order_context(order_id):
        return {
            "found": False,
            "order": None,
            "items": [],
        }

    monkeypatch.setattr(
        sql_tools,
        "lookup_order_context",
        fake_lookup_order_context,
    )

    result = sql_tools.lookup_order_context_for_user(
        "ORD9999",
        "USR001",
    )

    assert result["found"] is False
    assert result["order"] is None
    assert result["items"] == []
    assert not result.get("access_denied", False)



def test_live_database_denies_another_customers_order():
    from ai.sql.sql_tools import lookup_order_context_for_user

    result = lookup_order_context_for_user(
        "ORD1005",
        "USR001",
    )

    assert result["found"] is False
    assert result["access_denied"] is True
    assert "order" not in result
    assert "items" not in result

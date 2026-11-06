from ai.sql.queries import (
    get_order_by_id,
    get_order_items,
)
from ai.sql.queries import (
    get_order_by_id,
    get_order_items,
    get_user_orders,
    get_user_tickets,
    get_product_by_id,
)


def get_order_context(order_id: str):
    order = get_order_by_id(order_id)

    if order is None:
        return {
            "found": False,
            "order": None,
            "items": []
        }

    items = get_order_items(order_id)

    return {
        "found": True,
        "order": order,
        "items": items
    }

def execute_sql_operation(operation: str, identifier: str):
    """
    Execute a supported SQL operation.

    Supported operations:
    - order
    - order_context
    - order_items
    - user_orders
    - user_tickets
    - product
    """

    if operation == "order":
        return get_order_by_id(identifier)

    if operation == "order_context":
        return get_order_context(identifier)

    if operation == "order_items":
        return get_order_items(identifier)

    if operation == "user_orders":
        return get_user_orders(identifier)

    if operation == "user_tickets":
        return get_user_tickets(identifier)

    if operation == "product":
        return get_product_by_id(identifier)

    raise ValueError(f"Unsupported SQL operation: {operation}")
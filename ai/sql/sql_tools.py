from ai.sql.pipeline import get_order_context

from ai.sql.queries import (
    get_order_by_id,
    get_order_items,
    get_user_orders,
    get_user_tickets,
    get_product_by_id,
)


SQL_TOOL_DESCRIPTIONS = {
    "lookup_order": "Look up an order using an order ID.",
    "lookup_order_items": "Get the products/items contained in an order.",
    "lookup_order_context": "Get complete order information and its products.",
    "lookup_user_orders": "Get all orders belonging to a customer using a user ID.",
    "lookup_user_tickets": "Get support tickets belonging to a customer using a user ID.",
    "lookup_product": "Look up product information using a product ID.",
}


def lookup_order(order_id: str):
    """
    Look up a customer's order using the order ID.
    """

    return get_order_by_id(order_id)


def lookup_order_items(order_id: str):
    """
    Get all products/items contained in an order.
    """

    return get_order_items(order_id)


def lookup_order_context(order_id: str):
    """
    Get complete order information including its products.
    """

    return get_order_context(order_id)


def lookup_order_context_for_user(
    order_id: str,
    user_id: str,
) -> dict:
    """
    Return order details only when the order belongs
    to the requesting customer.
    """

    result = lookup_order_context(order_id)

    # Order does not exist.
    if not result.get("found"):
        return result

    order = result.get("order") or {}

    # Order belongs to another customer.
    if order.get("user_id") != user_id:
        return {
            "found": False,
            "access_denied": True,
        }

    # Order belongs to the requesting customer.
    return result


def lookup_user_orders(user_id: str):
    """
    Get all orders belonging to a customer.
    """

    return get_user_orders(user_id)


def lookup_user_tickets(user_id: str):
    """
    Get support tickets belonging to a customer.
    """

    return get_user_tickets(user_id)


def lookup_product(product_id: str):
    """
    Look up product information using the product ID.
    """

    return get_product_by_id(product_id)
from sqlalchemy import text

from ai.sql.database import engine


def get_order_by_id(order_id: str):
    query = text("""
        SELECT
            order_id,
            user_id,
            order_status,
            total_amount_inr,
            tracking_number,
            expected_delivery_date,
            delivery_date,
            created_at
        FROM orders
        WHERE order_id = :order_id
    """)

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"order_id": order_id}
        ).mappings().first()

    if result is None:
        return None

    return dict(result)
def get_order_items(order_id: str):
    query = text("""
        SELECT
            oi.item_id,
            oi.order_id,
            oi.product_id,
            p.product_name,
            p.brand,
            p.category,
            p.variant,
            oi.quantity,
            oi.unit_price_inr
        FROM order_items oi
        JOIN products p
            ON oi.product_id = p.product_id
        WHERE oi.order_id = :order_id
        ORDER BY oi.item_id
    """)

    with engine.connect() as connection:
        results = connection.execute(
            query,
            {"order_id": order_id}
        ).mappings().all()

    return [dict(row) for row in results]

def get_user_orders(user_id: str):
    query = text("""
        SELECT
            order_id,
            user_id,
            order_status,
            total_amount_inr,
            tracking_number,
            expected_delivery_date,
            delivery_date,
            created_at
        FROM orders
        WHERE user_id = :user_id
        ORDER BY created_at DESC
    """)

    with engine.connect() as connection:
        results = connection.execute(
            query,
            {"user_id": user_id}
        ).mappings().all()

    return [dict(row) for row in results]

def get_user_tickets(user_id: str):
    query = text("""
        SELECT
            ticket_id,
            user_id,
            order_id,
            category,
            priority,
            status,
            customer_message,
            ai_response,
            created_at,
            updated_at
        FROM support_tickets
        WHERE user_id = :user_id
        ORDER BY created_at DESC
    """)

    with engine.connect() as connection:
        results = connection.execute(
            query,
            {"user_id": user_id}
        ).mappings().all()

    return [dict(row) for row in results]

def get_product_by_id(product_id: str):
    query = text("""
        SELECT
            product_id,
            brand,
            product_name,
            category,
            variant,
            price_inr,
            description,
            stock_quantity,
            status
        FROM products
        WHERE product_id = :product_id
    """)

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"product_id": product_id}
        ).mappings().first()

    if result is None:
        return None

    return dict(result)
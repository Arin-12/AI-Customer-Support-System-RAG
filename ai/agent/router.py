from enum import Enum
import re


class Route(str, Enum):
    RAG = "rag"
    SQL = "sql"
    SQL_AND_RAG = "sql_and_rag"
    TICKET = "ticket"
    UNKNOWN = "unknown"




def route_query(query: str) -> Route:
    """Determine which system should handle the customer query."""

    query_lower = query.lower().strip()

    # Explicit requests for a human take priority.
    human_phrases = [
        "human",
        "human agent",
        "real person",
        "real support representative",
        "support representative",
        "speak to a person",
        "talk to a person",
        "talk to someone",
        "speak to someone",
        "speak with someone",
        "talk with someone",
        "customer support",
        "connect me",
        "connect me to support",
        "connect me with support",
        "live agent",
        "representative",
        "escalate this",
    ]

    if any(phrase in query_lower for phrase in human_phrases):
        return Route.TICKET

    # An explicit order ID is strong evidence of an order-specific query.
    has_order_id = bool(re.search(r"\bORD\d+\b", query.upper()))

    policy_case_keywords = [
        "refund", "return", "exchange", "cancel", "cancellation",
        "damage", "damaged", "broken", "defective",
    ]

    has_policy_case = any(
        keyword in query_lower for keyword in policy_case_keywords
    )

    # Policy questions mentioning a specific order need both systems.
    if has_order_id and has_policy_case:
        return Route.SQL_AND_RAG

    # Specific order IDs without policy issues go to SQL.
    if has_order_id:
        return Route.SQL

    # Product information questions go to RAG.
    product_keywords = [
        "iphone", "samsung", "galaxy", "pixel", "oneplus", "xiaomi",
        "redmi", "realme", "sony", "laptop", "phone", "smartphone",
        "tablet", "product", "model", "specification", "specifications",
        "features", "price", "cost", "storage", "ram",
    ]

    if any(keyword in query_lower for keyword in product_keywords):
        return Route.RAG

    # General policy and FAQ questions go to RAG.
    rag_keywords = [
        "policy", "return", "refund", "exchange", "shipping", "payment",
        "warranty", "faq", "damage", "damaged", "broken", "defective",
        "how long", "how can i", "cancel", "cancellation",
        "tracking", "track", "delivery",
    ]

    if any(keyword in query_lower for keyword in rag_keywords):
        return Route.RAG

    # Order-specific language without an ID is handled conservatively.
    sql_keywords = [
        "where is my", "my order", "my orders", "my purchase",
        "my shipment", "my delivery", "order status",
        "track my order", "track my package",
    ]

    if any(keyword in query_lower for keyword in sql_keywords):
        return Route.SQL

    return Route.UNKNOWN


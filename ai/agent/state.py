from dataclasses import dataclass, field
from typing import Any, Optional

from .router import Route


@dataclass
class AgentState:
    query: str

    user_id: Optional[str] = None

    route: Optional[Route] = None

    rag_context: list[str] = field(
        default_factory=list
    )

    sql_context: dict[str, Any] = field(
        default_factory=dict
    )

    response: Optional[str] = None

    confidence: Optional[float] = None

    requires_escalation: bool = False

    ticket_id: Optional[str] = None

    # -------------------------------------------------
    # Multi-order selection
    # -------------------------------------------------

    order_selection_required: bool = False

    available_orders: list[dict[str, Any]] = field(
        default_factory=list
    )
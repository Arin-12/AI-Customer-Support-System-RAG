import re
from typing import Callable, Any
from uuid import UUID

from ai.rag.pipeline import retrieve_context

from ai.sql.sql_tools import (
    lookup_order_context,
    lookup_order_context_for_user,
    lookup_user_orders,
)

from ai.verification.confidence import calculate_confidence
from ai.verification.response_verifier import verify_response
from ai.ticket.escalation import should_escalate

from .router import Route, route_query
from .state import AgentState
from .prompts import SYSTEM_PROMPT, build_user_prompt
from ai.llm.llm import LLMProvider


def rag_tool(query: str) -> list[str]:
    """
    Adapter between the RAG pipeline and the agent.
    """

    result = retrieve_context(query, k=5)

    if not result["context"]:
        return []

    return [result["context"]]



def sql_tool(
    query: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve one or multiple orders from a customer query.

    When user_id is provided, each order is checked
    against the requesting customer's ID.
    """

    order_ids = list(dict.fromkeys(
        re.findall(r"\bORD\d+\b", query.upper())
    ))

    if not order_ids:
        raise ValueError(
            "No order ID found in the customer query."
        )

    # Preserve the existing single-order behavior.
    if len(order_ids) == 1:
        order_id = order_ids[0]

        if user_id is not None:
            return lookup_order_context_for_user(
                order_id=order_id,
                user_id=user_id,
            )

        return lookup_order_context(order_id)

    # Retrieve every explicitly requested order.
    orders = []
    missing_orders = []
    denied_orders = []

    for order_id in order_ids:
        if user_id is not None:
            result = lookup_order_context_for_user(
                order_id=order_id,
                user_id=user_id,
            )
        else:
            result = lookup_order_context(order_id)

        if result.get("access_denied"):
            denied_orders.append(order_id)
        elif not result.get("found"):
            missing_orders.append(order_id)
        else:
            orders.append(result)

    return {
        "found": len(orders) == len(order_ids),
        "orders": orders,
        "missing_orders": missing_orders,
        "access_denied_orders": denied_orders,
    }


def user_orders_tool(
    user_id: str,
) -> list[dict[str, Any]]:
    """Retrieve all orders belonging to the customer."""
    return lookup_user_orders(user_id)


def is_product_query(query: str) -> bool:
    """Detect common product-information queries for RAG."""
    product_keywords = [
        "iphone", "samsung", "galaxy", "pixel", "oneplus",
        "xiaomi", "redmi", "realme", "sony", "laptop",
        "phone", "smartphone", "tablet", "product", "model",
        "specification", "specifications", "features", "price",
        "cost", "storage", "ram",
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in product_keywords)


class AgentOrchestrator:
    """
    Coordinates routing, RAG, SQL, response generation,
    verification, confidence calculation, and escalation.
    """

    def __init__(
        self,
        llm: LLMProvider,
        rag_tool: Callable[[str], list[str]] | None = None,
        sql_tool: Callable[..., dict[str, Any]] | None = None,
        ticket_tool: Callable[[str], str] | None = None,
    ):
        self.llm = llm
        self.rag_tool = rag_tool
        self.sql_tool = sql_tool
        self.ticket_tool = ticket_tool
        self._conversation_issues: dict[tuple[str, str], str] = {}

    def handle(
        self,
        query: str,
        user_id: str | None = None,
        conversation_id: UUID | str | None = None,
    ) -> AgentState:
        """
        Process a customer query and return the final agent state.
        """

        original_query = query
        is_vague_help = False

        conversation_key = None

        if user_id is not None and conversation_id is not None:
            conversation_key = (
                str(user_id),
                str(conversation_id),
            )

        # If the customer sends only an order ID, attach their
        # previous issue from this same conversation.
        order_only = re.fullmatch(
            r"\s*ORD\d+\s*[.!]?\s*",
            query,
            flags=re.IGNORECASE,
        )

        if order_only and conversation_key is not None:
            previous_issue = self._conversation_issues.get(
                conversation_key
            )

            # print(f"DEBUG lookup key: {conversation_key}")
            # print(f"DEBUG previous issue found: {previous_issue!r}")


            if previous_issue:
                query = f"{previous_issue} Order ID: {query.strip()}"
                # print(f"DEBUG combined follow-up query: {query}")

        state = AgentState(
            query=query,
            user_id=user_id,
        )

        # -------------------------------------------------
        # Step 1: Determine route
        # -------------------------------------------------
        state.route = route_query(query)

        # Remember explicit issue descriptions for later order-ID follow-ups.
        if conversation_key is not None:
            query_lower = original_query.lower()

            issue_indicators = (
                "problem",
                "issue",
                "defective",
                "defect",
                "damaged",
                "broken",
                "not working",
                "hardware",
                "malfunction",
            )

            if (
                not order_only
                and any(word in query_lower for word in issue_indicators)
            ):
                self._conversation_issues[conversation_key] = original_query
                # print(f"DEBUG remembered issue for {conversation_key}: {original_query}")

        # Product-information questions should use RAG.
        if state.route == Route.UNKNOWN and is_product_query(query):
            state.route = Route.RAG

        # -------------------------------------------------
        # Step 2: Execute selected tools
        # -------------------------------------------------
        if state.route == Route.RAG:
            self._run_rag(state)

        elif state.route == Route.SQL:
            self._run_sql(state)

        elif state.route == Route.SQL_AND_RAG:
            self._run_sql(state)

            # If SQL cannot run because the customer did not provide
            # an order ID, still retrieve the general policy information
            # from RAG.
            if not state.sql_context and not state.response:
                state.requires_escalation = False
                self._run_rag(state)

            elif not state.requires_escalation:
                self._run_rag(state)

        elif state.route == Route.TICKET:
            self._run_ticket(state)



        elif state.route == Route.UNKNOWN:
            query_lower = query.lower().strip()

            vague_help_phrases = [
                "can you help me",
                "could you help me",
                "i need help",
                "i have a problem",
                "i have an issue",
                "help me",
                "need assistance",
            ]

            if any(
                phrase in query_lower
                for phrase in vague_help_phrases
            ):
                is_vague_help = True
                state.response = (
                    "Of course! I'm here to help. What do you need "
                    "assistance with? You can ask about an order, "
                    "a return or refund, a damaged product, product "
                    "information, or something else."
                )
                state.requires_escalation = False
                state.confidence = 0.80
            else:
                state.requires_escalation = True
                state.confidence = 0.20



        # -------------------------------------------------
        # Step 3: Generate LLM response
        # -------------------------------------------------
        if not state.requires_escalation and not state.response:
            prompt = self._build_prompt(state)

            try:
                state.response = self.llm.generate(prompt)

            except Exception:
                state.requires_escalation = True
                state.confidence = 0.20
                return state

        # -------------------------------------------------
        # Step 4: Verification + confidence
        # -------------------------------------------------
        if (
            state.response
            and not state.requires_escalation
            and not is_vague_help
        ):

            verification_context = (
                self._build_verification_context(state)
            )

            verification = verify_response(
                answer=state.response,
                context=verification_context,
            )

            sql_verified = bool(state.sql_context)

            if "found" in state.sql_context:
                sql_verified = state.sql_context["found"]

            rag_supported = bool(state.rag_context)

            context_available = bool(
                verification_context.strip()
            )

            state.confidence = calculate_confidence(
                context_available=context_available,
                sql_verified=sql_verified,
                rag_supported=rag_supported,
            )

            # Verification failure forces escalation.
            # Do not allow an unverified answer to reach the customer.
            if not verification.verified:
                state.response = None
                state.requires_escalation = True
                state.confidence = min(state.confidence, 0.20)
                return state

            # Apply escalation policy.
            if not is_vague_help:
                state.requires_escalation = should_escalate(
                    user_query=state.query,
                    confidence=state.confidence,
                    human_requested=(
                        state.route == Route.TICKET
                    ),
                    verification_failed=(
                        not verification.verified
                    ),
                )

        return state

    def _run_rag(
        self,
        state: AgentState,
    ) -> None:
        """
        Execute the RAG tool.
        """

        if self.rag_tool is None:
            state.requires_escalation = True
            return

        try:
            state.rag_context = self.rag_tool(
                state.query
            )

        except Exception:
            state.requires_escalation = True

    def _run_sql(
        self,
        state: AgentState,
    ) -> None:
        """
        Execute customer/order SQL operations.

        If an explicit order ID is present, verify and retrieve
        that order.

        If no order ID is present, retrieve the customer's orders.
        """

        if self.sql_tool is None:
            state.requires_escalation = True
            return

        # -------------------------------------------------
        # Check whether an order ID was supplied
        # -------------------------------------------------

        match = re.search(
            r"\bORD\d+\b",
            state.query.upper(),
        )

        # -------------------------------------------------
        # No order ID:
        # retrieve customer's orders
        # -------------------------------------------------

        if match is None:

            if state.user_id is None:
                state.requires_escalation = True
                return

            try:
                orders = lookup_user_orders(state.user_id)

                if not orders:
                    state.response = (
                        "I couldn't find any orders associated "
                        "with your account."
                    )
                    state.confidence = 0.20
                    state.requires_escalation = True
                    return

                # Exactly one order: use it automatically.
                if len(orders) == 1:
                    state.sql_context = {
                        "found": True,
                        "order": orders[0],
                        "orders": orders,
                    }
                    return

        
                # Multiple orders: ask the customer to choose.
                state.sql_context = {
                    "found": True,
                    "orders": orders,
                }
                
                lines = [
                    "I found multiple orders associated with your account:",
                    "",
                ]

                for index, order in enumerate(orders, start=1):
                    order_id = order.get("order_id", "Unknown order")
                    status = order.get("order_status", "Unknown status")
                    lines.append(f"{index}. {order_id} — {status}")

                example_order_id = orders[0].get("order_id", "your order")

                lines.extend([
                    "",
                    "Please tell me which order you would like help with "
                    "and what you would like to know.",
                    "",
                    f'Example: "{example_order_id} — what is the tracking number?"',
                ])


                state.response = "\n".join(lines)
                state.confidence = 0.20
                state.requires_escalation = False
                return

            except Exception:
                state.requires_escalation = True
                state.confidence = 0.20
                return

        
        # -------------------------------------------------
        # Explicit order ID
        # -------------------------------------------------

        try:
            if state.user_id is not None:
                state.sql_context = self.sql_tool(
                    state.query,
                    state.user_id,
                )
            else:
                state.sql_context = self.sql_tool(
                    state.query
                )

            # -------------------------------------------------
            # Handle multiple order IDs
            # -------------------------------------------------
            if "orders" in state.sql_context:
                requested_orders = state.sql_context["orders"]
                missing_orders = state.sql_context.get(
                    "missing_orders", []
                )
                denied_orders = state.sql_context.get(
                    "access_denied_orders", []
                )

                if denied_orders:
                    state.sql_context = {}
                    state.response = (
                        "I can only provide order details "
                        "that belong to your account. "
                        "Please check the order IDs and try again."
                    )
                    state.confidence = 0.20
                    state.requires_escalation = True
                    return

                if missing_orders:
                    state.response = (
                        "I could verify the following requested "
                        "order(s): "
                        + ", ".join(
                            item["order"]["order_id"]
                            for item in requested_orders
                        )
                        + ". I couldn't verify: "
                        + ", ".join(missing_orders)
                        + ". Please check those order IDs."
                    )
                    state.confidence = 0.20
                    state.requires_escalation = True
                    return

                state.sql_context = {
                    "found": True,
                    "orders": requested_orders,
                }

            # -------------------------------------------------
            # Existing ownership protection
            # -------------------------------------------------
            if state.sql_context.get("access_denied"):
                state.response = (
                    "I’m unable to verify or provide details "
                    "for that order for this account."
                )
                state.confidence = 0.20
                state.requires_escalation = True

        except Exception:
            if state.route == Route.SQL_AND_RAG:
                state.sql_context = {}
            else:
                state.requires_escalation = True
                state.confidence = 0.20


    def _run_ticket(self, state: AgentState):
        if self.ticket_tool is None:
            state.requires_escalation = True
            state.confidence = 0.20
            return

        try:
            state.ticket_id = self.ticket_tool(
                state.query,
                state.user_id,
            )
        except TypeError:
            # Backward compatibility with simple test/mock ticket tools
            # that accept only the customer query.
            state.ticket_id = self.ticket_tool(state.query)

        state.requires_escalation = True
        state.confidence = 0.20

    def _build_verification_context(
        self,
        state: AgentState,
    ) -> str:
        """
        Build the context used by the response verifier.
        """

        context_parts = []

        if state.rag_context:
            context_parts.extend(
                state.rag_context
            )

        if state.sql_context:
            context_parts.append(
                f"Verified SQL context:\n"
                f"{state.sql_context}"
            )

        return "\n\n".join(
            context_parts
        )

    def _build_prompt(
        self,
        state: AgentState,
    ) -> str:
        """
        Build the final LLM prompt.
        """

        return (
            SYSTEM_PROMPT
            + "\n\n"
            + build_user_prompt(
                query=state.query,
                rag_context=state.rag_context,
                sql_context=state.sql_context,
            )
        )
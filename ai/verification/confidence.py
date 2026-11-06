def calculate_confidence(
    context_available: bool,
    sql_verified: bool = False,
    rag_supported: bool = False
) -> float:

    if sql_verified and rag_supported:
        return 0.95

    if sql_verified:
        return 0.90

    if rag_supported:
        return 0.85

    if context_available:
        return 0.60

    return 0.20
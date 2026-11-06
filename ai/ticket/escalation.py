def should_escalate(
    user_query,
    confidence,
    human_requested=False,
    verification_failed=False
):
    """
    Determine whether the conversation should be escalated.
    """

    if human_requested:
        return True

    if verification_failed:
        return True

    if confidence < 0.60:
        return True

    return False
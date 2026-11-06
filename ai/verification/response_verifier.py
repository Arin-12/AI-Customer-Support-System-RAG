
from dataclasses import dataclass
import re


@dataclass
class VerificationResult:
    verified: bool
    confidence: float
    reason: str


# Claims that require explicit supporting evidence.
RETURN_CLAIM_GROUPS = {
    "unused or unopened condition": (
        "unused",
        "unopened",
        "not used",
        "never used",
        "must be unused",
        "must remain unused",
    ),
    "original packaging requirement": (
        "original packaging",
        "original package",
        "original box",
        "in its original packaging",
    ),
    "identical brand return rules": (
        "same return rules",
        "identical return rules",
        "same return policy",
        "identical return policy",
    ),
}

# Strong statements that imply a refund is guaranteed.
REFUND_GUARANTEE_PHRASES = (
    "will receive a refund",
    "you are guaranteed a refund",
    "refund is guaranteed",
    "will be refunded",
    "your refund has been approved",
    "your money has been refunded",
)

# Strong statements that imply warranty eligibility is confirmed.
WARRANTY_GUARANTEE_PHRASES = (
    "is covered under warranty",
    "is definitely covered by warranty",
    "your warranty claim is approved",
    "your laptop qualifies for warranty",
    "your device is eligible for warranty repair",
    "will be repaired under warranty",
    "will be replaced under warranty",
    "your warranty is guaranteed",
    "warranty is guaranteed",
    "guaranteed warranty",
    "definitely covered under warranty",
    "definitely covered by warranty",
    "covered under warranty",
    "covered by warranty",
    "eligible for warranty repair",
    "qualifies for warranty",
)

# Claims about a specific warranty duration.

WARRANTY_DURATION_PATTERNS = (
    r"\b\d+\s*[- ]?\s*year(?:s)?\s+warranty\b",
    r"\b\d+\s*[- ]?\s*month(?:s)?\s+warranty\b",
    r"\bwarranty\s+(?:period|duration)\s+(?:is|of|for)\s+"
    r"\d+\s*[- ]?\s*(?:years?|months?)\b",
    r"\bwarranty\s+(?:lasts|is valid for|is for|covers for)\s+"
    r"\d+\s*[- ]?\s*(?:years?|months?)\b",
    r"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*[- ]?\s*year(?:s)?\s+(?:manufacturer\s+)?warranty\b",
    r"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*[- ]?\s*month(?:s)?\s+(?:manufacturer\s+)?warranty\b",
)


# Do not promise contact unless a real support workflow confirms it.
CALLBACK_GUARANTEE_PHRASES = (
    "a specialist will call you",
    "our specialist will call you",
    "an agent will call you",
    "someone will contact you shortly",
    "our team will contact you shortly",
    "you will receive a call",
)

def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _extract_return_windows(text: str) -> set[int]:
    """Extract numeric day-based return windows from text."""
    patterns = (
        r"\bwithin\s+(\d+)\s+days?\b",
        r"\b(\d+)[- ]day\s+return\s+window\b",
        r"\breturn(?:ed|ing)?\s+within\s+(\d+)\s+days?\b",
    )

    windows = set()

    for pattern in patterns:
        windows.update(
            int(match)
            for match in re.findall(pattern, text.lower())
        )

    return windows

def verify_response(
    answer: str,
    context: str,
) -> VerificationResult:

    if not answer.strip():
        return VerificationResult(
            verified=False,
            confidence=0.0,
            reason="Empty response",
        )

    if not context.strip():
        return VerificationResult(
            verified=False,
            confidence=0.2,
            reason="No supporting context",
        )

    answer_lower = answer.lower()
    context_lower = context.lower()

    answer_return_windows = _extract_return_windows(answer_lower)
    context_return_windows = _extract_return_windows(context_lower)

    if (
        answer_return_windows
        and not answer_return_windows.issubset(context_return_windows)
    ):
        return VerificationResult(
            verified=False,
            confidence=0.2,
            reason="Return window lacks supporting context",
        )

    for claim_name, phrases in RETURN_CLAIM_GROUPS.items():
        if (
            _has_any(answer_lower, phrases)
            and not _has_any(context_lower, phrases)
        ):
            return VerificationResult(
                verified=False,
                confidence=0.2,
                reason=f"Claim lacks explicit supporting context: {claim_name}",
            )

    if (
        _has_any(answer_lower, REFUND_GUARANTEE_PHRASES)
        and not _has_any(context_lower, REFUND_GUARANTEE_PHRASES)
    ):
        return VerificationResult(
            verified=False,
            confidence=0.2,
            reason="Refund guarantee lacks supporting context",
        )

    if (
        _has_any(answer_lower, WARRANTY_GUARANTEE_PHRASES)
        and not _has_any(context_lower, WARRANTY_GUARANTEE_PHRASES)
    ):
        return VerificationResult(
            verified=False,
            confidence=0.2,
            reason="Warranty eligibility lacks supporting context",
        )

    for pattern in WARRANTY_DURATION_PATTERNS:
        if re.search(pattern, answer_lower):
            if not re.search(pattern, context_lower):
                return VerificationResult(
                    verified=False,
                    confidence=0.2,
                    reason="Warranty duration lacks supporting context",
                )

    if (
        _has_any(answer_lower, CALLBACK_GUARANTEE_PHRASES)
        and not _has_any(context_lower, CALLBACK_GUARANTEE_PHRASES)
    ):
        return VerificationResult(
            verified=False,
            confidence=0.2,
            reason="Callback promise lacks supporting context",
        )

    return VerificationResult(
        verified=True,
        confidence=0.8,
        reason="No unsupported claims detected by configured checks",
    )

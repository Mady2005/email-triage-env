from __future__ import annotations

from .models import EmailAction, EmailRecord, EmailReward


WILDLY_WRONG = {
    ("urgent", "spam"),
    ("spam", "urgent"),
    ("urgent", "normal"),
    ("spam", "inquiry"),
}


def _normalize_text(text: str | None) -> str:
    return (text or "").strip().lower()


def compute_reply_quality(reply_body: str | None, email: EmailRecord) -> dict[str, float]:
    reply_text = _normalize_text(reply_body)
    quality = {
        "reply_sender_name": 0.0,
        "reply_keywords": 0.0,
        "reply_length": 0.0,
    }
    if not reply_text:
        return quality

    if email.sender_name.lower() in reply_text:
        quality["reply_sender_name"] = 0.15

    keyword_hits = sum(1 for keyword in email.required_keywords if keyword.lower() in reply_text)
    if email.required_keywords and keyword_hits / len(email.required_keywords) >= 0.5:
        quality["reply_keywords"] = 0.15

    if 50 <= len(reply_body or "") <= 500:
        quality["reply_length"] = 0.10

    return quality


def compute_reward(action: EmailAction, email: EmailRecord) -> EmailReward:
    breakdown: dict[str, float] = {
        "classification": 0.0,
        "reply_quality": 0.0,
        "routing": 0.0,
    }

    if action.category is not None:
        if action.category == email.true_category:
            breakdown["classification"] = 0.30
        elif (email.true_category, action.category) in WILDLY_WRONG:
            breakdown["classification"] = -0.25
        else:
            breakdown["classification"] = -0.10

    if action.action_type == "reply":
        breakdown["reply_quality"] = round(sum(compute_reply_quality(action.reply_body, email).values()), 4)
        if email.true_category == "spam":
            breakdown["routing"] = -0.15
    elif action.action_type == "archive":
        if email.true_category == "urgent":
            breakdown["routing"] = -0.20
    elif action.action_type == "forward":
        if action.forward_to is not None and action.forward_to == email.correct_routing:
            breakdown["routing"] = 0.30
        elif email.correct_routing is not None:
            breakdown["routing"] = -0.10
    elif action.action_type == "escalate":
        if email.true_category == "urgent" and email.correct_routing in {"legal", "hr"}:
            breakdown["routing"] = 0.30
        elif email.true_category == "urgent" and email.correct_routing in {"billing", "support"}:
            breakdown["routing"] = -0.05

    reward_value = max(-1.0, min(1.0, sum(breakdown.values())))
    return EmailReward(value=reward_value, breakdown=breakdown)

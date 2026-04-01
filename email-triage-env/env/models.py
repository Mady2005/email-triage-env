from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


ALLOWED_CATEGORIES = {"urgent", "normal", "spam", "inquiry"}
ALLOWED_DEPARTMENTS = {"billing", "support", "legal", "hr"}


class BaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmailObservation(BaseSchema):
    email_id: str
    subject: str
    body: str
    sender: str
    sender_name: str
    timestamp: str
    thread_history: list[str]
    queue_remaining: int


class EmailAction(BaseSchema):
    action_type: Literal["classify", "reply", "forward", "archive", "escalate"]
    category: Optional[str] = None
    reply_body: Optional[str] = None
    forward_to: Optional[str] = None

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.strip().lower()
        if normalized not in ALLOWED_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(ALLOWED_CATEGORIES)}")
        return normalized

    @field_validator("forward_to")
    @classmethod
    def validate_forward_to(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.strip().lower()
        if normalized not in ALLOWED_DEPARTMENTS:
            raise ValueError(f"forward_to must be one of {sorted(ALLOWED_DEPARTMENTS)}")
        return normalized


class EmailReward(BaseSchema):
    value: float = Field(..., ge=-1.0, le=1.0)
    breakdown: dict[str, float]


class EmailRecord(BaseSchema):
    email_id: str
    subject: str
    body: str
    sender: str
    sender_name: str
    timestamp: str
    thread_history: list[str]
    true_category: Literal["urgent", "normal", "spam", "inquiry"]
    required_keywords: list[str]
    correct_routing: Optional[Literal["billing", "support", "legal", "hr"]] = None
    requires_reply: bool
    expected_primary_action: Literal["classify", "reply", "forward", "archive", "escalate"]
    escalation_required: bool = False
    risk_flags: list[str] = []

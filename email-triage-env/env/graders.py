from __future__ import annotations

import json

from .email_generator import EmailDataset
from .models import EmailAction, EmailObservation
from .reward import compute_reply_quality


CATEGORY_ALIASES = {
    "promotional": "spam",
    "promotion": "spam",
    "junk": "spam",
    "phishing": "spam",
    "marketing": "spam",
    "question": "inquiry",
    "query": "inquiry",
    "request": "inquiry",
    "routine": "normal",
    "general": "normal",
    "regular": "normal",
    "critical": "urgent",
    "high": "urgent",
    "high_priority": "urgent",
}

ACTION_ALIASES = {
    "respond": "reply",
    "answer": "reply",
    "send_reply": "reply",
    "label": "classify",
}

DEPARTMENT_ALIASES = {
    "customer support": "support",
    "tech support": "support",
    "finance": "billing",
    "accounting": "billing",
    "people": "hr",
    "people ops": "hr",
}


class BaseGrader:
    def __init__(self) -> None:
        self.dataset = EmailDataset()
        self._last_observation: EmailObservation | None = None

    def _extract_action(self, text: str) -> EmailAction:
        candidate = text.strip()
        if "```" in candidate:
            for block in candidate.split("```"):
                cleaned = block.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                if cleaned.startswith("{") and cleaned.endswith("}"):
                    candidate = cleaned
                    break
        else:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = candidate[start : end + 1]
        payload = json.loads(candidate)

        action_type = str(payload.get("action_type", "")).strip().lower()
        payload["action_type"] = ACTION_ALIASES.get(action_type, action_type or "classify")

        category = payload.get("category")
        if isinstance(category, str):
            normalized = category.strip().lower().replace(" ", "_")
            normalized = CATEGORY_ALIASES.get(normalized, normalized)
            if normalized not in {"urgent", "normal", "spam", "inquiry"}:
                normalized = "normal"
            payload["category"] = normalized

        forward_to = payload.get("forward_to")
        if isinstance(forward_to, str):
            normalized = forward_to.strip().lower()
            normalized = DEPARTMENT_ALIASES.get(normalized, normalized)
            if normalized not in {"billing", "support", "legal", "hr"}:
                normalized = None
            payload["forward_to"] = normalized

        return EmailAction.model_validate(payload)

    def _record_for_last_obs(self):
        if self._last_observation is None:
            raise RuntimeError("build_prompt must be called before parse_action")
        return self.dataset.get_by_id(self._last_observation.email_id)


class ClassificationGrader(BaseGrader):
    def __init__(self) -> None:
        super().__init__()
        self.correct_classifications = 0
        self.scored_items = 0

    def build_prompt(self, obs: EmailObservation) -> str:
        self._last_observation = obs
        return (
            "Classify the email and return JSON only in this format:\n"
            '{"action_type": "classify", "category": "urgent|normal|spam|inquiry"}\n\n'
            f"Email ID: {obs.email_id}\n"
            f"Sender: {obs.sender_name} <{obs.sender}>\n"
            f"Subject: {obs.subject}\n"
            f"Body: {obs.body}\n"
            f"Thread history: {obs.thread_history}\n"
            f"Queue remaining: {obs.queue_remaining}\n"
        )

    def parse_action(self, text: str) -> EmailAction:
        action = self._extract_action(text)
        if self.scored_items < 10:
            record = self._record_for_last_obs()
            self.correct_classifications += int(action.category == record.true_category)
            self.scored_items += 1
        return action

    def final_score(self) -> float:
        score = max(0.0, min(1.0, self.correct_classifications / 10))
        assert 0.0 <= score <= 1.0
        return score


class TriageReplyGrader(BaseGrader):
    def __init__(self) -> None:
        super().__init__()
        self.correct_classifications = 0
        self.reply_quality_total = 0.0
        self.scored_items = 0

    def build_prompt(self, obs: EmailObservation) -> str:
        self._last_observation = obs
        return (
            "Classify the email and draft a suitable reply. Return JSON only in this format:\n"
            '{"action_type": "reply", "category": "urgent|normal|spam|inquiry", "reply_body": "..."}\n\n'
            f"Email ID: {obs.email_id}\n"
            f"Sender: {obs.sender_name} <{obs.sender}>\n"
            f"Subject: {obs.subject}\n"
            f"Body: {obs.body}\n"
            f"Thread history: {obs.thread_history}\n"
        )

    def parse_action(self, text: str) -> EmailAction:
        action = self._extract_action(text)
        if self.scored_items < 10:
            record = self._record_for_last_obs()
            self.correct_classifications += int(action.category == record.true_category)
            self.reply_quality_total += sum(compute_reply_quality(action.reply_body, record).values()) / 0.40
            self.scored_items += 1
        return action

    def final_score(self) -> float:
        classification_accuracy = self.correct_classifications / 10
        avg_reply_quality = self.reply_quality_total / 10
        score = max(0.0, min(1.0, 0.5 * classification_accuracy + 0.5 * avg_reply_quality))
        assert 0.0 <= score <= 1.0
        return score


class QueueTriageGrader(BaseGrader):
    def __init__(self) -> None:
        super().__init__()
        self.correct_classifications = 0
        self.routing_correct = 0
        self.routing_opportunities = 0
        self.reply_quality_total = 0.0
        self.destructive_actions = 0
        self.steps = 0

    def build_prompt(self, obs: EmailObservation) -> str:
        self._last_observation = obs
        current_position = 21 - obs.queue_remaining
        return (
            "Process this inbox item. Available action types: classify, reply, forward, archive, escalate.\n"
            "Return JSON only with keys action_type, category, reply_body, forward_to.\n"
            "Operational policy: escalate only when the email is urgent and HR/legal sensitive; "
            "forward operational urgent issues to billing or support; archive suspicious spam; "
            "reply with concrete acknowledgement when the sender expects an answer.\n\n"
            f"Queue position: {current_position} of 20\n"
            f"Email ID: {obs.email_id}\n"
            f"Sender: {obs.sender_name} <{obs.sender}>\n"
            f"Subject: {obs.subject}\n"
            f"Body: {obs.body}\n"
            f"Thread history: {obs.thread_history}\n"
        )

    def parse_action(self, text: str) -> EmailAction:
        action = self._extract_action(text)
        record = self._record_for_last_obs()
        self.correct_classifications += int(action.category == record.true_category)

        expected_escalation = record.escalation_required
        if action.action_type in {"forward", "escalate"} or record.correct_routing is not None:
            self.routing_opportunities += 1
            route_ok = False
            if expected_escalation:
                route_ok = action.action_type == "escalate"
            elif record.correct_routing is not None:
                route_ok = action.action_type == "forward" and action.forward_to == record.correct_routing
            self.routing_correct += int(route_ok)

        self.reply_quality_total += sum(compute_reply_quality(action.reply_body, record).values()) / 0.40

        destructive = (
            action.action_type == "archive" and record.true_category == "urgent"
        ) or (
            action.action_type == "reply" and record.true_category == "spam"
        ) or (
            action.action_type == "forward" and record.escalation_required
        ) or (
            action.action_type == "escalate" and not record.escalation_required and record.expected_primary_action != "escalate"
        )
        self.destructive_actions += int(destructive)
        self.steps += 1
        return action

    def final_score(self) -> float:
        denominator = max(1, self.steps)
        classification_acc = self.correct_classifications / denominator
        routing_acc = self.routing_correct / max(1, self.routing_opportunities)
        avg_reply_quality = self.reply_quality_total / denominator
        safety_score = 1.0 if self.destructive_actions == 0 else max(0.0, 1.0 - (self.destructive_actions / denominator))
        score = (
            0.30 * classification_acc
            + 0.30 * routing_acc
            + 0.30 * avg_reply_quality
            + 0.10 * safety_score
        )
        score = max(0.0, min(1.0, score))
        assert 0.0 <= score <= 1.0
        return score

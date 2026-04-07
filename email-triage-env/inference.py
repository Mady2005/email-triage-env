from __future__ import annotations

import json
import os
import sys
import time

from openai import OpenAI

from env.environment import EmailTriageEnv
from env.models import EmailAction, EmailObservation
from tasks.task_easy import ClassificationGrader
from tasks.task_hard import QueueTriageGrader
from tasks.task_medium import TriageReplyGrader


def _resolve_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or HF_TOKEN before running inference.py")
    return api_key


MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")


def _build_client() -> OpenAI:
    return OpenAI(api_key=_resolve_api_key(), base_url=os.getenv("API_BASE_URL"))


def _classify_observation(obs: EmailObservation) -> str:
    text = " ".join([obs.subject, obs.body, *obs.thread_history]).lower()
    spam_markers = ["unsubscribe", "gift card", "crypto", "newsletter", "traffic", "backlinks", "click", "wallet"]
    urgent_markers = ["outage", "urgent", "subpoena", "legal", "deadline", "failing", "cutoff", "incident", "blocked"]
    inquiry_markers = ["question", "clarify", "how", "can you", "details", "support", "policy", "whether"]

    if any(marker in text for marker in spam_markers):
        return "spam"
    if any(marker in text for marker in urgent_markers):
        return "urgent"
    if any(marker in text for marker in inquiry_markers):
        return "inquiry"
    return "normal"


def _reply_body(obs: EmailObservation, category: str) -> str:
    text = f"{obs.subject} {obs.body}".lower()
    if category == "urgent":
        return (
            f"Hi {obs.sender_name}, thanks for flagging this. We understand the urgency and are reviewing the "
            "issue now. We will coordinate the right team, share the next update shortly, and keep you posted "
            "until there is a clear resolution or workaround."
        )
    if "billing" in text or "invoice" in text or "payment" in text:
        return (
            f"Hi {obs.sender_name}, thanks for your note. We are reviewing the billing details and will confirm "
            "the invoice, payment status, and next steps shortly so you have a clear answer."
        )
    return (
        f"Hi {obs.sender_name}, thanks for reaching out. We reviewed your message and will share the relevant "
        "details, policy context, and next steps shortly so you have a clear answer."
    )


def _fallback_action(obs: EmailObservation, grader) -> EmailAction:
    category = _classify_observation(obs)
    text = f"{obs.subject} {obs.body}".lower()

    if isinstance(grader, ClassificationGrader):
        return EmailAction(action_type="classify", category=category)

    if isinstance(grader, TriageReplyGrader):
        return EmailAction(
            action_type="reply",
            category=category,
            reply_body=_reply_body(obs, category),
        )

    if category == "spam":
        return EmailAction(action_type="archive", category="spam")

    if category == "urgent":
        if any(marker in text for marker in ["legal", "subpoena", "termination", "employee", "leave", "hr"]):
            return EmailAction(action_type="escalate", category="urgent")
        forward_to = "billing" if any(marker in text for marker in ["invoice", "payment", "refund", "charge"]) else "support"
        return EmailAction(action_type="forward", category="urgent", forward_to=forward_to)

    if category == "inquiry":
        return EmailAction(
            action_type="reply",
            category="inquiry",
            reply_body=_reply_body(obs, category),
        )

    return EmailAction(action_type="classify", category="normal")


def _action_via_grader(grader, obs: EmailObservation, text: str | None) -> EmailAction:
    if text:
        try:
            return grader.parse_action(text)
        except Exception as exc:
            print(f"[warn] parse fallback for {obs.email_id}: {exc}", file=sys.stderr)

    fallback_action = _fallback_action(obs, grader)
    return grader.parse_action(json.dumps(fallback_action.model_dump(exclude_none=True)))


def _query_model(client: OpenAI | None, prompt: str, obs: EmailObservation) -> str | None:
    if client is None:
        return None

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt)

    print(f"[warn] model fallback for {obs.email_id}: {last_error}", file=sys.stderr)
    return None


def run_task(grader_class) -> float:
    try:
        client = _build_client()
    except Exception as exc:
        print(f"[warn] client init fallback: {exc}", file=sys.stderr)
        client = None

    env = EmailTriageEnv()
    grader = grader_class()
    obs = env.reset()
    while True:
        prompt = grader.build_prompt(obs)
        model_text = _query_model(client, prompt, obs)
        action = _action_via_grader(grader, obs, model_text)
        obs, reward, done, _ = env.step(action)
        if done:
            break
    return grader.final_score()


if __name__ == "__main__":
    for name, cls in [
        ("classification", ClassificationGrader),
        ("triage_reply", TriageReplyGrader),
        ("queue_triage", QueueTriageGrader),
    ]:
        score = run_task(cls)
        print(f"{name}: {score:.3f}")

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

from openai import OpenAI

from env.environment import EmailTriageEnv
from env.models import EmailAction, EmailObservation
from tasks.task_easy import ClassificationGrader
from tasks.task_hard import QueueTriageGrader
from tasks.task_medium import TriageReplyGrader


BENCHMARK = "email-triage-env"
MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_BASE_URL = os.getenv("API_BASE_URL")


def _resolve_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or HF_TOKEN before running inference.py")
    return api_key


def _build_client() -> OpenAI:
    return OpenAI(api_key=_resolve_api_key(), base_url=API_BASE_URL)


def _classify_observation(obs: EmailObservation) -> str:
    text = " ".join([obs.subject, obs.body, *obs.thread_history]).lower()
    spam_markers = ["unsubscribe", "gift card", "crypto", "newsletter", "traffic", "backlinks", "click", "wallet"]
    urgent_markers = ["outage", "urgent", "subpoena", "legal", "deadline", "failing", "cutoff", "incident", "blocked"]
    inquiry_markers = ["question", "clarify", "how", "can you", "details", "policy", "whether", "support"]

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
    if "billing" in text or "invoice" in text or "payment" in text or "refund" in text:
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
        if any(marker in text for marker in ["legal", "subpoena", "termination", "employee", "leave", "hr", "policy"]):
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


def _query_model(client: Optional[OpenAI], prompt: str) -> tuple[Optional[str], Optional[str]]:
    if client is None:
        return None, "client_unavailable"

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return resp.choices[0].message.content or "", None
        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt)

    return None, str(last_error) if last_error else "model_request_failed"


def _action_from_text(grader, obs: EmailObservation, text: Optional[str]) -> tuple[EmailAction, Optional[str]]:
    if text:
        try:
            return grader.parse_action(text), None
        except Exception as exc:
            fallback_action = _fallback_action(obs, grader)
            fallback_text = json.dumps(fallback_action.model_dump(exclude_none=True))
            return grader.parse_action(fallback_text), str(exc)

    fallback_action = _fallback_action(obs, grader)
    fallback_text = json.dumps(fallback_action.model_dump(exclude_none=True))
    return grader.parse_action(fallback_text), "fallback_action"


def _action_string(action: EmailAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))


def _log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL}", flush=True)


def _log_step(step: int, action: EmailAction, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={_action_string(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def run_task(task_name: str, grader_class) -> float:
    try:
        client = _build_client()
    except Exception as exc:
        print(f"[warn] client init fallback: {exc}", file=sys.stderr)
        client = None

    env = EmailTriageEnv()
    grader = grader_class()
    obs = env.reset()
    rewards: list[float] = []
    step_count = 0
    success = False
    score = 0.0

    _log_start(task_name)

    try:
        while True:
            step_count += 1
            prompt = grader.build_prompt(obs)
            model_text, model_error = _query_model(client, prompt)
            action, parse_error = _action_from_text(grader, obs, model_text)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward.value)

            error_message = parse_error or model_error
            _log_step(step_count, action, reward.value, done, error_message)

            if done:
                break

        score = grader.final_score()
        success = True
        return score
    except Exception as exc:
        success = False
        score = 0.0
        print(f"[warn] task fallback failure for {task_name}: {exc}", file=sys.stderr)
        return score
    finally:
        _log_end(success, step_count, score, rewards)


if __name__ == "__main__":
    for name, cls in [
        ("classification", ClassificationGrader),
        ("triage_reply", TriageReplyGrader),
        ("queue_triage", QueueTriageGrader),
    ]:
        run_task(name, cls)

from __future__ import annotations

import os

from openai import OpenAI

from env.environment import EmailTriageEnv
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


def run_task(grader_class) -> float:
    client = _build_client()
    env = EmailTriageEnv()
    grader = grader_class()
    obs = env.reset()
    while True:
        prompt = grader.build_prompt(obs)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        action = grader.parse_action(resp.choices[0].message.content or "")
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

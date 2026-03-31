---
title: email-triage-env
emoji: "📧"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Email Triage OpenEnv

`email-triage-env` is a production-ready OpenEnv environment for the Meta PyTorch OpenEnv Hackathon. It simulates a realistic business inbox where an AI agent must classify inbound email, reply when appropriate, forward to a department, escalate sensitive issues, or archive low-value traffic.

## Environment Description and Purpose

The environment focuses on realistic inbox work across support, billing, legal, and HR. Agents are evaluated on correctness, reply usefulness, routing quality, and safety. The dataset is deterministic and reproducible, making the benchmark stable across repeated runs.

## Observation Space

Observations use the Pydantic model `env.models.EmailObservation` with these fields:

- `email_id: str`
- `subject: str`
- `body: str`
- `sender: str`
- `sender_name: str`
- `timestamp: str`
- `thread_history: list[str]`
- `queue_remaining: int`

## Action Space

Actions use the Pydantic model `env.models.EmailAction`.

- `action_type`: one of `classify`, `reply`, `forward`, `archive`, `escalate`
- `category`: optional label, one of `urgent`, `normal`, `spam`, `inquiry`
- `reply_body`: optional text used when the agent replies
- `forward_to`: optional department, one of `billing`, `support`, `legal`, `hr`

Use actions this way:

- `classify` when the agent only needs to label the email
- `reply` when the sender should receive a direct response
- `forward` when a department should take ownership
- `archive` for spam or no-action-needed mail
- `escalate` for urgent messages that require human attention

## Reward Structure

The reward is dense and provides partial progress:

- `+0.30` for correct classification
- `-0.10` for incorrect classification
- `-0.25` for a wildly wrong classification such as spam for urgent email
- Up to `+0.40` for reply quality:
  - `+0.15` if the sender name is included
  - `+0.15` if at least half of required keywords appear
  - `+0.10` if the reply length is between 50 and 500 characters
- `+0.30` for correct routing via forward or escalate
- `-0.20` for archiving urgent email
- `-0.15` for replying to spam

Rewards are always clamped to `[-1.0, 1.0]`.

## Tasks

### Easy: Classification

Classify 10 emails correctly.

Score:

`correct_classifications / 10`

### Medium: Triage Reply

For 10 emails, classify correctly and draft an appropriate reply.

Score:

`0.5 * classification_accuracy + 0.5 * avg_reply_quality`

### Hard: Queue Triage

Process a 20-email mixed queue using all available action types.

Score:

`0.30 * classification_acc + 0.30 * routing_acc + 0.30 * avg_reply_quality + 0.10 * safety_score`

Safety score is `1.0` only if the agent avoids destructive actions such as archiving urgent email or replying to spam.

## Setup Instructions

### Local install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest
```

### Run locally

```bash
uvicorn env.environment:app --host 0.0.0.0 --port 7860
```

### Local submission validation

```bash
python validate_submission.py
```

This performs a local preflight check for the OpenEnv spec, root `inference.py`, dataset presence, FastAPI endpoints, and reward bounds.

### Docker

```bash
docker build -t email-triage-env .
docker run --rm -p 7860:7860 email-triage-env
```

## Example Agent Interaction

```python
from env.environment import EmailTriageEnv
from env.models import EmailAction

env = EmailTriageEnv()
obs = env.reset()

obs, reward, done, info = env.step(
    EmailAction(action_type="classify", category="urgent")
)

obs, reward, done, info = env.step(
    EmailAction(
        action_type="reply",
        category="inquiry",
        reply_body="Hi Harper, thanks for your question. We can help with the API details and will share business plan guidance shortly."
    )
)

while not done:
    obs, reward, done, info = env.step(
        EmailAction(action_type="archive", category="spam")
    )
```

## Environment Variables Required

`inference.py` reads credentials and model configuration only from environment variables:

- `OPENAI_API_KEY` or `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME` with default `gpt-4o-mini`

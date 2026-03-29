from __future__ import annotations

from collections import Counter

from fastapi.testclient import TestClient

from env.email_generator import EmailDataset
from env.environment import EmailTriageEnv, app
from env.models import EmailAction, EmailObservation
from tasks.task_easy import ClassificationGrader
from tasks.task_hard import QueueTriageGrader
from tasks.task_medium import TriageReplyGrader


def _action_for(action_type: str) -> EmailAction:
    if action_type == "classify":
        return EmailAction(action_type="classify", category="normal")
    if action_type == "reply":
        return EmailAction(
            action_type="reply",
            category="inquiry",
            reply_body="Hi Alex, thanks for your note. We are reviewing the request and will share a billing update with the team today.",
        )
    if action_type == "forward":
        return EmailAction(action_type="forward", category="urgent", forward_to="support")
    if action_type == "archive":
        return EmailAction(action_type="archive", category="spam")
    if action_type == "escalate":
        return EmailAction(action_type="escalate", category="urgent")
    raise AssertionError(action_type)


def test_reset_returns_valid_observation() -> None:
    env = EmailTriageEnv()
    obs = env.reset()
    assert isinstance(obs, EmailObservation)
    assert obs.queue_remaining == 20


def test_step_with_every_action_type_returns_correct_types() -> None:
    for action_type in ["classify", "reply", "forward", "archive", "escalate"]:
        env = EmailTriageEnv()
        env.reset()
        obs, reward, done, info = env.step(_action_for(action_type))
        assert obs is None or isinstance(obs, EmailObservation)
        assert isinstance(reward.value, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


def test_reward_is_always_in_range() -> None:
    env = EmailTriageEnv()
    env.reset()
    for _ in range(20):
        _, reward, done, _ = env.step(
            EmailAction(
                action_type="reply",
                category="urgent",
                reply_body="Hi Team, urgent billing incident acknowledged. We are investigating the outage and will update you shortly.",
            )
        )
        assert -1.0 <= reward.value <= 1.0
        if done:
            break


def test_done_after_twenty_steps() -> None:
    env = EmailTriageEnv()
    env.reset()
    done = False
    for _ in range(20):
        _, _, done, _ = env.step(EmailAction(action_type="archive", category="spam"))
    assert done is True


def test_state_tracks_current_index() -> None:
    env = EmailTriageEnv()
    env.reset()
    for expected_idx in range(1, 4):
        env.step(EmailAction(action_type="classify", category="normal"))
        assert env.state()["current_idx"] == expected_idx


def test_all_graders_return_scores_in_range() -> None:
    env = EmailTriageEnv()
    obs = env.reset()

    easy = ClassificationGrader()
    for _ in range(10):
        easy.build_prompt(obs)
        easy.parse_action('{"action_type":"classify","category":"normal"}')
    assert 0.0 <= easy.final_score() <= 1.0

    medium = TriageReplyGrader()
    for _ in range(10):
        medium.build_prompt(obs)
        medium.parse_action(
            '{"action_type":"reply","category":"inquiry","reply_body":"Hi Harper, thanks for your question. We can help with billing details and will share the business plan options shortly."}'
        )
    assert 0.0 <= medium.final_score() <= 1.0

    hard = QueueTriageGrader()
    for _ in range(20):
        hard.build_prompt(obs)
        hard.parse_action('{"action_type":"archive","category":"spam"}')
    assert 0.0 <= hard.final_score() <= 1.0


def test_email_dataset_loads_expected_distribution() -> None:
    dataset = EmailDataset()
    assert len(dataset) == 50
    counts = Counter(record.true_category for record in dataset.records)
    assert counts == {"urgent": 12, "normal": 15, "spam": 12, "inquiry": 11}


def test_fastapi_health_endpoint_returns_http_200() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_fastapi_reset_and_step_endpoints_return_valid_payloads() -> None:
    client = TestClient(app)
    reset_response = client.post("/reset")
    assert reset_response.status_code == 200
    payload = reset_response.json()
    assert payload["email_id"].startswith("email-")
    assert payload["queue_remaining"] == 20

    step_response = client.post(
        "/step",
        json={"action_type": "classify", "category": "normal"},
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert isinstance(step_payload["done"], bool)
    assert -1.0 <= step_payload["reward"]["value"] <= 1.0
    assert "email_id" in step_payload["info"]

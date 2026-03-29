from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from .email_generator import EmailDataset
from .models import EmailAction, EmailObservation, EmailReward
from .reward import compute_reward


class StepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obs: EmailObservation | None
    reward: EmailReward
    done: bool
    info: dict


class EmailTriageEnv:
    def __init__(self, queue_size: int = 20, seed: int = 42):
        self.dataset = EmailDataset()
        self.queue_size = queue_size
        self.seed = seed
        self.queue = []
        self.idx = 0
        self.done = False
        self.scores: list[float] = []

    def _current_record(self):
        if self.done or not self.queue:
            raise RuntimeError("Environment is done or has not been reset")
        return self.queue[self.idx]

    def _observation_for_index(self, index: int) -> EmailObservation:
        record = self.queue[index]
        return self.dataset.to_observation(record, queue_remaining=len(self.queue) - index)

    def reset(self) -> EmailObservation:
        self.queue = self.dataset.sample_queue(size=self.queue_size, seed=self.seed)
        self.idx = 0
        self.done = False
        self.scores = []
        return self._observation_for_index(self.idx)

    def step(self, action: EmailAction) -> tuple[EmailObservation | None, EmailReward, bool, dict]:
        if self.done:
            raise RuntimeError("Environment already completed")

        current_email = self._current_record()
        reward = compute_reward(action, current_email)
        self.scores.append(reward.value)

        self.idx += 1
        if self.idx >= len(self.queue):
            self.done = True
            obs = None
        else:
            obs = self._observation_for_index(self.idx)

        return obs, reward, self.done, {"email_id": current_email.email_id, "step": self.idx}

    def state(self) -> dict:
        return {
            "queue_size": len(self.queue),
            "current_idx": self.idx,
            "done": self.done,
            "total_reward_so_far": round(sum(self.scores), 4),
        }


app = FastAPI(title="Email Triage OpenEnv")
_APP_ENV = EmailTriageEnv()


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=EmailObservation)
def reset_env() -> EmailObservation:
    return _APP_ENV.reset()


@app.post("/step", response_model=StepResponse)
def step_env(action: EmailAction) -> StepResponse:
    obs, reward, done, info = _APP_ENV.step(action)
    return StepResponse(obs=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state_env() -> dict:
    return _APP_ENV.state()


@app.get("/tasks")
def list_tasks() -> list[dict[str, str]]:
    return [
        {"id": "classification", "difficulty": "easy", "grader": "tasks.task_easy.ClassificationGrader"},
        {"id": "triage_reply", "difficulty": "medium", "grader": "tasks.task_medium.TriageReplyGrader"},
        {"id": "queue_triage", "difficulty": "hard", "grader": "tasks.task_hard.QueueTriageGrader"},
    ]

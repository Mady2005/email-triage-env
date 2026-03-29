from __future__ import annotations

import importlib
import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parent


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def main() -> int:
    openenv_path = ROOT / "openenv.yaml"
    dockerfile_path = ROOT / "Dockerfile"
    inference_path = ROOT / "inference.py"
    dataset_path = ROOT / "data" / "emails.json"

    _check(openenv_path.exists(), "openenv.yaml exists at project root")
    _check(dockerfile_path.exists(), "Dockerfile exists at project root")
    _check(inference_path.exists(), "inference.py exists at project root")
    _check(dataset_path.exists(), "data/emails.json exists")

    spec = yaml.safe_load(openenv_path.read_text(encoding="utf-8"))
    _check(spec["name"] == "email-triage-env", "OpenEnv spec name matches project")
    _check(spec["reward_range"] == [-1.0, 1.0], "reward range is [-1.0, 1.0]")
    _check(len(spec["tasks"]) == 3, "three tasks are declared in openenv.yaml")

    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    _check('EXPOSE 7860' in dockerfile, "Dockerfile exposes port 7860")
    _check('env.environment:app' in dockerfile, "Dockerfile starts uvicorn with FastAPI app")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    _check(len(dataset) == 50, "dataset contains 50 emails")

    inference = importlib.import_module("inference")
    _check(hasattr(inference, "run_task"), "inference.py exposes run_task")

    environment = importlib.import_module("env.environment")
    client = TestClient(environment.app)

    health = client.get("/")
    _check(health.status_code == 200 and health.json() == {"status": "ok"}, "GET / returns HTTP 200 with health payload")

    reset = client.post("/reset")
    _check(reset.status_code == 200, "POST /reset returns HTTP 200")
    reset_json = reset.json()
    _check(reset_json["queue_remaining"] == 20, "POST /reset returns first observation")

    step = client.post("/step", json={"action_type": "classify", "category": "normal"})
    _check(step.status_code == 200, "POST /step returns HTTP 200")
    step_json = step.json()
    _check(-1.0 <= step_json["reward"]["value"] <= 1.0, "step reward is clamped to [-1.0, 1.0]")

    state = client.get("/state")
    _check(state.status_code == 200, "GET /state returns HTTP 200")

    tasks = client.get("/tasks")
    _check(tasks.status_code == 200 and len(tasks.json()) == 3, "GET /tasks returns three task definitions")

    print("validation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

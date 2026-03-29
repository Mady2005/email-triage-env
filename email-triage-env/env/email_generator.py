from __future__ import annotations

import json
import random
from pathlib import Path

from .models import EmailRecord, EmailObservation


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "emails.json"
SEED = 42


def _keywordize(*parts: str) -> list[str]:
    return [part.lower() for part in parts]


def build_seeded_emails() -> list[dict]:
    random.seed(SEED)

    category_specs = {
        "urgent": {
            "count": 12,
            "subjects": [
                "Production outage in region",
                "Payment processor failures",
                "Immediate legal notice",
                "Payroll cutoff issue",
                "Data export deadline problem",
                "Termination dispute response",
                "SSO login outage",
                "Duplicate annual charge",
                "Subpoena for audit records",
                "Pending wire transfer",
                "Employee file exposure",
                "Contract signature deadline",
            ],
            "senders": [
                ("Nina", "ops"),
                ("Marcus", "finance"),
                ("Jordan", "legal"),
                ("Olivia", "hr"),
                ("Sameer", "compliance"),
                ("Rebecca", "legal"),
                ("Leo", "it"),
                ("Claire", "accounts"),
                ("David", "legal"),
                ("Helen", "ap"),
                ("Amelia", "peopleops"),
                ("Mohan", "procurement"),
            ],
            "bodies": [
                "We have a live production issue that is blocking critical work and needs immediate action from your team today.",
                "Several users are unable to complete a required workflow and the failure is causing business disruption right now.",
                "This message requires confirmation and a same-day response because the operational risk is escalating quickly.",
            ],
            "keywords": [
                _keywordize("urgent", "update", "investigating"),
                _keywordize("billing", "payment", "urgent"),
                _keywordize("legal", "response", "today"),
                _keywordize("incident", "outage", "workaround"),
            ],
            "routing_cycle": ["support", "billing", "legal", "hr"],
            "requires_reply": True,
            "thread_history": [
                ["Automated alert triggered 20 minutes ago."],
                ["Customer followed up asking for an update.", "Internal team confirmed reproducibility."],
                [],
            ],
        },
        "normal": {
            "count": 15,
            "subjects": [
                "Weekly project update",
                "Reschedule roadmap meeting",
                "Monthly usage report",
                "Partnership memo review",
                "Onboarding completed",
                "Steering committee agenda",
                "Training session reminder",
                "Design feedback request",
                "Quarterly review notes",
                "Implementation workshop timing",
                "Documentation refresh complete",
                "Pilot adoption status",
                "Confirm attendee list",
                "Implementation retrospective",
                "Renewal prep timeline",
            ],
            "senders": [
                ("Priya", "pm"),
                ("Aaron", "strategy"),
                ("Lucy", "analytics"),
                ("Gabriel", "bizdev"),
                ("Mei", "success"),
                ("Ben", "operations"),
                ("Sophia", "enablement"),
                ("Noah", "design"),
                ("Elena", "accounts"),
                ("Owen", "cs"),
                ("Isla", "docs"),
                ("Ethan", "adoption"),
                ("Zoe", "events"),
                ("Adam", "delivery"),
                ("Mia", "procurement"),
            ],
            "bodies": [
                "This is a routine operational update with no immediate urgency, but a short acknowledgement would be helpful.",
                "We are sharing normal project coordination details and would appreciate confirmation when convenient.",
                "No urgent action is required. This is mainly for planning, scheduling, or status visibility.",
            ],
            "keywords": [
                _keywordize("thanks", "update", "schedule"),
                _keywordize("meeting", "confirm", "timeline"),
                _keywordize("status", "review", "notes"),
            ],
            "routing_cycle": [None],
            "requires_reply": False,
            "thread_history": [
                [],
                ["Previous update shared last week."],
                ["Meeting notes attached in the prior message."],
            ],
        },
        "spam": {
            "count": 12,
            "subjects": [
                "Boost traffic fast",
                "Mailbox quota full",
                "Crypto newsletter invite",
                "Claim your gift card",
                "Unexpected newsletter signup",
                "Cheap backlinks available",
                "Reset payroll account",
                "Warm B2B leads",
                "Urgent invoice attachment",
                "Free AI market report",
                "Remote assistant offer",
                "Domain security renewal",
            ],
            "senders": [
                ("Promo", "growth"),
                ("Security", "alerts"),
                ("VIP", "profits"),
                ("Rewards", "voucher"),
                ("Hello", "newsletter"),
                ("Sales", "rank"),
                ("Alerts", "payroll"),
                ("Contact", "leads"),
                ("Billing", "invoice"),
                ("Reports", "market"),
                ("Careers", "hiring"),
                ("Notice", "domain"),
            ],
            "bodies": [
                "This promotional or suspicious message should not receive a direct reply and usually belongs in the archive.",
                "The sender is attempting to drive clicks, collect credentials, or force engagement through marketing language.",
                "Treat this as low-value or potentially malicious email rather than a legitimate request.",
            ],
            "keywords": [
                _keywordize("spam", "archive"),
                _keywordize("phishing", "do not click"),
                _keywordize("fraud", "ignore"),
            ],
            "routing_cycle": [None],
            "requires_reply": False,
            "thread_history": [[], []],
        },
        "inquiry": {
            "count": 11,
            "subjects": [
                "API rate limits question",
                "Annual billing for seats",
                "Parental leave policy",
                "Need DPA details",
                "SAML and SCIM support",
                "Invoice recipients question",
                "Vacation workflow question",
                "Export formats support",
                "Auto-renewal terms",
                "Failed reimbursements",
                "Policy acknowledgement scope",
            ],
            "senders": [
                ("Harper", "dev"),
                ("Carter", "procurement"),
                ("Hannah", "people"),
                ("Sara", "procurement"),
                ("Trent", "it"),
                ("Lila", "finance"),
                ("Miguel", "hr"),
                ("Chloe", "data"),
                ("Kevin", "legalops"),
                ("Rachel", "finance"),
                ("Jasmine", "people"),
            ],
            "bodies": [
                "The sender is evaluating a feature or policy and needs a clear informational response rather than urgent incident handling.",
                "This is a legitimate product or policy question that should be answered accurately and routed to the right team when needed.",
                "The request asks for clarification on billing, support, legal, or HR behavior.",
            ],
            "keywords": [
                _keywordize("question", "details", "help"),
                _keywordize("billing", "policy", "support"),
                _keywordize("contract", "workflow", "feature"),
            ],
            "routing_cycle": ["support", "billing", "hr", "legal"],
            "requires_reply": True,
            "thread_history": [[], ["Security review already completed."], ["Comparing vendors this quarter."]],
        },
    }

    emails: list[dict] = []
    base_day = 1
    for category, spec in category_specs.items():
        for index in range(spec["count"]):
            sender_name, team = spec["senders"][index]
            subject = spec["subjects"][index]
            body = f"{subject}. {spec['bodies'][index % len(spec['bodies'])]}"
            keywords = spec["keywords"][index % len(spec["keywords"])]
            routing = spec["routing_cycle"][index % len(spec["routing_cycle"])]
            thread_history = spec["thread_history"][index % len(spec["thread_history"])]
            emails.append(
                {
                    "email_id": f"email-{len(emails) + 1:03d}",
                    "subject": subject,
                    "body": body,
                    "sender": f"{sender_name.lower()}.{team}@example.com",
                    "sender_name": sender_name,
                    "timestamp": f"2026-03-{base_day + (index % 8):02d}T{8 + (index % 10):02d}:{10 + (index % 40):02d}:00Z",
                    "thread_history": thread_history,
                    "true_category": category,
                    "required_keywords": keywords,
                    "correct_routing": routing,
                    "requires_reply": bool(spec["requires_reply"] or category == "inquiry"),
                }
            )

    random.shuffle(emails)
    return emails


def ensure_dataset(path: Path = DATA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(build_seeded_emails(), handle, indent=2)
    return path


class EmailDataset:
    def __init__(self, path: Path | None = None):
        self.path = path or DATA_PATH
        if not self.path.exists():
            ensure_dataset(self.path)
        with self.path.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)
        self.records = [EmailRecord.model_validate(item) for item in raw_data]

    def __len__(self) -> int:
        return len(self.records)

    def get_record(self, index: int) -> EmailRecord:
        return self.records[index]

    def get_by_id(self, email_id: str) -> EmailRecord:
        for record in self.records:
            if record.email_id == email_id:
                return record
        raise KeyError(email_id)

    def sample_queue(self, size: int = 20, seed: int = SEED) -> list[EmailRecord]:
        rng = random.Random(seed)
        indices = list(range(len(self.records)))
        rng.shuffle(indices)
        return [self.records[i] for i in indices[:size]]

    @staticmethod
    def to_observation(record: EmailRecord, queue_remaining: int) -> EmailObservation:
        return EmailObservation(
            email_id=record.email_id,
            subject=record.subject,
            body=record.body,
            sender=record.sender,
            sender_name=record.sender_name,
            timestamp=record.timestamp,
            thread_history=record.thread_history,
            queue_remaining=queue_remaining,
        )

import json
import os
from datetime import datetime
from crewai.tools import tool


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PENDING_DIR = os.path.join(
    BASE_DIR,
    "pending"
)

ESCALATION_FILE = os.path.join(
    PENDING_DIR,
    "escalations.json"
)


def load_escalations():

    os.makedirs(PENDING_DIR, exist_ok=True)

    if not os.path.exists(ESCALATION_FILE):
        return []

    try:
        with open(
            ESCALATION_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    except Exception:
        return []


def save_escalations(escalations):

    os.makedirs(PENDING_DIR, exist_ok=True)

    with open(
        ESCALATION_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            escalations,
            f,
            indent=4,
            ensure_ascii=False
        )


@tool("escalate_to_human")
def escalate_to_human(
    issue_summary: str,
    conversation_summary: str
) -> str:
    """
    Escalate a customer support issue to a human.

    Use this tool when:
    1. The customer explicitly asks for a human.
    2. The agent cannot confidently solve the issue.
    3. The company knowledge does not provide enough
       information to resolve the issue.

    Creates a pending support request and returns
    the generated ticket ID.
    """

    escalations = load_escalations()

    ticket_number = len(escalations) + 1

    ticket_id = (
        f"TKT-2026-{ticket_number:04d}"
    )

    ticket = {
        "ticket_id": ticket_id,
        "status": "Pending",
        "issue_summary": issue_summary,
        "conversation_summary": conversation_summary,
        "created_at": datetime.now().isoformat()
    }

    escalations.append(ticket)

    save_escalations(escalations)

    return f"""
Support request created successfully.

Ticket ID: {ticket_id}
Status: Pending

The customer's issue has been escalated to human support.
"""

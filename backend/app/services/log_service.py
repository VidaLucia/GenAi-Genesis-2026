import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4

LOG_PATH = Path("data/analyst_logs.jsonl")


def save_analyst_log(transaction: dict, features: dict, model_outputs: dict, decision: dict) -> str:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_id = f"log_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"

    payload = {
        "log_id": log_id,
        "created_at": datetime.utcnow().isoformat(),
        "transaction": transaction,
        "features": features,
        "model_outputs": model_outputs,
        "decision": decision,
        "review_status": "pending"
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

    return log_id


def list_analyst_logs() -> list[dict]:
    if not LOG_PATH.exists():
        return []

    logs = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs


def list_user_transactions(user_id: str) -> list[dict]:
    logs = list_analyst_logs()
    return [log for log in logs if str(log.get("transaction", {}).get("user_id")) == user_id]


def review_log_status(log_id: str, review_status: str) -> bool:
    logs = list_analyst_logs()
    found = False

    for log in logs:
        if log["log_id"] == log_id:
            log["review_status"] = review_status
            found = True
            break

    if not found:
        return False

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")

    return True
import json
from pathlib import Path

DATA_PATH = Path("data/user_profiles.json")


def load_user_profile(user_id: str):
    if not DATA_PATH.exists():
        return None

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    for profile in profiles:
        if profile["user_id"] == user_id:
            return profile

    return None
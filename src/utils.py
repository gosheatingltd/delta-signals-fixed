import yaml
from datetime import datetime, timezone

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def utcnow():
    return datetime.now(timezone.utc)

"""Append-only JSONL cache of API responses keyed by a hash of (model, input, questions)."""
import hashlib
import json
import threading
from pathlib import Path

from .env import ROOT

RAW_DIR = ROOT / "results" / "raw"


def request_key(model: str, input_, questions: dict, tag: str = "") -> str:
    payload = json.dumps({"model": model, "input": input_, "questions": questions, "tag": tag},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, experiment: str):
        self.path = RAW_DIR / f"{experiment}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.records = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    # Only successful or blocked results count as cached; transient errors get retried.
                    if r.get("status") in ("ok", "blocked"):
                        self.records[r["key"]] = r

    def get(self, key: str):
        return self.records.get(key)

    def put(self, record: dict) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record.get("status") in ("ok", "blocked"):
                self.records[record["key"]] = record


def load_records(experiment: str) -> list:
    """All ok/blocked records for an experiment, latest per key."""
    return list(Cache(experiment).records.values())

"""Structured data recorder. Saves JSON/JSONL to logs/<run_dir>/records/."""

import json
from pathlib import Path
from typing import Any


class Recorder:
    def __init__(self, records_dir: str):
        self._dir = Path(records_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, name: str, data: Any) -> str:
        path = self._dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return str(path)

    def append_jsonl(self, name: str, record: dict) -> str:
        path = self._dir / f"{name}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return str(path)

    def save_text(self, name: str, text: str) -> str:
        path = self._dir / f"{name}.txt"
        path.write_text(text)
        return str(path)

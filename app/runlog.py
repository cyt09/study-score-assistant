"""运行日志：每次会话一个 run_id，逐步写入 JSONL，便于事后核对与脱敏导出。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config import settings

_SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{8,}"), "sk-***REDACTED***"),
    (re.compile(r"(?i)(api[_-]?key\"?\s*[:=]\s*\"?)[^\s\",]+"), r"\1***REDACTED***"),
]


def sanitize(text: str, limit: int = 1500) -> str:
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    if len(text) > limit:
        return text[:limit] + f"…(截断，共 {len(text)} 字符)"
    return text


class RunLogger:
    def __init__(self, run_id: str, filename: str = "run.jsonl") -> None:
        self.run_id = run_id
        self.path: Path = settings.log_dir / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: str, event: str, payload: dict[str, Any] | None = None) -> None:
        entry = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "step": step,
            "event": event,
            "payload": self._sanitize_payload(payload or {}),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _sanitize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, str):
                cleaned[key] = sanitize(value)
            elif isinstance(value, (int, float, bool)) or value is None:
                cleaned[key] = value
            else:
                cleaned[key] = sanitize(json.dumps(value, ensure_ascii=False, default=str))
        return cleaned

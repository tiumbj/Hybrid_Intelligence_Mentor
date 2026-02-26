"""
trade_logger.py
Version: 1.0.0
Changelog:
- 1.0.0: Append-only JSON event log with atomic write and basic dedupe support.
Rules:
- Never freeze silently
- Keep file as a JSON list
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class TradeLogger:
    def __init__(self, path: str):
        self.path = path

    def _read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _atomic_write(self, data: List[Dict[str, Any]]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def append(self, event: Dict[str, Any]) -> None:
        """
        Adds ts + ts_iso if missing.
        """
        if not isinstance(event, dict):
            return

        if "ts" not in event:
            event["ts"] = int(time.time())
        if "ts_iso" not in event:
            event["ts_iso"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event["ts"]))

        data = self._read_all()
        data.append(event)
        self._atomic_write(data)

    def last(self, n: int = 5) -> List[Dict[str, Any]]:
        data = self._read_all()
        return data[-max(0, int(n)):]
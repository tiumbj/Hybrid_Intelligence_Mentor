"""
telegram_notifier.py
Version: 1.1.0
Changelog:
- 1.1.0:
  - load_dotenv override=True (use .env as source of truth for commissioning)
  - send_text supports parse_mode=None (omit parse_mode)
  - add send_text_debug() to return Telegram HTTP status + response body
Rules:
- Never freeze silently: return False + log on any failure
- config.json stores ENV KEY NAMES (token_env/chat_id_env), not raw secrets
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv


logger = logging.getLogger("HIM")


class TelegramNotifier:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.base_dir = os.path.dirname(os.path.abspath(config_path))

        # Load .env from project root (same folder as config.json)
        env_path = os.path.join(self.base_dir, ".env")

        # IMPORTANT: override=True to ensure .env takes precedence during commissioning
        load_dotenv(env_path, override=True)

    def _load_config(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.config_path):
                return {}
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception as e:
            logger.error(f"CRITICAL: TelegramNotifier config load failed: {e}")
            return {}

    def _resolve_credentials(self) -> tuple[bool, Optional[str], Optional[str], list[str]]:
        cfg = self._load_config()
        tg = (cfg.get("telegram") or {}) if isinstance(cfg, dict) else {}

        enabled = bool(tg.get("enabled", False))
        notify_on = tg.get("notify_on", ["signal", "trade", "error"])
        if not isinstance(notify_on, list):
            notify_on = ["signal", "trade", "error"]

        token_env = tg.get("token_env")
        chat_id_env = tg.get("chat_id_env")

        # token_env/chat_id_env must be ENV KEY NAMES (e.g. TELEGRAM_BOT_TOKEN)
        token = os.getenv(str(token_env)) if token_env else None
        chat_id = os.getenv(str(chat_id_env)) if chat_id_env else None

        return enabled, token, chat_id, notify_on

    def send_text_debug(
        self,
        text: str,
        event_type: str = "signal",
        parse_mode: Optional[str] = None,
    ) -> Tuple[bool, int, str]:
        """
        Returns: (ok, http_status, response_text)
        """
        try:
            enabled, token, chat_id, notify_on = self._resolve_credentials()

            if not enabled:
                return False, 0, "telegram.enabled=false"

            if event_type not in notify_on:
                return False, 0, f"event_type={event_type} not in notify_on={notify_on}"

            if not token or not chat_id:
                return False, 0, "missing token/chat_id (check .env + config.json token_env/chat_id_env)"

            url = f"https://api.telegram.org/bot{token}/sendMessage"

            payload = {"chat_id": chat_id, "text": text}
            if parse_mode:
                payload["parse_mode"] = parse_mode  # else omit parse_mode completely

            resp = requests.post(url, json=payload, timeout=10)
            ok = (resp.status_code == 200)

            return ok, int(resp.status_code), resp.text

        except Exception as e:
            return False, 0, f"exception: {e}"

    def send_text(self, text: str, event_type: str = "signal", parse_mode: Optional[str] = None) -> bool:
        ok, status, body = self.send_text_debug(text=text, event_type=event_type, parse_mode=parse_mode)
        if not ok:
            logger.warning(f"Telegram send failed: status={status} body={body[:400]}")
        return ok
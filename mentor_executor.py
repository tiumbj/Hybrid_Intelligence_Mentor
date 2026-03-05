"""
mentor_executor.py
Version: v1.0.0
Purpose: Production Orchestrator for HIM
         Signal Source -> AI Final Confirm -> MT5 Executor

========================================================
DESIGN PRINCIPLES (Production)
========================================================
- Confirm-only: AI must NOT modify plan (entry/sl/tp). It only approves/denies.
- Fail-closed: Missing critical fields -> SKIP
- Single execution entry: always call mt5_executor.execute(execution_package)
- Observability: JSONL logging for every cycle
- Minimal assumptions: Signal/AI accessed via HTTP endpoints (configurable)

========================================================
FILES
========================================================
- logs/mentor_executor.jsonl          : orchestrator audit log
- uses mt5_executor.py (v1.3.1+)      : final gate + order_send + dedup/log

========================================================
CONFIG (env overrides)
========================================================
SIGNAL_URL      default: http://127.0.0.1:5000/api/signal_preview
AI_CONFIRM_URL  default: http://127.0.0.1:5000/api/ai_confirm
POLL_INTERVAL   default: 2.0 seconds
DRY_RUN         default: 0 (0/1)
SYMBOL          default: GOLD

Notes:
- You can keep SIGNAL_URL pointing to your engine endpoint.
- You can keep AI_CONFIRM_URL pointing to your AI confirmation service endpoint.
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import hashlib
import datetime as dt
from typing import Any, Dict, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


# -----------------------------
# VERSION
# -----------------------------
VERSION = "v1.0.0"


# -----------------------------
# DEFAULT CONFIG
# -----------------------------
DEFAULT_SIGNAL_URL = "http://127.0.0.1:5000/api/signal_preview"
DEFAULT_AI_CONFIRM_URL = "http://127.0.0.1:5000/api/ai_confirm"
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_DRY_RUN = "0"
DEFAULT_SYMBOL = "GOLD"

MENTOR_LOG_FILE = os.path.join("logs", "mentor_executor.jsonl")


# -----------------------------
# Utilities
# -----------------------------
def now_epoch() -> float:
    return time.time()


def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()


def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def ensure_logs_dir() -> None:
    os.makedirs(os.path.dirname(MENTOR_LOG_FILE), exist_ok=True)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    line = safe_json(record)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def http_get_json(url: str, timeout_sec: float = 8.0) -> Tuple[bool, Any]:
    try:
        req = Request(url, method="GET", headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return True, json.loads(body)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
        return False, str(e)


def http_post_json(url: str, payload: Dict[str, Any], timeout_sec: float = 12.0) -> Tuple[bool, Any]:
    try:
        data = safe_json(payload).encode("utf-8")
        req = Request(
            url,
            method="POST",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return True, json.loads(body)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
        return False, str(e)


def normalize_decision(x: Any) -> str:
    return str(x or "").upper().strip()


def is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def minimal_plan_ok(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    return all(k in plan for k in ("entry", "sl", "tp"))


def make_request_id(symbol: str, decision: str, plan: Dict[str, Any]) -> str:
    """
    request_id must be unique and stable enough for dedup.
    Strategy:
    - UTC timestamp (to seconds)
    - short hash of (symbol, decision, sl, tp) to help trace
    - random suffix to avoid collision in same second
    """
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    key = f"{symbol}|{decision}|{plan.get('sl')}|{plan.get('tp')}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    r = uuid.uuid4().hex[:6]
    return f"{ts}_{symbol}_{decision}_{h}_{r}"


# -----------------------------
# Core Orchestrator
# -----------------------------
class MentorExecutor:
    def __init__(self) -> None:
        ensure_logs_dir()

        self.signal_url = os.environ.get("SIGNAL_URL", DEFAULT_SIGNAL_URL).strip()
        self.ai_confirm_url = os.environ.get("AI_CONFIRM_URL", DEFAULT_AI_CONFIRM_URL).strip()
        self.poll_interval = float(os.environ.get("POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL)))
        self.dry_run = os.environ.get("DRY_RUN", DEFAULT_DRY_RUN).strip() in ("1", "true", "TRUE", "yes", "YES")
        self.symbol = os.environ.get("SYMBOL", DEFAULT_SYMBOL).strip()

        # import mt5_executor lazily to keep mentor stable even if MT5 not ready
        from mt5_executor import MT5Executor  # type: ignore

        self.mt5 = MT5Executor(symbol=self.symbol)

    def log(self, record: Dict[str, Any]) -> None:
        record.setdefault("ts", now_epoch())
        record.setdefault("utc", now_utc_iso())
        record.setdefault("mentor_version", VERSION)
        append_jsonl(MENTOR_LOG_FILE, record)

    def fetch_signal(self) -> Tuple[bool, Any]:
        return http_get_json(self.signal_url)

    def ai_confirm(self, execution_package: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Sends only what AI needs to approve/deny.
        IMPORTANT: AI must not modify plan; mentor will ignore plan changes if any.
        """
        payload = {
            "request_id": execution_package.get("request_id"),
            "symbol": self.symbol,
            "decision": execution_package.get("decision"),
            "plan": execution_package.get("plan"),
            "context": execution_package.get("context", {}),
        }
        return http_post_json(self.ai_confirm_url, payload)

    def build_execution_package(self, raw_signal: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
        decision = normalize_decision(raw_signal.get("decision"))
        plan = raw_signal.get("plan")

        # Fail-closed: if not actionable, still package it (AI can approve/deny), but mt5_executor will SKIP
        if not minimal_plan_ok(plan):
            return False, {}, "signal_missing_plan"

        if not (is_number(plan.get("entry")) and is_number(plan.get("sl")) and is_number(plan.get("tp"))):
            return False, {}, "signal_plan_invalid_numbers"

        # request_id unique per cycle
        req_id = raw_signal.get("request_id")
        if not isinstance(req_id, str) or not req_id.strip():
            req_id = make_request_id(self.symbol, decision, plan)

        pkg = {
            "request_id": req_id,
            "decision": decision,
            "plan": {
                "entry": float(plan.get("entry")),
                "sl": float(plan.get("sl")),
                "tp": float(plan.get("tp")),
            },
            # optional: any engine metadata / indicators snapshot
            "context": raw_signal.get("context", {}),
            "source": raw_signal.get("source", {}),
        }
        return True, pkg, "ok"

    def enforce_confirm_only(self, pkg: Dict[str, Any], ai_resp: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge AI response into pkg, but DO NOT allow AI to overwrite plan.
        Accept only ai_confirm fields:
          - approved (bool)
          - reason (str)
          - confidence (0..1)
        """
        ai_confirm = {}
        if isinstance(ai_resp, dict):
            approved = ai_resp.get("approved")
            reason = ai_resp.get("reason", "")
            conf = ai_resp.get("confidence", None)

            ai_confirm["approved"] = bool(approved is True)
            ai_confirm["reason"] = str(reason)[:500]

            if is_number(conf):
                c = float(conf)
                if c < 0:
                    c = 0.0
                if c > 1:
                    c = 1.0
                ai_confirm["confidence"] = c
            else:
                ai_confirm["confidence"] = None

        pkg["ai_confirm"] = ai_confirm
        return pkg

    def run_once(self) -> Dict[str, Any]:
        # 1) Fetch signal
        ok, raw = self.fetch_signal()
        if not ok:
            out = {"status": "SKIP", "reason": "signal_fetch_failed", "detail": raw}
            self.log({"event": "signal_fetch_failed", "out": out})
            return out

        if not isinstance(raw, dict):
            out = {"status": "SKIP", "reason": "signal_invalid_format"}
            self.log({"event": "signal_invalid_format", "raw": raw, "out": out})
            return out

        # 2) Build execution package (pre-AI)
        ok_pkg, pkg, why = self.build_execution_package(raw)
        if not ok_pkg:
            out = {"status": "SKIP", "reason": why}
            self.log({"event": "build_pkg_failed", "raw": raw, "out": out})
            return out

        # 3) AI final confirm
        ok_ai, ai = self.ai_confirm(pkg)
        if not ok_ai:
            # Fail-closed: AI unreachable -> deny by default
            pkg["ai_confirm"] = {"approved": False, "reason": "ai_unreachable", "confidence": None}
            self.log({"event": "ai_call_failed", "request_id": pkg["request_id"], "detail": ai})

            if self.dry_run:
                out = {"status": "DRY_RUN", "request_id": pkg["request_id"], "reason": "ai_unreachable"}
                self.log({"event": "dry_run_end", "pkg": pkg, "out": out})
                return out

            # pass to mt5_executor (it will SKIP ai_denied)
            mt5_out = self.mt5.execute(pkg)
            self.log({"event": "mt5_execute", "request_id": pkg["request_id"], "pkg": pkg, "mt5_out": mt5_out})
            return mt5_out

        if not isinstance(ai, dict):
            ai = {"approved": False, "reason": "ai_invalid_response", "confidence": None}

        pkg = self.enforce_confirm_only(pkg, ai)

        # 4) Dry-run option
        if self.dry_run:
            out = {"status": "DRY_RUN", "request_id": pkg["request_id"], "ai_confirm": pkg.get("ai_confirm")}
            self.log({"event": "dry_run_end", "pkg": pkg, "out": out})
            return out

        # 5) Execute via mt5_executor (final authority)
        mt5_out = self.mt5.execute(pkg)
        self.log({"event": "mt5_execute", "request_id": pkg["request_id"], "pkg": pkg, "mt5_out": mt5_out})
        return mt5_out

    def loop(self) -> None:
        self.log(
            {
                "event": "mentor_start",
                "signal_url": self.signal_url,
                "ai_confirm_url": self.ai_confirm_url,
                "poll_interval": self.poll_interval,
                "dry_run": self.dry_run,
                "symbol": self.symbol,
            }
        )

        while True:
            out = self.run_once()
            print(safe_json(out))
            time.sleep(self.poll_interval)


# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    print(f"[MentorExecutor] file={os.path.abspath(__file__)} version={VERSION}")
    try:
        m = MentorExecutor()
    except Exception as e:
        print(safe_json({"status": "FATAL", "reason": "init_failed", "detail": str(e)}))
        sys.exit(1)

    # default: loop forever
    m.loop()
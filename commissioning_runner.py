"""
Hybrid Intelligence Mentor (HIM)
File: commissioning_runner.py
Version: v2.12.4 (Event-Based Commissioning)
Date: 2026-03-02 (Asia/Bangkok)

CHANGELOG
- v2.12.4:
  - Implement event-based gating:
      * NEW_BAR: trigger only when a new bar appears (per timeframe)
      * SIGNAL_CHANGE: trigger when signal signature changes
  - Fail-closed guards:
      * MT5 initialize must succeed
      * Tick freshness must pass (uses mt5.time_current vs tick.time)
      * If signal eval unavailable, fallback to NEW_BAR only (still reduces duplicates)
  - JSONL event log: logs/commissioning_events.jsonl
  - Does NOT rely on PowerShell loop for production; runs as a controlled runner

RATIONALE (Production)
- Prior issue: logs/commissioning.jsonl duplicated by evaluation loop (e.g., total=401 repeated)
- Fix: define "event" and log/execute only on events.
"""

from __future__ import annotations

import json
import os
import sys
import time
import hashlib
import traceback
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List

try:
    import MetaTrader5 as mt5
except Exception as e:
    print(f"[FATAL] MetaTrader5 import failed: {e}")
    sys.exit(1)


# ----------------------------
# Utility
# ----------------------------

def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ----------------------------
# Config Load (fail-closed)
# ----------------------------

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.json must be an object/dict")
    return cfg


def cfg_get(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ----------------------------
# MT5 Guards (tick freshness)
# ----------------------------

@dataclass
class TickSnapshot:
    bid: float
    ask: float
    last: float
    tick_time: int
    server_time: int
    tick_age_sec: float


def get_tick_snapshot(symbol: str) -> Optional[TickSnapshot]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    # Production-safe time base: mt5.time_current()
    server_time = int(mt5.time_current())
    tick_time = int(getattr(tick, "time", 0))

    age = server_time - tick_time
    if age < 0:
        age = 0  # clamp (fail-closed would be "treat as fresh?" — but clamp is safer to avoid negative noise)

    return TickSnapshot(
        bid=safe_float(getattr(tick, "bid", 0.0)),
        ask=safe_float(getattr(tick, "ask", 0.0)),
        last=safe_float(getattr(tick, "last", 0.0)),
        tick_time=tick_time,
        server_time=server_time,
        tick_age_sec=float(age),
    )


def tick_is_fresh(tick: TickSnapshot, max_age_sec: float) -> bool:
    return tick.tick_age_sec <= max_age_sec


# ----------------------------
# Timeframe mapping
# ----------------------------

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


def get_latest_bar_time(symbol: str, tf_code: int) -> Optional[int]:
    """
    Return latest bar open time (unix seconds). Fail-closed returns None on errors.
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, 1)
        if rates is None or len(rates) == 0:
            return None
        # numpy structured array: rates[0]['time']
        return int(rates[0]["time"])
    except Exception:
        return None


# ----------------------------
# Signal evaluation (best-effort)
# ----------------------------

def try_eval_signal_signature(config_path: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Best-effort: import TradingEngine and attempt to produce a *stable signature*.

    Returns:
      (signature_str_or_None, debug_payload)
    If cannot evaluate, returns (None, debug payload).
    """
    dbg: Dict[str, Any] = {"ok": False, "method": None, "error": None}
    try:
        from engine import TradingEngine  # type: ignore

        e = TradingEngine(config_path)

        # Try common method names in order (avoid breaking existing engine)
        candidates = ["generate_signal", "build_signal", "run_once", "evaluate", "get_signal"]
        signal_obj: Any = None
        used = None

        for name in candidates:
            if hasattr(e, name) and callable(getattr(e, name)):
                used = name
                signal_obj = getattr(e, name)()
                break

        if used is None:
            dbg["error"] = "TradingEngine has no known eval method"
            return None, dbg

        dbg["method"] = used
        dbg["ok"] = True

        # Normalize to dict for hashing
        if isinstance(signal_obj, dict):
            sig_dict = signal_obj
        else:
            # Attempt common attribute patterns
            if hasattr(signal_obj, "to_dict") and callable(getattr(signal_obj, "to_dict")):
                sig_dict = signal_obj.to_dict()
            elif hasattr(signal_obj, "__dict__"):
                sig_dict = dict(signal_obj.__dict__)
            else:
                sig_dict = {"repr": repr(signal_obj)}

        # Keep only key fields (reduce noise, keep "decision intent")
        # We do not know exact schema, so we pick by existence.
        keep_keys = [
            "symbol", "timeframe", "mode",
            "direction", "side", "signal",
            "decision", "action",
            "blocked_by", "gates", "reasons",
            "entry", "sl", "tp", "rr",
            "confidence",
        ]
        compact: Dict[str, Any] = {}
        for k in keep_keys:
            if k in sig_dict:
                compact[k] = sig_dict[k]

        # If blocked_by is nested, keep stable string form
        if "blocked_by" in compact and isinstance(compact["blocked_by"], (list, dict)):
            compact["blocked_by"] = json_dumps_compact(compact["blocked_by"])

        # Fallback: if compact empty, hash full repr but may be noisy
        base = compact if compact else {"repr": repr(sig_dict)}

        sig_raw = json_dumps_compact(base)
        signature = sha1_hex(sig_raw)
        dbg["signature_base"] = base
        return signature, dbg

    except Exception as ex:
        dbg["error"] = f"{type(ex).__name__}: {ex}"
        dbg["trace"] = traceback.format_exc(limit=2)
        return None, dbg


# ----------------------------
# Event Gating State
# ----------------------------

@dataclass
class EventState:
    last_bar_time: Dict[str, int]        # key=tf_name
    last_signature: Optional[str]


def default_event_state() -> EventState:
    return EventState(last_bar_time={}, last_signature=None)


def should_fire_event(
    symbol: str,
    tfs: List[str],
    state: EventState,
    config_path: str,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Decide whether to trigger an event.

    Returns:
      (fire, reasons, details)
    """
    reasons: List[str] = []
    details: Dict[str, Any] = {"bars": {}, "signal": {}}

    # 1) NEW_BAR (per TF)
    new_bar = False
    for tf_name in tfs:
        tf_code = TF_MAP.get(tf_name)
        if tf_code is None:
            continue
        bt = get_latest_bar_time(symbol, tf_code)
        details["bars"][tf_name] = bt
        if bt is None:
            continue
        prev = state.last_bar_time.get(tf_name)
        if prev is None:
            # first init does not count as event (prevents immediate burst)
            state.last_bar_time[tf_name] = bt
        elif bt != prev:
            new_bar = True
            state.last_bar_time[tf_name] = bt

    if new_bar:
        reasons.append("NEW_BAR")

    # 2) SIGNAL_CHANGE (best-effort)
    sig, sig_dbg = try_eval_signal_signature(config_path)
    details["signal"] = {"signature": sig, "dbg": sig_dbg}

    if sig is not None:
        if state.last_signature is None:
            # first signature init does not count as event
            state.last_signature = sig
        elif sig != state.last_signature:
            reasons.append("SIGNAL_CHANGE")
            state.last_signature = sig

    fire = len(reasons) > 0
    return fire, reasons, details


# ----------------------------
# Event Log (JSONL)
# ----------------------------

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(path)
    line = json_dumps_compact(obj)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------------------
# Executor Invocation (one-shot)
# ----------------------------

def run_mentor_executor(py_exe: str, config_path: str) -> Tuple[int, str]:
    """
    Calls mentor_executor.py as a subprocess (one-shot).
    Fail-closed: returns non-zero code if failed.
    """
    cmd = [py_exe, "mentor_executor.py", "--config", config_path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return int(p.returncode), out.strip()
    except Exception as e:
        return 99, f"executor_call_failed: {type(e).__name__}: {e}"


# ----------------------------
# Main Runner
# ----------------------------

def main() -> int:
    config_path = os.environ.get("HIM_CONFIG", "config.json")
    py_exe = sys.executable  # use current venv python

    try:
        cfg = load_config(config_path)
    except Exception as e:
        print(f"[FATAL] load_config failed: {e}")
        return 2

    symbol = str(cfg.get("symbol", "GOLD"))
    tick_max_age_sec = safe_float(cfg_get(cfg, ["execution", "tick_max_age_sec"], 15), 15)
    poll_sec = safe_float(cfg_get(cfg, ["commissioning", "poll_sec"], 1.0), 1.0)

    # Which timeframes define NEW_BAR event
    # If not set, default to ["M5"] (practical commissioning TF)
    tfs = cfg_get(cfg, ["commissioning", "event_timeframes"], ["M5"])
    if not isinstance(tfs, list) or not tfs:
        tfs = ["M5"]
    tfs = [str(x).upper() for x in tfs if str(x).upper() in TF_MAP]

    events_log_path = str(cfg_get(cfg, ["commissioning", "events_log_path"], "logs/commissioning_events.jsonl"))

    # MT5 init (fail-closed)
    if not mt5.initialize():
        print("[FATAL] mt5.initialize() failed")
        return 3

    # Ensure symbol selected/visible
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        print(f"[FATAL] symbol_info is None for symbol={symbol}")
        return 4
    if not sym_info.visible:
        mt5.symbol_select(symbol, True)

    state = default_event_state()

    print(json_dumps_compact({
        "ts": utc_iso_now(),
        "msg": "commissioning_runner_started",
        "version": "v2.12.4",
        "symbol": symbol,
        "tick_max_age_sec": tick_max_age_sec,
        "poll_sec": poll_sec,
        "event_timeframes": tfs,
        "events_log_path": events_log_path,
    }))

    while True:
        try:
            tick = get_tick_snapshot(symbol)
            if tick is None:
                # fail-closed: no tick => no event, no executor
                append_jsonl(events_log_path, {
                    "ts": utc_iso_now(),
                    "type": "HEARTBEAT",
                    "symbol": symbol,
                    "ok": False,
                    "reason": "NO_TICK",
                })
                time.sleep(poll_sec)
                continue

            if not tick_is_fresh(tick, tick_max_age_sec):
                append_jsonl(events_log_path, {
                    "ts": utc_iso_now(),
                    "type": "HEARTBEAT",
                    "symbol": symbol,
                    "ok": False,
                    "reason": "STALE_TICK",
                    "tick_age_sec": tick.tick_age_sec,
                    "tick_time": tick.tick_time,
                    "server_time": tick.server_time,
                })
                time.sleep(poll_sec)
                continue

            fire, reasons, details = should_fire_event(symbol, tfs, state, config_path)
            if not fire:
                # quiet heartbeat (optional)
                time.sleep(poll_sec)
                continue

            # Log the event (event-based)
            event_payload = {
                "ts": utc_iso_now(),
                "type": "EVENT",
                "symbol": symbol,
                "reasons": reasons,
                "tick": {
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "last": tick.last,
                    "tick_time": tick.tick_time,
                    "server_time": tick.server_time,
                    "tick_age_sec": tick.tick_age_sec,
                },
                "details": details,
            }
            append_jsonl(events_log_path, event_payload)

            # Trigger executor only on event
            rc, out = run_mentor_executor(py_exe, config_path)
            append_jsonl(events_log_path, {
                "ts": utc_iso_now(),
                "type": "EXECUTOR_RESULT",
                "symbol": symbol,
                "returncode": rc,
                "output_tail": out[-4000:],  # prevent huge logs
            })

            # pacing
            time.sleep(poll_sec)

        except KeyboardInterrupt:
            print(json_dumps_compact({"ts": utc_iso_now(), "msg": "stopped_by_user"}))
            break
        except Exception as e:
            append_jsonl(events_log_path, {
                "ts": utc_iso_now(),
                "type": "ERROR",
                "symbol": symbol,
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=3),
            })
            time.sleep(max(poll_sec, 1.0))

    mt5.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
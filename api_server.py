"""
api_server.py
Version: 1.1.4
Changelog:
- 1.1.4:
  - Fix: normalize config values (replace None/invalid/empty dict) with sane defaults
  - Auto-migrate: if config was normalized, write back to config.json to prevent recurring engine float(None) crashes
  - /api/status config snapshot becomes stable (no blank risk/supertrend/timeframes if they were None)
Notes:
- Commissioning default for Telegram test remains plain text (parse_mode=None)
- Never calls mt5.shutdown()
"""

from __future__ import annotations

import os
import json
import time
import threading
import logging
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request, send_from_directory

import MetaTrader5 as mt5

from engine import TradingEngine
from telegram_notifier import TelegramNotifier
from performance_tracker import PerformanceTracker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("him_system.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("HIM")


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error(f"CRITICAL: cannot read json: {path} err={e}")
        return {}


def _safe_write_json(path: str, data: Dict[str, Any]) -> bool:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return True
    except Exception as e:
        logger.error(f"CRITICAL: cannot write json: {path} err={e}")
        return False


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(dst) if isinstance(dst, dict) else {}
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _ensure_mt5() -> Tuple[bool, str]:
    try:
        if mt5.terminal_info() is not None:
            return True, "already_initialized"
        ok = mt5.initialize()
        if not ok:
            return False, f"initialize_failed: {mt5.last_error()}"
        return True, "initialized"
    except Exception as e:
        return False, f"exception: {e}"


def _get_tick(symbol: str) -> Dict[str, Any]:
    ok, msg = _ensure_mt5()
    if not ok:
        return {"ok": False, "error": msg, "last_error": str(mt5.last_error())}

    t = mt5.symbol_info_tick(symbol)
    if t is None:
        return {"ok": False, "error": f"tick_none for {symbol}", "last_error": str(mt5.last_error())}

    return {
        "ok": True,
        "symbol": symbol,
        "time_msc": int(getattr(t, "time_msc", 0)),
        "bid": float(getattr(t, "bid", 0.0)),
        "ask": float(getattr(t, "ask", 0.0)),
        "last": float(getattr(t, "last", 0.0)),
    }


def _mt5_snapshot(symbol: str) -> Dict[str, Any]:
    ok, msg = _ensure_mt5()
    snap: Dict[str, Any] = {
        "mt5_ok": ok,
        "mt5_state": msg,
        "symbol": symbol,
        "account": None,
        "positions": [],
        "positions_count": 0,
        "last_error": str(mt5.last_error()),
    }
    if not ok:
        return snap

    acc = mt5.account_info()
    if acc:
        snap["account"] = {
            "login": getattr(acc, "login", None),
            "balance": float(getattr(acc, "balance", 0.0)),
            "equity": float(getattr(acc, "equity", 0.0)),
            "margin": float(getattr(acc, "margin", 0.0)),
            "margin_free": float(getattr(acc, "margin_free", 0.0)),
            "profit": float(getattr(acc, "profit", 0.0)),
            "currency": getattr(acc, "currency", None),
        }

    positions = mt5.positions_get(symbol=symbol)
    if positions:
        snap["positions_count"] = len(positions)
        for p in positions[:30]:
            snap["positions"].append(
                {
                    "ticket": int(getattr(p, "ticket", 0)),
                    "type": "BUY" if int(getattr(p, "type", 0)) == mt5.ORDER_TYPE_BUY else "SELL",
                    "volume": float(getattr(p, "volume", 0.0)),
                    "price_open": float(getattr(p, "price_open", 0.0)),
                    "price_current": float(getattr(p, "price_current", 0.0)),
                    "sl": float(getattr(p, "sl", 0.0)),
                    "tp": float(getattr(p, "tp", 0.0)),
                    "profit": float(getattr(p, "profit", 0.0)),
                }
            )

    return snap


def _build_dry_run_order(symbol: str, direction: str) -> Dict[str, Any]:
    tick = _get_tick(symbol)
    if not tick.get("ok"):
        return {"ok": False, "error": tick.get("error"), "tick": tick}

    entry = tick["ask"] if direction == "BUY" else tick["bid"]
    return {
        "ok": True,
        "symbol": symbol,
        "direction": direction,
        "entry_ref": entry,
        "note": "DRY-RUN only (no order_send). SL/TP must come from engine/executor policy.",
        "tick": tick,
    }


def _is_blank_dict(x: Any) -> bool:
    return isinstance(x, dict) and len(x) == 0


def _normalize_config(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Replace None/invalid values with defaults (commissioning safe).
    Returns (normalized_cfg, changed_flag).
    """
    changed = False
    out = dict(cfg) if isinstance(cfg, dict) else {}

    def set_if_none_or_missing(key: str, default: Any) -> None:
        nonlocal changed, out
        if key not in out or out.get(key) is None:
            out[key] = default
            changed = True

    def ensure_dict(key: str, default: Dict[str, Any]) -> None:
        nonlocal changed, out
        v = out.get(key)
        if v is None or not isinstance(v, dict) or _is_blank_dict(v):
            out[key] = default
            changed = True

    def ensure_number(key: str, default: float) -> None:
        nonlocal changed, out
        v = out.get(key)
        if v is None:
            out[key] = default
            changed = True
            return
        try:
            # Accept int/float/str numeric
            out[key] = float(v)
        except Exception:
            out[key] = default
            changed = True

    def ensure_int(key: str, default: int) -> None:
        nonlocal changed, out
        v = out.get(key)
        if v is None:
            out[key] = default
            changed = True
            return
        try:
            out[key] = int(float(v))
        except Exception:
            out[key] = default
            changed = True

    # Top-level defaults
    set_if_none_or_missing("symbol", "GOLD")
    set_if_none_or_missing("enable_execution", False)
    ensure_int("confidence_threshold", 75)
    ensure_number("min_score", 7.0)
    ensure_number("min_rr", 2.0)
    ensure_number("lot", 0.01)

    # Nested defaults (critical: prevent None flowing into engine)
    ensure_dict("supertrend", {"period": 21, "multiplier": 4.0})
    if isinstance(out.get("supertrend"), dict):
        st = out["supertrend"]
        if st.get("period") is None:
            st["period"] = 21
            changed = True
        if st.get("multiplier") is None:
            st["multiplier"] = 4.0
            changed = True

    ensure_dict("structure_sensitivity", {"htf": 4, "mtf": 3, "ltf": 1})
    ensure_dict("timeframes", {"htf": "H1", "mtf": "M30", "ltf": "M5"})
    ensure_dict("risk", {"atr_period": 14, "atr_sl_mult": 1.8})

    # Optional section
    if out.get("breakout_proximity") is None:
        out["breakout_proximity"] = {}
        changed = True

    # AI defaults (status only)
    ensure_dict("ai", {"enabled": False, "provider": "openai", "api_key_env": "OPENAI_API_KEY", "model": ""})
    ai = out.get("ai", {})
    if isinstance(ai, dict):
        if ai.get("enabled") is None:
            ai["enabled"] = False
            changed = True
        if not (ai.get("provider") or "").strip():
            ai["provider"] = "openai"
            changed = True
        if not (ai.get("api_key_env") or "").strip():
            ai["api_key_env"] = "OPENAI_API_KEY"
            changed = True
        if ai.get("model") is None:
            ai["model"] = ""
            changed = True
        out["ai"] = ai

    return out, changed


MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "A": {
        "confidence_threshold": 85,
        "min_score": 8.5,
        "min_rr": 2.5,
        "structure_sensitivity": {"htf": 4, "mtf": 3, "ltf": 1},
        "timeframes": {"htf": "H1", "mtf": "M30", "ltf": "M5"},
        "supertrend": {"period": 21, "multiplier": 4.0},
        "breakout_proximity": {"threshold": 2.0, "min_score": 60},
    },
    "B": {
        "confidence_threshold": 75,
        "min_score": 7.0,
        "min_rr": 2.0,
        "structure_sensitivity": {"htf": 4, "mtf": 3, "ltf": 1},
        "timeframes": {"htf": "H1", "mtf": "M30", "ltf": "M5"},
        "supertrend": {"period": 21, "multiplier": 4.0},
        "breakout_proximity": {"threshold": 2.0, "min_score": 50},
    },
    "C": {
        "confidence_threshold": 65,
        "min_score": 6.0,
        "min_rr": 1.7,
        "structure_sensitivity": {"htf": 4, "mtf": 3, "ltf": 1},
        "timeframes": {"htf": "H1", "mtf": "M30", "ltf": "M5"},
        "supertrend": {"period": 21, "multiplier": 4.0},
        "breakout_proximity": {"threshold": 2.5, "min_score": 40},
    },
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DASHBOARD_HTML = os.path.join(BASE_DIR, "dashboard.html")

app = Flask(__name__)
mt5_lock = threading.Lock()

engine = TradingEngine(config_path=CONFIG_PATH)
tg = TelegramNotifier(config_path=CONFIG_PATH)


def _load_config() -> Dict[str, Any]:
    raw = _safe_read_json(CONFIG_PATH)
    cfg, changed = _normalize_config(raw)

    if changed:
        ok = _safe_write_json(CONFIG_PATH, cfg)
        logger.info(f"Config normalized (changed=True) write_back={ok}")

    return cfg


def _save_config_patch(patch: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(patch, dict):
        return False, "patch_not_object"
    cur = _load_config()
    merged = _deep_merge(cur, patch)
    merged, changed = _normalize_config(merged)
    ok = _safe_write_json(CONFIG_PATH, merged)
    return ok, ("ok" if ok else "write_failed")


def _ai_status(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ai = cfg.get("ai") if isinstance(cfg, dict) else {}
    if not isinstance(ai, dict):
        ai = {}

    provider = str(ai.get("provider") or "openai").strip().lower()
    enabled = bool(ai.get("enabled", False))

    api_key_env = str(ai.get("api_key_env") or ("OPENAI_API_KEY" if provider == "openai" else "AI_API_KEY")).strip()
    model = str(ai.get("model") or "").strip()

    key_present = bool(os.getenv(api_key_env)) if api_key_env else False
    ok = bool(enabled and key_present and bool(model))

    return {
        "ok": ok,
        "enabled": enabled,
        "provider": provider,
        "model": model,
        "api_key_env": api_key_env,
        "key_present": key_present,
        "note": "Status only (no API call).",
    }


def _telegram_status() -> Dict[str, Any]:
    try:
        enabled, token, chat_id, notify_on = tg._resolve_credentials()  # type: ignore[attr-defined]
        return {
            "ok": bool(enabled and token and chat_id),
            "enabled": bool(enabled),
            "has_token": bool(token),
            "has_chat_id": bool(chat_id),
            "notify_on": notify_on,
            "note": "Status only (no send).",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/")
def root():
    if os.path.exists(DASHBOARD_HTML):
        return send_from_directory(BASE_DIR, "dashboard.html")
    return jsonify({"error": "dashboard.html not found"}), 404


@app.get("/dashboard.html")
def dashboard_html():
    if os.path.exists(DASHBOARD_HTML):
        return send_from_directory(BASE_DIR, "dashboard.html")
    return jsonify({"error": "dashboard.html not found"}), 404


@app.get("/api/config")
def api_get_config():
    return jsonify(_load_config())


@app.post("/api/config")
def api_post_config():
    patch = request.get_json(silent=True) or {}
    ok, msg = _save_config_patch(patch)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400
    return jsonify({"ok": True})


@app.post("/api/mode/<mode_id>")
def api_apply_mode(mode_id: str):
    mode_id = (mode_id or "").strip().upper()
    preset = MODE_PRESETS.get(mode_id)
    if not preset:
        return jsonify({"ok": False, "error": f"unknown_mode: {mode_id}"}), 404

    ok, msg = _save_config_patch(preset)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400
    return jsonify({"ok": True, "mode": mode_id})


@app.get("/api/ai_status")
def api_ai_status():
    cfg = _load_config()
    return jsonify(_ai_status(cfg))


@app.get("/api/telegram_status")
def api_telegram_status():
    return jsonify(_telegram_status())


@app.get("/api/status")
def api_status():
    cfg = _load_config()
    symbol = str(cfg.get("symbol", "GOLD"))

    with mt5_lock:
        mt5snap = _mt5_snapshot(symbol)
        price = _get_tick(symbol)

    return jsonify(
        {
            "ts": int(time.time()),
            "config": {
                "symbol": symbol,
                "enable_execution": bool(cfg.get("enable_execution", False)),
                "confidence_threshold": int(cfg.get("confidence_threshold", 75)),
                "min_score": float(cfg.get("min_score", 7.0)),
                "min_rr": float(cfg.get("min_rr", 2.0)),
                "lot": float(cfg.get("lot", 0.01)),
                "timeframes": cfg.get("timeframes", {}),
                "structure_sensitivity": cfg.get("structure_sensitivity", {}),
                "supertrend": cfg.get("supertrend", {}),
                "risk": cfg.get("risk", {}),
                "breakout_proximity": cfg.get("breakout_proximity", {}),
            },
            "mt5": mt5snap,
            "price": price,
            "ai": _ai_status(cfg),
            "telegram": _telegram_status(),
            "log_file": {
                "exists": os.path.exists(os.path.join(BASE_DIR, "him_system.log")),
                "path": os.path.join(BASE_DIR, "him_system.log"),
            },
        }
    )


@app.get("/api/signal_preview")
def api_signal_preview():
    cfg = _load_config()
    symbol = str(cfg.get("symbol", "GOLD"))

    with mt5_lock:
        try:
            pkg = engine.generate_signal_package()
        except Exception as e:
            logger.error(f"CRITICAL: engine.generate_signal_package failed: {e}")
            return jsonify({"ok": False, "error": f"engine_failed: {e}"}), 500

        direction = str((pkg or {}).get("direction", "NONE")).upper()
        dry = _build_dry_run_order(symbol=symbol, direction=direction) if direction in ("BUY", "SELL") else {"ok": True, "note": "direction=NONE"}

    return jsonify({"ok": True, "symbol": symbol, "package": pkg, "dry_run_order": dry})


@app.get("/api/performance")
def api_performance():
    cfg = _load_config()
    symbol = str(cfg.get("symbol", "GOLD"))

    with mt5_lock:
        try:
            pt = PerformanceTracker(stats_file="performance_stats.json", symbol=symbol, lookback_days=365)
            pt.sync_trade_closes()
            s = pt.summary()

            trades_raw = pt.stats.get("trades", []) if isinstance(pt.stats, dict) else []
            trades = [t for t in trades_raw if isinstance(t, dict) and t.get("deal_ticket") is not None and t.get("time_iso")]
            trades = sorted(trades, key=lambda x: int(x.get("time", 0)), reverse=True)[:20]

        except Exception as e:
            logger.error(f"CRITICAL: performance failed: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify(
        {
            "ok": True,
            "symbol": symbol,
            "summary": {
                "total_trades": s.total_trades,
                "wins": s.wins,
                "losses": s.losses,
                "win_rate": s.win_rate,
                "net_profit": s.net_profit,
                "gross_profit": s.gross_profit,
                "gross_loss": s.gross_loss,
                "profit_factor": s.profit_factor,
                "best_trade": s.best_trade,
                "worst_trade": s.worst_trade,
            },
            "recent": trades,
        }
    )


@app.post("/api/telegram_test")
def api_telegram_test():
    payload = request.get_json(silent=True) or {}

    # Commissioning default: plain text only
    text = str(payload.get("text") or "HIM Telegram Test: OK\n(plain text)")
    event_type = str(payload.get("event_type") or "signal")

    ok, status, body = tg.send_text_debug(text=text, event_type=event_type, parse_mode=None)

    return jsonify(
        {
            "ok": bool(ok),
            "event_type": event_type,
            "telegram_http_status": int(status),
            "telegram_response_snippet": (body or "")[:400],
            "telegram": _telegram_status(),
        }
    )


def main() -> None:
    logger.info("HIM API Server starting...")
    logger.info("Open: http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
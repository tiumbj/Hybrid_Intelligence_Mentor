"""
api_server.py
Version: 3.2.2
Production-safe config reload + resolver fix
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory

from ai_mentor import AIMentor
from config_resolver import resolve_effective_config

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


app = Flask(__name__)
CONFIG_PATH = "config.json"


# =========================================================
# Helpers
# =========================================================

def _load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_config(data: Dict[str, Any]) -> bool:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _effective_config() -> Dict[str, Any]:
    """
    Deterministic config loader:
    - Always reload from disk
    - Always resolve with dict
    - Always provide safe defaults
    """
    raw = _load_config()
    eff = resolve_effective_config(raw) or {}

    # Safe fallbacks (prevent null in dashboard)
    eff.setdefault("symbol", "GOLD")
    eff.setdefault("enable_execution", False)
    eff.setdefault("confidence_threshold", 0)
    eff.setdefault("min_score", 0)
    eff.setdefault("min_rr", 0)
    eff.setdefault("lot", 0.0)

    return eff


# =========================================================
# Static
# =========================================================

@app.get("/")
def root_dashboard():
    return send_from_directory(".", "dashboard.html")


@app.get("/favicon.ico")
def favicon():
    if os.path.exists("favicon.ico"):
        return send_from_directory(".", "favicon.ico")
    return ("", 204)


# =========================================================
# API: CONFIG
# =========================================================

@app.get("/api/config")
def api_get_config():
    return jsonify(_load_config()), 200


@app.post("/api/config")
def api_set_config():
    patch = request.get_json(silent=True) or {}
    if not isinstance(patch, dict):
        return jsonify({"ok": False, "error": "payload_must_be_object"}), 400

    current = _load_config()
    merged = _deep_merge(current, patch)

    ok = _save_config(merged)
    return jsonify({"ok": ok}), (200 if ok else 500)


@app.post("/api/mode/<mode_id>")
def api_set_mode(mode_id: str):
    cfg = _load_config()
    cfg["mode"] = str(mode_id)
    ok = _save_config(cfg)
    return jsonify({"ok": ok, "mode": cfg.get("mode")}), 200


# =========================================================
# API: STATUS
# =========================================================

def _mt5_snapshot() -> Dict[str, Any]:
    out = {
        "mt5_ok": False,
        "account_login": None,
        "account_server": None,
        "account_currency": None,
        "account_leverage": None,
        "positions_count": None,
        "last_error": None,
    }

    if mt5 is None:
        out["last_error"] = "MetaTrader5 not installed"
        return out

    try:
        if not mt5.initialize():
            out["last_error"] = str(mt5.last_error())
            return out

        acc = mt5.account_info()
        if acc:
            out["mt5_ok"] = True
            out["account_login"] = acc.login
            out["account_server"] = acc.server
            out["account_currency"] = acc.currency
            out["account_leverage"] = acc.leverage
            out["positions_count"] = mt5.positions_total()
    except Exception as e:
        out["last_error"] = str(e)
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

    return out


@app.get("/api/status")
def api_status():
    eff = _effective_config()

    return jsonify({
        "ok": True,
        "ts": time.time(),
        "config": {
            "mode": eff.get("mode"),
            "symbol": eff.get("symbol"),
            "enable_execution": bool(eff.get("enable_execution")),
            "confidence_threshold": eff.get("confidence_threshold"),
            "min_score": eff.get("min_score"),
            "min_rr": eff.get("min_rr"),
            "lot": eff.get("lot"),
        },
        "mt5": _mt5_snapshot(),
    }), 200


# =========================================================
# API: SIGNAL PREVIEW
# =========================================================

@app.get("/api/signal_preview")
def api_signal_preview():
    eff = _effective_config()
    return jsonify({
        "ok": True,
        "ts": time.time(),
        "symbol": eff.get("symbol"),
        "mode": eff.get("mode"),
        "note": "engine snapshot integration pending"
    }), 200


# =========================================================
# AI CONFIRM
# =========================================================

@app.post("/api/ai_confirm")
def api_ai_confirm():
    payload = request.get_json(silent=True) or {}
    mentor = AIMentor()
    out = mentor.evaluate(payload) or {}

    decision = "CONFIRM"
    if out.get("execution", {}).get("reject"):
        decision = "REJECT"

    return jsonify({
        "schema_version": "1.0",
        "decision": decision,
        "confidence": float(out.get("execution", {}).get("conf", 0)) / 100.0,
        "entry": out.get("execution", {}).get("entry"),
        "sl": out.get("execution", {}).get("sl"),
        "tp": out.get("execution", {}).get("tp"),
        "note": (out.get("mentor") or {}).get("headline"),
    }), 200


# =========================================================

def main() -> int:
    host = os.environ.get("HIM_API_HOST", "127.0.0.1")
    port = int(os.environ.get("HIM_API_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
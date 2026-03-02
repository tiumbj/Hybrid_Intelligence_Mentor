"""
api_server.py
Version: 3.2.4

CHANGELOG
- 3.2.4 (2026-03-02)
  - FIX: resolve_effective_config now receives loaded dict (not path string)
  - FIX: /api/status returns MT5 snapshot again (so Dashboard shows CONNECTED correctly)
  - KEEP: confirm_live guard
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Tuple, Optional

from flask import Flask, jsonify, request, send_from_directory

from config_resolver import resolve_effective_config

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


app = Flask(__name__)

CONFIG_PATH_DEFAULT = "config.json"
CONFIRM_LIVE_TEXT = "CONFIRM LIVE"


# -----------------------------
# JSON helpers
# -----------------------------
def _load_json_file(path: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_json_file(path: str, data: Dict[str, Any]) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def _get_effective_config() -> Dict[str, Any]:
    raw = _load_json_file(CONFIG_PATH_DEFAULT)
    eff = resolve_effective_config(raw) or {}
    return eff


# -----------------------------
# MT5 snapshot (best-effort)
# -----------------------------
def _mt5_snapshot(symbol: str) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "mt5_ok": False,
        "account_login": None,
        "account_server": None,
        "account_currency": None,
        "account_leverage": None,
        "positions_count": None,
        "last_error": None,
    }

    if mt5 is None:
        snap["last_error"] = "MetaTrader5 module not available"
        return snap

    try:
        ok = mt5.initialize()
        snap["last_error"] = mt5.last_error()
        if not ok:
            snap["mt5_ok"] = False
            return snap

        ai = mt5.account_info()
        if ai is not None:
            snap["account_login"] = getattr(ai, "login", None)
            snap["account_server"] = getattr(ai, "server", None)
            snap["account_currency"] = getattr(ai, "currency", None)
            snap["account_leverage"] = getattr(ai, "leverage", None)

        # positions count
        pos = mt5.positions_get()
        snap["positions_count"] = 0 if pos is None else len(pos)

        snap["mt5_ok"] = True
        return snap
    except Exception as e:
        snap["mt5_ok"] = False
        snap["last_error"] = f"exception: {e}"
        return snap
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


# -----------------------------
# Dashboard
# -----------------------------
@app.get("/")
def root_dashboard():
    return send_from_directory(".", "dashboard.html")


@app.get("/favicon.ico")
def favicon():
    if os.path.exists("favicon.ico"):
        return send_from_directory(".", "favicon.ico")
    return ("", 204)


# -----------------------------
# Health
# -----------------------------
@app.get("/api/health")
def api_health():
    cfg_eff = _get_effective_config()
    return jsonify({"ok": True, "ts": time.time(), "config_loaded": bool(cfg_eff)}), 200


# -----------------------------
# Config
# -----------------------------
@app.get("/api/config")
def api_get_config():
    return jsonify(_load_json_file(CONFIG_PATH_DEFAULT)), 200


def _require_confirm_live_if_needed(
    patch: Dict[str, Any], current_effective: Dict[str, Any]
) -> Tuple[bool, str]:
    wants_enable = patch.get("enable_execution", None)
    if wants_enable is None:
        return True, ""

    wants_enable_b = bool(wants_enable)
    if not wants_enable_b:
        return True, ""

    already_enabled = bool(current_effective.get("enable_execution", False))
    if already_enabled:
        return True, ""

    confirm = patch.get("confirm_live", None)
    if confirm != CONFIRM_LIVE_TEXT:
        return (
            False,
            "confirm_live_required: send confirm_live='CONFIRM LIVE' to enable live execution",
        )

    return True, ""


@app.post("/api/config")
def api_set_config():
    patch = request.get_json(silent=True) or {}
    if not isinstance(patch, dict):
        return jsonify({"ok": False, "error": "invalid_json"}), 400

    cfg_eff = _get_effective_config()
    ok_confirm, err = _require_confirm_live_if_needed(patch, cfg_eff)
    if not ok_confirm:
        return jsonify({"ok": False, "error": err}), 400

    patch.pop("confirm_live", None)

    current = _load_json_file(CONFIG_PATH_DEFAULT)
    current.update(patch)

    ok = _save_json_file(CONFIG_PATH_DEFAULT, current)
    return jsonify({"ok": ok}), 200


@app.post("/api/mode/<mode_id>")
def api_set_mode(mode_id: str):
    cfg = _load_json_file(CONFIG_PATH_DEFAULT)
    cfg["mode"] = str(mode_id)
    ok = _save_json_file(CONFIG_PATH_DEFAULT, cfg)
    return jsonify({"ok": ok}), 200


# -----------------------------
# Status
# -----------------------------
@app.get("/api/status")
def api_status():
    cfg_eff = _get_effective_config()

    symbol = cfg_eff.get("symbol", "GOLD")
    enable_exec = bool(cfg_eff.get("enable_execution", False))
    exec_mode = "LIVE" if enable_exec else "DRY_RUN"
    telegram_enabled = bool((cfg_eff.get("telegram") or {}).get("enabled", False))

    mt5_info = _mt5_snapshot(symbol)

    return jsonify(
        {
            "ok": True,
            "ts": time.time(),
            "config": {
                "symbol": symbol,
                "mode": cfg_eff.get("mode"),
                "enable_execution": enable_exec,
                "execution_mode": exec_mode,
                "telegram": {"enabled": telegram_enabled},
            },
            "mt5": mt5_info,
        }
    ), 200


def main() -> int:
    host = os.environ.get("HIM_API_HOST", "127.0.0.1")
    port = int(os.environ.get("HIM_API_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
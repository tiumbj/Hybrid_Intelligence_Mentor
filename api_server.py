"""
api_server.py
Version: 3.1.0

CHANGELOG
- 3.1.0 (2026-02-28)
  - ADD: POST /api/ai_confirm
      * Accept compact payload from ai_bridge_v1.py
      * Call AIMentor (rule-based / mock) to generate execution plan
      * Return AI response in Schema v1.0:
          {schema_version:"1.0", decision:"CONFIRM|REJECT", confidence:0..1, entry, sl, tp, note}
  - KEEP: Existing endpoints
      GET/POST /api/config
      POST /api/mode/<mode_id>
      GET /api/status
      GET /api/signal_preview
      GET /api/performance

NOTES
- This server is local; deterministic enforcement remains in validator_v1_0.
- /api/ai_confirm returns v1.0 schema; ai_bridge_v1 will validate before allowing confirm.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

# project modules (local imports)
from ai_mentor import AIMentor
from config_resolver import resolve_effective_config

app = Flask(__name__)


# -----------------------------
# Helpers
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


def _sf(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _norm_dir(x: Any) -> str:
    s = str(x or "").strip().upper()
    return s if s in ("BUY", "SELL") else "NONE"


def _rr(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    return abs(tp - entry) / risk


# -----------------------------
# Existing endpoints (as seen)
# -----------------------------
CONFIG_PATH_DEFAULT = "config.json"


@app.get("/api/config")
def api_get_config():
    cfg = _load_json_file(CONFIG_PATH_DEFAULT)
    return jsonify(cfg), 200


@app.post("/api/config")
def api_set_config():
    data = request.get_json(silent=True) or {}
    ok = _save_json_file(CONFIG_PATH_DEFAULT, data)
    return jsonify({"ok": ok}), (200 if ok else 500)


@app.post("/api/mode/<mode_id>")
def api_set_mode(mode_id: str):
    cfg = _load_json_file(CONFIG_PATH_DEFAULT)
    cfg["mode"] = str(mode_id)
    ok = _save_json_file(CONFIG_PATH_DEFAULT, cfg)
    return jsonify({"ok": ok, "mode": cfg.get("mode")}), (200 if ok else 500)


@app.get("/api/status")
def api_status():
    cfg_eff = resolve_effective_config(CONFIG_PATH_DEFAULT) or {}
    return jsonify(
        {
            "ok": True,
            "ts": time.time(),
            "mode": cfg_eff.get("mode"),
            "symbol": cfg_eff.get("symbol", "GOLD"),
        }
    ), 200


@app.get("/api/signal_preview")
def api_signal_preview():
    # If you already have a signal preview generator elsewhere, keep it.
    # Here we return a minimal placeholder snapshot to keep endpoint stable.
    cfg_eff = resolve_effective_config(CONFIG_PATH_DEFAULT) or {}
    return jsonify(
        {
            "ok": True,
            "ts": time.time(),
            "symbol": cfg_eff.get("symbol", "GOLD"),
            "mode": cfg_eff.get("mode"),
            "note": "signal preview placeholder (engine snapshot is implemented elsewhere in project)",
        }
    ), 200


@app.get("/api/performance")
def api_performance():
    # placeholder summary endpoint (project already had computed summary)
    return jsonify({"ok": True, "ts": time.time(), "summary": {}}), 200


# -----------------------------
# NEW: AI Confirm Endpoint
# -----------------------------
@app.post("/api/ai_confirm")
def api_ai_confirm():
    """
    Accepts compact payload from ai_bridge_v1.AIConfirmClient.build_compact_payload()
    Returns Schema v1.0 for deterministic validation at boundary.
    """
    payload = request.get_json(silent=True) or {}

    # baseline from payload (compact)
    direction = _norm_dir(payload.get("direction"))
    entry = _sf(payload.get("entry"))
    sl = _sf(payload.get("sl"))
    tp = _sf(payload.get("tp"))
    lot = _sf(payload.get("lot"), 0.0)
    atr = _sf(payload.get("atr"), 0.0)
    mode = str(payload.get("mode") or payload.get("regime") or "")

    # If missing baseline, fail closed
    if direction not in ("BUY", "SELL") or entry == 0.0 or sl == 0.0 or tp == 0.0:
        return jsonify(
            {
                "schema_version": "1.0",
                "decision": "REJECT",
                "confidence": 0.0,
                "note": "missing_baseline_fields(direction/entry/sl/tp)",
            }
        ), 200

    # Map to AIMentor expected package (minimal)
    ai_pkg = {
        "symbol": payload.get("symbol", "GOLD"),
        "baseline": {
            "dir": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": _rr(entry, sl, tp),
            "atr": atr,
            "lot": lot,
            "mode": mode,
        },
        "constraints": {
            # keep minimal; AIMentor internal defaults apply if missing
            "min_rr": 1.5,
            "entry_shift_max_atr": _sf(payload.get("entry_shift_max_atr"), 0.20),
        },
        "context": {
            "watch_state": payload.get("watch_state"),
            "breakout_state": payload.get("breakout_state"),
            "htf_trend": payload.get("htf_trend"),
            "mtf_trend": payload.get("mtf_trend"),
            "ltf_trend": payload.get("ltf_trend"),
            "proximity_score": payload.get("proximity_score"),
        },
    }

    mentor = AIMentor()
    out = mentor.evaluate(ai_pkg) or {}
    ex = out.get("execution") or {}

    # Convert AIMentor output to schema v1.0
    # Expect execution fields: entry, sl, tp, conf (0..100) OR reject flag.
    conf_pct = _sf(ex.get("conf"), 0.0)
    conf01 = max(0.0, min(1.0, conf_pct / 100.0))

    # decision logic (fail closed)
    decision = "CONFIRM"
    if bool(ex.get("reject", False)) or conf01 <= 0.0:
        decision = "REJECT"

    resp = {
        "schema_version": "1.0",
        "decision": decision,
        "confidence": conf01,
        "entry": ex.get("entry", entry),
        "sl": ex.get("sl", sl),
        "tp": ex.get("tp", tp),
        "note": (out.get("mentor") or {}).get("headline") or "AIMentor",
    }
    return jsonify(resp), 200


def main() -> int:
    host = os.environ.get("HIM_API_HOST", "127.0.0.1")
    port = int(os.environ.get("HIM_API_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
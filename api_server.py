# ============================================================
#  api_server.py — Production-grade API Server for HIM
#  Version: v4.0.0
#  Changelog:
#   - v4.0.0 (2026-03-05):
#       * Add missing endpoints:
#           - GET  /api/signal_preview  -> engine.generate_signal_package()
#           - POST /api/ai_confirm      -> confirm-only policy (fail-closed)
#       * Keep existing endpoints: /api/health, /api/status, /api/config (GET/POST)
#       * Thread-safe config loader with reload-on-change
#       * JSONL audit logging to logs/api_server.jsonl
#       * Dashboard redirect support via config.dashboard.external_url
# ============================================================

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable

from flask import Flask, jsonify, request, redirect

# ----------------------------
# Constants
# ----------------------------
APP_VERSION = "v4.0.0"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
DEFAULT_CONFIG_PATH = "config.json"

LOG_DIR = os.path.join(os.getcwd(), "logs")
API_AUDIT_LOG = os.path.join(LOG_DIR, "api_server.jsonl")


# ----------------------------
# Helpers: JSON, Logging
# ----------------------------
def _now_ts() -> int:
    return int(time.time())


def _ensure_log_dir() -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        # do not crash server for logging directory errors
        pass


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    _ensure_log_dir()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        # do not crash server for logging errors
        pass


def _ok(data: Any = None, **meta: Any):
    out = {"ok": True, "ts": _now_ts(), "version": APP_VERSION}
    if data is not None:
        out["data"] = data
    if meta:
        out.update(meta)
    return jsonify(out), 200


def _err(code: str, message: str, http_status: int = 400, **meta: Any):
    out = {
        "ok": False,
        "ts": _now_ts(),
        "version": APP_VERSION,
        "error": {"code": code, "message": message},
    }
    if meta:
        out.update(meta)
    return jsonify(out), http_status


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _upper_str(x: Any) -> str:
    try:
        return str(x).strip().upper()
    except Exception:
        return ""


def _first_present(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return default


# ----------------------------
# Config Loader (thread-safe, reload-on-change)
# ----------------------------
@dataclass
class ConfigState:
    path: str
    mtime: float
    data: Dict[str, Any]


class ConfigManager:
    def __init__(self, config_path: str):
        self._lock = threading.Lock()
        self._state = ConfigState(path=config_path, mtime=0.0, data={})

    def get(self) -> Dict[str, Any]:
        # Reload if file changed
        with self._lock:
            path = self._state.path
            try:
                st = os.stat(path)
                if st.st_mtime > self._state.mtime:
                    self._state = ConfigState(path=path, mtime=st.st_mtime, data=self._load_file(path))
            except FileNotFoundError:
                # keep old config but expose error via status
                if not self._state.data:
                    self._state = ConfigState(path=path, mtime=0.0, data={})
            except Exception:
                # keep old config
                pass
            return dict(self._state.data)

    def set(self, new_data: Dict[str, Any]) -> Tuple[bool, str]:
        with self._lock:
            path = self._state.path
            try:
                # basic validation: must be dict
                if not isinstance(new_data, dict):
                    return False, "config_must_be_object"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)
                st = os.stat(path)
                self._state = ConfigState(path=path, mtime=st.st_mtime, data=new_data)
                return True, "ok"
            except Exception as e:
                return False, f"write_failed: {e}"

    @staticmethod
    def _load_file(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}


# ----------------------------
# Engine Adapter (best-effort import)
# ----------------------------
class EngineAdapter:
    """
    Goal: obtain a callable generate_signal_package() from engine.py
    Supports:
      - from engine import TradingEngine; TradingEngine(cfg).generate_signal_package(...)
      - from engine import generate_signal_package (function)
      - from engine import TradingEngine (instance method) with flexible signature
    """

    def __init__(self, config_mgr: ConfigManager):
        self.config_mgr = config_mgr
        self._lock = threading.Lock()
        self._cached: Optional[Callable[..., Dict[str, Any]]] = None
        self._cached_signature: str = ""

    def generate_signal_package(self) -> Tuple[bool, Dict[str, Any], str]:
        """
        Returns: (ok, payload, reason)
        """
        try:
            fn = self._get_callable()
            cfg = self.config_mgr.get()
            # Decide args: prefer event_timeframe from config if present
            event_tf = None
            try:
                event_tf = (
                    cfg.get("commissioning", {}).get("event_timeframe")
                    or cfg.get("commissioning", {}).get("event_tf")
                    or cfg.get("event_timeframe")
                )
            except Exception:
                event_tf = None

            # Try calling with best-effort signatures:
            # 1) generate_signal_package(event_timeframe="M1")
            # 2) generate_signal_package()
            if event_tf:
                try:
                    pkg = fn(event_timeframe=str(event_tf))
                    return True, self._normalize_signal(pkg), "ok"
                except TypeError:
                    pass

            pkg = fn()
            return True, self._normalize_signal(pkg), "ok"
        except Exception as e:
            return False, {}, f"engine_error: {e}"

    def _get_callable(self) -> Callable[..., Dict[str, Any]]:
        with self._lock:
            if self._cached is not None:
                return self._cached

            # Ensure project root in sys.path
            if os.getcwd() not in sys.path:
                sys.path.insert(0, os.getcwd())

            import importlib

            eng = importlib.import_module("engine")

            # Case A: module-level function
            if hasattr(eng, "generate_signal_package") and callable(getattr(eng, "generate_signal_package")):
                self._cached = getattr(eng, "generate_signal_package")
                self._cached_signature = "engine.generate_signal_package (module function)"
                return self._cached

            # Case B: TradingEngine class
            if hasattr(eng, "TradingEngine"):
                TradingEngine = getattr(eng, "TradingEngine")
                if callable(TradingEngine):
                    cfg = self.config_mgr.get()
                    try:
                        obj = TradingEngine(cfg)
                    except TypeError:
                        # Some engines use TradingEngine(config_path=...) or no-arg
                        try:
                            obj = TradingEngine(config_path=self.config_mgr._state.path)  # type: ignore[attr-defined]
                        except Exception:
                            obj = TradingEngine()

                    if hasattr(obj, "generate_signal_package") and callable(getattr(obj, "generate_signal_package")):
                        self._cached = getattr(obj, "generate_signal_package")
                        self._cached_signature = "TradingEngine(cfg).generate_signal_package"
                        return self._cached

                    # fallback method names
                    for name in ("eval_signal", "generate_signal", "run_once", "analyze"):
                        if hasattr(obj, name) and callable(getattr(obj, name)):
                            self._cached = getattr(obj, name)
                            self._cached_signature = f"TradingEngine(cfg).{name}"
                            return self._cached

            raise RuntimeError("engine_callable_not_found")

    @staticmethod
    def _normalize_signal(raw: Any) -> Dict[str, Any]:
        """
        Normalize to mentor_executor expected fields:
          request_id, decision, plan{entry,sl,tp}, confidence?, metrics?, context?, source?
        """
        if not isinstance(raw, dict):
            return {"request_id": f"RAW-{_now_ts()}", "decision": "HOLD", "plan": {"entry": 0.0, "sl": 0.0, "tp": 0.0}, "raw": raw}

        decision = _upper_str(_first_present(raw, ("decision", "action", "signal", "side"), "HOLD"))
        if decision not in ("BUY", "SELL", "HOLD", "NONE"):
            # map common variants
            if decision in ("LONG", "BULL", "UP"):
                decision = "BUY"
            elif decision in ("SHORT", "BEAR", "DOWN"):
                decision = "SELL"
            else:
                decision = "HOLD"

        plan = raw.get("plan", None)
        if not isinstance(plan, dict):
            # some engines nest under dry_run_order/order/signal/result/payload
            for k in ("dry_run_order", "order", "result", "payload", "signal"):
                v = raw.get(k)
                if isinstance(v, dict) and isinstance(v.get("plan"), dict):
                    plan = v.get("plan")
                    break
            if not isinstance(plan, dict):
                plan = {}

        entry = _safe_float(_first_present(plan, ("entry", "price", "open"), 0.0)) or 0.0
        sl = _safe_float(_first_present(plan, ("sl", "stop_loss"), 0.0)) or 0.0
        tp = _safe_float(_first_present(plan, ("tp", "take_profit"), 0.0)) or 0.0

        request_id = raw.get("request_id") or raw.get("id") or f"REQ-{_now_ts()}"
        request_id = str(request_id)[:80]

        out: Dict[str, Any] = {
            "request_id": request_id,
            "decision": "HOLD" if decision == "NONE" else decision,
            "plan": {"entry": entry, "sl": sl, "tp": tp},
        }

        # Optional fields (keep if present)
        for k in ("confidence", "metrics", "blocked_by", "context", "source", "symbol", "timeframe", "ts"):
            if k in raw:
                out[k] = raw.get(k)

        return out


# ----------------------------
# AI Confirm (confirm-only, pluggable)
# ----------------------------
class AIConfirmer:
    """
    confirm-only policy:
      - Can proxy to external AI confirm URL if configured.
      - Otherwise uses local fail-closed rules:
          approve only if:
            decision in BUY/SELL
            plan has positive sl/tp
            rr >= min_rr (if available)
            confidence >= min_conf (if available)
    """

    def __init__(self, config_mgr: ConfigManager):
        self.config_mgr = config_mgr

    def confirm(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config_mgr.get()
        ai_cfg = cfg.get("ai_confirm", {}) if isinstance(cfg.get("ai_confirm", {}), dict) else {}

        # 1) External proxy (optional)
        external_url = ai_cfg.get("external_url") or ai_cfg.get("api_url")
        if isinstance(external_url, str) and external_url.strip():
            ok, out = self._proxy_confirm(external_url.strip(), payload, timeout_sec=float(ai_cfg.get("timeout_sec", 8.0)))
            if ok:
                return self._sanitize_ai_response(out)
            return {"approved": False, "reason": "ai_proxy_failed", "confidence": None}

        # 2) Local confirm policy (fail-closed)
        decision = _upper_str(payload.get("decision"))
        plan = payload.get("plan", {}) if isinstance(payload.get("plan", {}), dict) else {}
        entry = _safe_float(plan.get("entry"))
        sl = _safe_float(plan.get("sl"))
        tp = _safe_float(plan.get("tp"))
        conf = _safe_float(payload.get("confidence"))
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), dict) else {}
        rr = _safe_float(metrics.get("rr"))

        # thresholds (safe defaults)
        min_conf = _safe_float(ai_cfg.get("min_confidence"))  # None allowed
        min_rr = _safe_float(ai_cfg.get("min_rr"))            # None allowed

        reasons = []

        if decision not in ("BUY", "SELL"):
            reasons.append("decision_not_trade")
        if sl is None or tp is None or (sl <= 0) or (tp <= 0):
            reasons.append("plan_sl_tp_invalid")
        if entry is not None and sl is not None and tp is not None:
            if decision == "BUY" and not (sl < entry < tp):
                reasons.append("invalid_stops_side_buy")
            if decision == "SELL" and not (tp < entry < sl):
                reasons.append("invalid_stops_side_sell")

        if min_rr is not None:
            if rr is None or rr < min_rr:
                reasons.append("rr_below_min")

        if min_conf is not None:
            if conf is None or conf < min_conf:
                reasons.append("confidence_below_min")

        if reasons:
            return {"approved": False, "reason": ";".join(reasons)[:500], "confidence": conf}

        return {"approved": True, "reason": "approved_by_local_policy", "confidence": conf}

    @staticmethod
    def _sanitize_ai_response(raw: Any) -> Dict[str, Any]:
        """
        Normalize external response into:
          {approved: bool, reason: str, confidence: float|None}
        """
        if isinstance(raw, dict):
            approved = raw.get("approved", None)
            reason = raw.get("reason", raw.get("message", ""))
            confidence = raw.get("confidence", None)
            return {
                "approved": bool(approved is True),
                "reason": str(reason)[:500],
                "confidence": _safe_float(confidence),
            }
        return {"approved": False, "reason": "ai_response_invalid", "confidence": None}

    @staticmethod
    def _proxy_confirm(url: str, payload: Dict[str, Any], timeout_sec: float) -> Tuple[bool, Any]:
        """
        Best-effort HTTP POST JSON without extra deps.
        Uses urllib to avoid requests dependency assumptions.
        """
        try:
            import urllib.request

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url=url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            try:
                return True, json.loads(body)
            except Exception:
                return True, {"approved": False, "reason": "ai_proxy_non_json", "raw": body[:800]}
        except Exception:
            return False, None


# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)

_config_path = os.environ.get("HIM_CONFIG_PATH", DEFAULT_CONFIG_PATH).strip() or DEFAULT_CONFIG_PATH
_host = os.environ.get("HIM_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
_port = int(os.environ.get("HIM_PORT", str(DEFAULT_PORT)))

config_mgr = ConfigManager(_config_path)
engine_adapter = EngineAdapter(config_mgr)
ai_confirmer = AIConfirmer(config_mgr)


def _audit(event: str, **fields: Any) -> None:
    _append_jsonl(API_AUDIT_LOG, {"ts": _now_ts(), "event": event, "version": APP_VERSION, **fields})


@app.route("/", methods=["GET"])
def root():
    cfg = config_mgr.get()
    # If dashboard external url provided, redirect root too (optional behavior)
    ext = None
    try:
        ext = cfg.get("dashboard", {}).get("external_url")
    except Exception:
        ext = None
    if isinstance(ext, str) and ext.strip():
        return redirect(ext.strip(), code=302)
    return _ok(
        {
            "service": "HIM API Server",
            "version": APP_VERSION,
            "endpoints": [
                "GET /api/health",
                "GET /api/status",
                "GET /api/config",
                "POST /api/config",
                "GET /api/signal_preview",
                "POST /api/ai_confirm",
            ],
        }
    )


@app.route("/dashboard", methods=["GET"])
def dashboard():
    cfg = config_mgr.get()
    ext = None
    try:
        ext = cfg.get("dashboard", {}).get("external_url")
    except Exception:
        ext = None
    if isinstance(ext, str) and ext.strip():
        return redirect(ext.strip(), code=302)
    # local dashboard removed (per v3 note): return informative JSON
    return _err(
        code="dashboard_removed",
        message="Local dashboard removed; set config.dashboard.external_url to redirect.",
        http_status=404,
    )


@app.route("/api/health", methods=["GET"])
def api_health():
    # minimal health check
    _audit("api_health")
    return _ok({"status": "healthy", "host": _host, "port": _port, "config_path": _config_path})


@app.route("/api/status", methods=["GET"])
def api_status():
    cfg = config_mgr.get()
    # Try quick engine callable detection without generating signal
    engine_ok = True
    engine_sig = ""
    try:
        fn = engine_adapter._get_callable()  # noqa
        engine_sig = getattr(engine_adapter, "_cached_signature", "")  # noqa
        _ = fn
    except Exception as e:
        engine_ok = False
        engine_sig = f"engine_unavailable: {e}"

    out = {
        "service": "HIM API Server",
        "host": _host,
        "port": _port,
        "config_path": _config_path,
        "engine": {"ok": engine_ok, "binding": engine_sig},
        "ai_confirm": {
            "mode": "proxy" if (isinstance(cfg.get("ai_confirm", {}), dict) and (cfg.get("ai_confirm", {}).get("external_url") or cfg.get("ai_confirm", {}).get("api_url"))) else "local_policy",
        },
    }
    _audit("api_status", engine_ok=engine_ok)
    return _ok(out)


@app.route("/api/config", methods=["GET"])
def api_get_config():
    cfg = config_mgr.get()
    _audit("api_get_config")
    return _ok(cfg)


@app.route("/api/config", methods=["POST", "OPTIONS"])
def api_set_config():
    if request.method == "OPTIONS":
        return "", 204

    try:
        d = request.get_json(force=True, silent=False)
    except Exception:
        return _err("invalid_json", "Body must be valid JSON object", 400)

    ok, reason = config_mgr.set(d if isinstance(d, dict) else {})
    _audit("api_set_config", ok=ok, reason=reason)
    if not ok:
        return _err("config_write_failed", reason, 500)
    return _ok({"status": "saved", "config_path": _config_path})


@app.route("/api/signal_preview", methods=["GET"])
def api_signal_preview():
    t0 = time.time()
    ok, sig, reason = engine_adapter.generate_signal_package()

    latency_ms = int((time.time() - t0) * 1000)
    _audit(
        "api_signal_preview",
        ok=ok,
        reason=reason,
        latency_ms=latency_ms,
        request_id=sig.get("request_id") if isinstance(sig, dict) else None,
    )

    if not ok:
        return _err("engine_failed", reason, 500, latency_ms=latency_ms)

    # ensure minimal schema exists
    if not isinstance(sig, dict) or "request_id" not in sig or "decision" not in sig or "plan" not in sig:
        return _err("signal_schema_invalid", "engine output missing required fields", 500, raw_type=str(type(sig)))

    # Return "signal object" directly (mentor_executor expects raw signal JSON)
    sig["api_meta"] = {"latency_ms": latency_ms, "version": APP_VERSION}
    return jsonify(sig), 200


@app.route("/api/ai_confirm", methods=["POST", "OPTIONS"])
def api_ai_confirm():
    if request.method == "OPTIONS":
        return "", 204

    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return _err("invalid_json", "Body must be valid JSON object", 400)

    if not isinstance(payload, dict):
        return _err("invalid_payload", "Payload must be JSON object", 400)

    request_id = str(payload.get("request_id") or payload.get("id") or "")[:80]
    if not request_id:
        # fail-closed
        _audit("api_ai_confirm", ok=False, reason="request_id_missing")
        return jsonify({"approved": False, "reason": "request_id_missing", "confidence": None}), 200

    t0 = time.time()
    ai = ai_confirmer.confirm(payload)
    latency_ms = int((time.time() - t0) * 1000)

    # enforce confirm-only schema
    out = {
        "approved": bool(ai.get("approved") is True),
        "reason": str(ai.get("reason", ""))[:500],
        "confidence": _safe_float(ai.get("confidence")),
        "request_id": request_id,
        "api_meta": {"latency_ms": latency_ms, "version": APP_VERSION},
    }

    _audit(
        "api_ai_confirm",
        ok=True,
        approved=out["approved"],
        latency_ms=latency_ms,
        request_id=request_id,
        reason=out["reason"][:120],
    )

    return jsonify(out), 200


# ----------------------------
# Error handlers (consistent JSON)
# ----------------------------
@app.errorhandler(404)
def not_found(e):
    _audit("http_404", path=request.path)
    return _err("not_found", f"Route not found: {request.path}", 404)


@app.errorhandler(500)
def internal_error(e):
    _audit("http_500", path=request.path)
    return _err("internal_error", "Internal server error", 500)


def _startup_log():
    cfg = config_mgr.get()
    note = "Local dashboard removed; redirect to config.dashboard.external_url"
    _append_jsonl(
        API_AUDIT_LOG,
        {
            "ts": _now_ts(),
            "msg": "api_server_start",
            "version": APP_VERSION,
            "host": _host,
            "port": _port,
            "config_path": _config_path,
            "note": note,
        },
    )


if __name__ == "__main__":
    _startup_log()
    # Flask dev server (local). For true production WSGI, serve app via gunicorn/waitress.
    app.run(host=_host, port=_port, debug=False, threaded=True)
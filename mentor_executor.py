"""
Mentor Executor - Execution Controller (Production Hardening Phase)
Version: 2.4.3

Changelog:
- 2.4.3 (2026-02-26):
  - FIX: BOS source of truth MUST come from pkg["bos"] (never context["bos"])
  - FIX: supertrend_ok source of truth MUST come from pkg["supertrend_ok"] (never context["supertrend_ok"])
  - FIX: NONE_REASON["bos"] and NONE_REASON["supertrend_ok"] are always boolean (never None)
  - ENFORCE: contract fields bos/supertrend_ok MUST exist and MUST NOT be None (fail-fast)
  - RESPECT: ai.enabled (if false -> Technical Gate only; if true -> AI confirm-only)
  - TELEGRAM: send ONLY after final approval (no reject/NONE spam)
  - SAFETY: keep production-safe guardrails (spread, stops/freeze check, dedupe, open position guard)

Locked Rules (Do not change):
- BOS required: if bos=False -> must be blocked (no_bos)
- Engine = owner of decision data (contract fields live at top-level)
- Executor MUST NOT use debug context as primary source for BOS/supertrend_ok
- Telegram sends approved only
- direction="NONE" => no trade

Notes:
- This file is a full replacement (No patch).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import MetaTrader5 as mt5

from ai_mentor import AIMentor
from config_resolver import resolve_effective_config
from engine import TradingEngine
from telegram_notifier import TelegramNotifier
from trade_logger import TradeLogger


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("him_system.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("HIM")


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _floor_to_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return math.floor(x / step) * step


def _require_bool_field(pkg: Dict[str, Any], key: str) -> bool:
    """
    Contract enforcement:
    - key MUST exist
    - value MUST NOT be None
    - returns bool(value)
    """
    if key not in pkg:
        raise RuntimeError(f"CONTRACT_VIOLATION: missing top-level field: {key}")
    v = pkg.get(key)
    if v is None:
        raise RuntimeError(f"CONTRACT_VIOLATION: top-level field '{key}' is None")
    return bool(v)


class HotConfig:
    def __init__(self, path: str):
        self.path = path
        self._mtime = 0.0
        self._cache: Dict[str, Any] = {}

    def load_raw(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.path):
                return self._cache or {}
            m = os.path.getmtime(self.path)
            if m <= self._mtime and self._cache:
                return self._cache
            with open(self.path, "r", encoding="utf-8") as f:
                self._cache = json.load(f) or {}
            self._mtime = m
            return self._cache
        except Exception as e:
            logger.error(f"CRITICAL: config.json load failed: {e}")
            return self._cache or {}

    def load_effective(self) -> Dict[str, Any]:
        raw = self.load_raw()
        try:
            return resolve_effective_config(raw)
        except Exception as e:
            logger.error(f"CRITICAL: resolve_effective_config failed: {e}")
            return raw or {}


class MentorExecutor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(base_dir, "config.json")

        self.hcfg = HotConfig(self.config_path)
        self.engine = TradingEngine(config_path=self.config_path)
        self.ai = AIMentor()
        self.tg = TelegramNotifier(config_path=self.config_path)
        self.tlog = TradeLogger(os.path.join(base_dir, "trade_history.json"))

        # Dedupe/cooldown (Telegram + Execution)
        self._last_tg_sig: Optional[str] = None
        self._last_tg_ts: float = 0.0
        self._last_exec_sig: Optional[str] = None
        self._last_exec_ts: float = 0.0

        self._ensure_mt5()

    # -----------------------------
    # MT5 helpers
    # -----------------------------
    def _ensure_mt5(self) -> None:
        if mt5.terminal_info() is not None:
            return
        if not mt5.initialize():
            err = mt5.last_error()
            raise RuntimeError(f"MT5 initialize failed: {err}")
        logger.info("MT5 Connected")

    @staticmethod
    def _get_tick(symbol: str):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Cannot get tick for symbol: {symbol} last_error={mt5.last_error()}")
        return tick

    @staticmethod
    def _get_info(symbol: str):
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol not found: {symbol}")
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Cannot select symbol: {symbol} last_error={mt5.last_error()}")
        return info

    @staticmethod
    def _positions_exist(symbol: str) -> bool:
        pos = mt5.positions_get(symbol=symbol)
        # Conservative: if MT5 cannot return positions, block to avoid duplicate risk
        if pos is None:
            return True
        return bool(len(pos) > 0)

    # -----------------------------
    # Config snapshot (audit visibility)
    # -----------------------------
    @staticmethod
    def _cfg_snapshot(cfg_eff: Dict[str, Any]) -> Dict[str, Any]:
        snap = {
            "mode": cfg_eff.get("mode"),
            "symbol": cfg_eff.get("symbol"),
            "enable_execution": cfg_eff.get("enable_execution"),
            "confidence_threshold": cfg_eff.get("confidence_threshold"),
            "min_score": cfg_eff.get("min_score"),
            "min_rr": cfg_eff.get("min_rr"),
            "lot": cfg_eff.get("lot"),
            "timeframes": cfg_eff.get("timeframes"),
            "risk": cfg_eff.get("risk"),
            "supertrend": cfg_eff.get("supertrend"),
        }
        for k in ("ai", "execution", "execution_safety", "telegram"):
            if k in cfg_eff:
                snap[k] = cfg_eff.get(k)
        return snap

    # -----------------------------
    # Safety layer
    # -----------------------------
    def _spread_points(self, symbol: str) -> float:
        info = self._get_info(symbol)
        tick = self._get_tick(symbol)
        if info.point <= 0:
            return float("inf")
        return float((tick.ask - tick.bid) / info.point)

    def _validate_stops_levels(
        self, symbol: str, direction: str, entry: float, sl: float, tp: float
    ) -> Tuple[bool, str]:
        info = self._get_info(symbol)
        point = float(info.point) if info.point else 0.0
        if point <= 0:
            return False, "invalid_symbol_point"

        stops_level_points = int(getattr(info, "trade_stops_level", 0) or 0)
        freeze_level_points = int(getattr(info, "trade_freeze_level", 0) or 0)
        min_dist = float(stops_level_points) * point

        if direction == "BUY":
            if not (sl < entry < tp):
                return False, "sanity_failed_buy"
            if (entry - sl) < min_dist:
                return False, f"stop_level_failed_sl dist={entry - sl:.5f} min={min_dist:.5f}"
            if (tp - entry) < min_dist:
                return False, f"stop_level_failed_tp dist={tp - entry:.5f} min={min_dist:.5f}"
        elif direction == "SELL":
            if not (tp < entry < sl):
                return False, "sanity_failed_sell"
            if (sl - entry) < min_dist:
                return False, f"stop_level_failed_sl dist={sl - entry:.5f} min={min_dist:.5f}"
            if (entry - tp) < min_dist:
                return False, f"stop_level_failed_tp dist={entry - tp:.5f} min={min_dist:.5f}"
        else:
            return False, "invalid_direction"

        # Freeze-level: conservative check
        if freeze_level_points > 0:
            tick = self._get_tick(symbol)
            freeze_dist = float(freeze_level_points) * point
            cur = float(tick.ask if direction == "BUY" else tick.bid)
            if abs(cur - entry) <= freeze_dist:
                return False, f"freeze_level_too_close cur-entry={abs(cur-entry):.5f} freeze={freeze_dist:.5f}"

        return True, "ok"

    # -----------------------------
    # Risk sizing (compat keys)
    # -----------------------------
    @staticmethod
    def _get_risk_pct(cfg_eff: Dict[str, Any]) -> float:
        """
        Supported:
        - risk.risk_pct: fraction (0.01 = 1%)
        - risk.risk_percent: percent (1.0 = 1%)
        Priority: risk_pct if exists else risk_percent
        Clamp: 0–5%
        """
        risk_cfg = (cfg_eff.get("risk") or {}) if isinstance(cfg_eff, dict) else {}
        if "risk_pct" in risk_cfg:
            v = _safe_float(risk_cfg.get("risk_pct"), 0.01)
            risk_pct = v
        elif "risk_percent" in risk_cfg:
            v = _safe_float(risk_cfg.get("risk_percent"), 1.0)
            risk_pct = v / 100.0
        else:
            risk_pct = 0.01
        return _clamp(float(risk_pct), 0.0, 0.05)

    def _calc_lot(
        self,
        cfg_eff: Dict[str, Any],
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        fallback_lot: float,
    ) -> Tuple[float, str]:
        try:
            acct = mt5.account_info()
            if acct is None:
                return float(fallback_lot), "account_info_none_fallback"

            equity = float(getattr(acct, "equity", 0.0) or 0.0)
            if equity <= 0:
                return float(fallback_lot), "equity_invalid_fallback"

            risk_pct = self._get_risk_pct(cfg_eff)
            risk_amount = equity * risk_pct
            if risk_amount <= 0:
                return float(fallback_lot), "risk_amount_invalid_fallback"

            info = self._get_info(symbol)
            vol_min = float(getattr(info, "volume_min", 0.01) or 0.01)
            vol_max = float(getattr(info, "volume_max", 100.0) or 100.0)
            vol_step = float(getattr(info, "volume_step", 0.01) or 0.01)

            order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
            loss_1lot = mt5.order_calc_profit(order_type, symbol, 1.0, float(entry), float(sl))
            if loss_1lot is None:
                return float(fallback_lot), "order_calc_profit_none_fallback"

            risk_per_lot = abs(float(loss_1lot))
            if risk_per_lot <= 0:
                return float(fallback_lot), "risk_per_lot_invalid_fallback"

            raw_lot = risk_amount / risk_per_lot
            lot = _floor_to_step(raw_lot, vol_step)  # floor only (never exceed risk)
            lot = _clamp(lot, vol_min, vol_max)

            return float(lot), (
                f"risk equity={equity:.2f} risk_pct={risk_pct:.4f} risk={risk_amount:.2f} "
                f"risk_per_1lot={risk_per_lot:.2f}"
            )
        except Exception as e:
            return float(fallback_lot), f"risk_calc_exception_fallback: {e}"

    # -----------------------------
    # Telegram approved-only
    # -----------------------------
    def _telegram_dedupe_ok(self, cfg_eff: Dict[str, Any], sig: str) -> bool:
        tg_cfg = (cfg_eff.get("telegram") or {}) if isinstance(cfg_eff, dict) else {}
        cooldown_sec = _safe_int(tg_cfg.get("cooldown_sec", 900), 900)
        now = time.time()
        if self._last_tg_sig == sig and (now - self._last_tg_ts) < cooldown_sec:
            return False
        self._last_tg_sig = sig
        self._last_tg_ts = now
        return True

    def _safe_send_telegram_trade(self, cfg_eff: Dict[str, Any], text: str, sig: str) -> None:
        try:
            if not self._telegram_dedupe_ok(cfg_eff, sig):
                logger.info("Telegram: dedupe/cooldown suppress.")
                return
            ok = self.tg.send_text(text, event_type="trade")
            if not ok:
                logger.warning("Telegram send failed or disabled.")
        except Exception:
            logger.error("CRITICAL: telegram send failed (isolated).")
            logger.error(traceback.format_exc())

    # -----------------------------
    # Execution dedupe
    # -----------------------------
    def _execution_dedupe_ok(self, cfg_eff: Dict[str, Any], sig: str) -> bool:
        exec_cfg = (cfg_eff.get("execution") or {}) if isinstance(cfg_eff, dict) else {}
        cooldown_sec = _safe_int(exec_cfg.get("cooldown_sec", 120), 120)
        now = time.time()
        if self._last_exec_sig == sig and (now - self._last_exec_ts) < cooldown_sec:
            return False
        self._last_exec_sig = sig
        self._last_exec_ts = now
        return True

    # -----------------------------
    # NONE reason tracing (TOP-LEVEL CONTRACT)
    # -----------------------------
    @staticmethod
    def _trace_none_reason(pkg: Dict[str, Any]) -> Dict[str, Any]:
        ctx = pkg.get("context", {}) or {}

        # IMPORTANT: MUST come from TOP-LEVEL (and enforced)
        bos = _require_bool_field(pkg, "bos")
        st_ok = _require_bool_field(pkg, "supertrend_ok")

        return {
            "blocked_by": ctx.get("blocked_by"),
            "watch_state": ctx.get("watch_state"),
            "breakout_state": ctx.get("breakout_state"),
            "breakout_side": ctx.get("breakout_side"),
            "bos": bos,  # boolean always
            "retest_ok": bool(ctx.get("retest_ok", False)),
            "proximity_score": ctx.get("proximity_score"),
            "proximity_side": ctx.get("proximity_side"),
            "proximity_best_dist": ctx.get("proximity_best_dist"),
            "threshold_points": ctx.get("breakout_proximity_threshold_points"),
            "supertrend_dir": ctx.get("supertrend_dir"),
            "supertrend_value": ctx.get("supertrend_value"),
            "supertrend_ok": st_ok,  # boolean always
            "HTF_trend": ctx.get("HTF_trend"),
            "MTF_trend": ctx.get("MTF_trend"),
            "LTF_trend": ctx.get("LTF_trend"),
            "bias_source": ctx.get("bias_source"),
        }

    # -----------------------------
    # Decision: Technical Gate (AI disabled)
    # -----------------------------
    @staticmethod
    def _sanity(direction: str, entry: float, sl: float, tp: float) -> bool:
        if direction == "BUY":
            return sl < entry < tp
        if direction == "SELL":
            return tp < entry < sl
        return False

    def _technical_gate_decision(self, cfg_eff: Dict[str, Any], pkg: Dict[str, Any]) -> Dict[str, Any]:
        direction = str(pkg.get("direction", "NONE"))
        score = _safe_float(pkg.get("score", 0.0), 0.0)
        rr = _safe_float(pkg.get("rr", 0.0), 0.0)
        entry = _safe_float(pkg.get("entry_candidate"), 0.0)
        sl = _safe_float(pkg.get("stop_candidate"), 0.0)
        tp = _safe_float(pkg.get("tp_candidate"), 0.0)

        # IMPORTANT: MUST come from TOP-LEVEL (and enforced)
        bos = _require_bool_field(pkg, "bos")
        st_ok = _require_bool_field(pkg, "supertrend_ok")

        min_score = _safe_float(cfg_eff.get("min_score", 7.0), 7.0)
        min_rr = _safe_float(cfg_eff.get("min_rr", 2.0), 2.0)
        conf_th = int(cfg_eff.get("confidence_threshold", 75))
        conf_py = int(pkg.get("confidence_py", 0))

        approved = (
            direction in ("BUY", "SELL")
            and bos
            and st_ok
            and score >= min_score
            and rr >= min_rr
            and conf_py >= conf_th
            and self._sanity(direction, entry, sl, tp)
        )

        reasoning = [
            "AI disabled -> Technical Gate",
            f"BOS={bos}, SuperTrendOK={st_ok}",
            f"Score={score:.1f} (min={min_score:.1f}), RR={rr:.2f} (min={min_rr:.2f})",
            f"confidence_py={conf_py} (th={conf_th})",
        ]

        warnings = []
        if direction not in ("BUY", "SELL"):
            warnings.append("Direction is NONE/invalid")
        if not bos:
            warnings.append("BOS required but false")
        if not st_ok:
            warnings.append("SuperTrend conflict")
        if score < min_score:
            warnings.append("Score below min")
        if rr < min_rr:
            warnings.append("RR below min")
        if conf_py < conf_th:
            warnings.append("confidence_py below threshold")
        if not self._sanity(direction, entry, sl, tp):
            warnings.append("price sanity failed")

        return {
            "approved": bool(approved),
            "confidence": int(conf_py),
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "reasoning": "\n".join(reasoning),
            "warnings": "\n".join(warnings),
        }

    # -----------------------------
    # Decision: AI confirm-only (ai.enabled=true)
    # -----------------------------
    def _ai_confirm_decision(self, pkg: Dict[str, Any], fallback_decision: Dict[str, Any], cfg_eff: Dict[str, Any]) -> Dict[str, Any]:
        """
        - ai.enabled=true => call AIMentor.approve_trade()
        - If AI response invalid and ai.fallback_to_technical=true => fallback_decision
        - IMPORTANT: AI is confirm-only; contract fields remain top-level.
          If AI module still reads context keys, we normalize a copy for AI only.
        """
        ai_cfg = (cfg_eff.get("ai") or {}) if isinstance(cfg_eff, dict) else {}
        fallback_to_technical = bool(ai_cfg.get("fallback_to_technical", True))

        try:
            # enforce contract first
            bos = _require_bool_field(pkg, "bos")
            st_ok = _require_bool_field(pkg, "supertrend_ok")

            normalized = dict(pkg)
            ctx = dict((pkg.get("context", {}) or {}))
            # normalize for AI compatibility (AI confirm-only)
            ctx["bos"] = bos
            ctx["supertrend_ok"] = st_ok
            normalized["context"] = ctx

            ai_resp = self.ai.approve_trade(normalized)
            if not AIMentor.validate_response(ai_resp):
                raise ValueError("AI response failed strict validation")

            return ai_resp
        except Exception as e:
            logger.error(f"AI confirm failed: {e}")
            logger.error(traceback.format_exc())
            if fallback_to_technical:
                return fallback_decision

            rej = dict(fallback_decision)
            rej["approved"] = False
            rej["warnings"] = (rej.get("warnings", "") + "\nAI confirm failed and fallback disabled").strip()
            return rej

    # -----------------------------
    # Order execution
    # -----------------------------
    def _place_market_order(self, symbol: str, direction: str, lot: float, sl: float, tp: float) -> Dict[str, Any]:
        tick = self._get_tick(symbol)
        price = float(tick.ask if direction == "BUY" else tick.bid)

        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 30,
            "magic": 20260226,
            "comment": "HIM v2.4.3",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"ok": False, "error": "order_send returned None", "last_error": mt5.last_error()}

        ok = (result.retcode == mt5.TRADE_RETCODE_DONE)
        return {
            "ok": bool(ok),
            "retcode": int(result.retcode),
            "order": int(getattr(result, "order", 0)),
            "deal": int(getattr(result, "deal", 0)),
            "price": float(price),
            "comment": str(getattr(result, "comment", "")),
            "request": {
                "symbol": symbol,
                "direction": direction,
                "volume": float(lot),
                "sl": float(sl),
                "tp": float(tp),
            },
        }

    # -----------------------------
    # Telegram message (approved only)
    # -----------------------------
    @staticmethod
    def _build_mentor_message(
        symbol: str,
        direction: str,
        pkg: Dict[str, Any],
        decision: Dict[str, Any],
        enable_execution: bool,
        lot: float,
    ) -> str:
        ctx = pkg.get("context", {}) or {}
        score = _safe_float(pkg.get("score", 0.0), 0.0)
        rr = _safe_float(pkg.get("rr", 0.0), 0.0)

        bos = _require_bool_field(pkg, "bos")
        st_ok = _require_bool_field(pkg, "supertrend_ok")

        confidence = int(decision.get("confidence", 0))
        entry = _safe_float(decision.get("entry", 0.0), 0.0)
        sl = _safe_float(decision.get("sl", 0.0), 0.0)
        tp = _safe_float(decision.get("tp", 0.0), 0.0)

        st_dir = str(ctx.get("supertrend_dir", "NA"))
        st_val = _safe_float(ctx.get("supertrend_value", 0.0), 0.0)

        status = "APPROVED"
        if not enable_execution:
            status += " (DRY-RUN)"

        msg = []
        msg.append(f"HIM Signal — {status}")
        msg.append(f"Symbol: {symbol}")
        msg.append(f"Direction: {direction}")
        msg.append(f"Lot: {lot:.2f}")
        msg.append(f"Confidence: {confidence}%")
        msg.append(f"Entry: {entry:.2f}")
        msg.append(f"SL: {sl:.2f}")
        msg.append(f"TP: {tp:.2f}")
        msg.append("")
        msg.append("Mentor Explanation")
        msg.append(
            f"1) HTF={ctx.get('HTF_trend')} MTF={ctx.get('MTF_trend')} LTF={ctx.get('LTF_trend')} (bias={ctx.get('bias_source')})"
        )
        msg.append(
            f"2) BOS={bos} Retest={bool(ctx.get('retest_ok', False))} Breakout={ctx.get('breakout_state')}({ctx.get('breakout_side')})"
        )
        msg.append(f"3) SuperTrend dir={st_dir} value={st_val:.2f} ok={st_ok}")
        msg.append(f"4) Score={score:.1f}/10 RR={rr:.2f} blocked_by={ctx.get('blocked_by')}")
        return "\n".join(msg)

    # -----------------------------
    # Main loop
    # -----------------------------
    def run_once(self) -> None:
        cfg_eff = self.hcfg.load_effective()
        logger.info(f"CFG_EFFECTIVE: {self._cfg_snapshot(cfg_eff)}")

        symbol = str(cfg_eff.get("symbol", "GOLD"))
        enable_execution = bool(cfg_eff.get("enable_execution", False))

        ai_cfg = (cfg_eff.get("ai") or {}) if isinstance(cfg_eff, dict) else {}
        ai_enabled = bool(ai_cfg.get("enabled", False))

        exec_cfg = (cfg_eff.get("execution") or {}) if isinstance(cfg_eff, dict) else {}
        max_spread_points = _safe_float(exec_cfg.get("max_spread_points", 60), 60)
        block_if_position_exists = bool(exec_cfg.get("block_if_position_exists", True))

        # 1) Engine package
        try:
            pkg = self.engine.generate_signal_package()
        except Exception as e:
            logger.error(f"CRITICAL: engine failed: {e}")
            logger.error(traceback.format_exc())
            self.tlog.append(
                {"type": "engine_error", "symbol": symbol, "error": str(e), "traceback": traceback.format_exc()}
            )
            return

        # Contract enforcement early (prevents bos=None in logs)
        try:
            bos = _require_bool_field(pkg, "bos")
            st_ok = _require_bool_field(pkg, "supertrend_ok")
        except Exception as e:
            logger.error(str(e))
            self.tlog.append({"type": "contract_violation", "symbol": symbol, "error": str(e), "package": pkg})
            raise

        direction = str(pkg.get("direction", "NONE"))
        score = _safe_float(pkg.get("score", 0.0), 0.0)
        rr = _safe_float(pkg.get("rr", 0.0), 0.0)
        confidence_py = _safe_int(pkg.get("confidence_py", 0), 0)

        logger.info(
            f"SIGNAL: {direction} score={score:.1f} rr={rr:.2f} confidence_py={confidence_py} bos={bos} supertrend_ok={st_ok}"
        )
        self.tlog.append({"type": "signal", "symbol": symbol, "package": pkg})

        # 2) If NONE, log reasons (approved-only telegram rule: do NOT send)
        if direction not in ("BUY", "SELL"):
            none_reason = self._trace_none_reason(pkg)
            logger.info(f"NONE_REASON: {none_reason}")
            self.tlog.append({"type": "none_reason", "symbol": symbol, "detail": none_reason})
            logger.info("No clear bias. Skip.")
            return

        # 3) Decide: Technical gate baseline
        technical_decision = self._technical_gate_decision(cfg_eff, pkg)

        # 4) Decide: AI confirm-only if enabled
        if ai_enabled:
            decision = self._ai_confirm_decision(pkg=pkg, fallback_decision=technical_decision, cfg_eff=cfg_eff)
            decision_mode = "AI_CONFIRM"
        else:
            decision = technical_decision
            decision_mode = "TECHNICAL_GATE"

        approved = bool(decision.get("approved", False))
        logger.info(f"DECISION({decision_mode}): approved={approved} conf={decision.get('confidence')}")
        self.tlog.append({"type": "decision", "symbol": symbol, "mode": decision_mode, "decision": decision})

        if not approved:
            # Telegram: approved-only => do not send
            return

        # 5) Execution safety hardening
        entry = _safe_float(decision.get("entry"), _safe_float(pkg.get("entry_candidate"), 0.0))
        sl = _safe_float(decision.get("sl"), _safe_float(pkg.get("stop_candidate"), 0.0))
        tp = _safe_float(decision.get("tp"), _safe_float(pkg.get("tp_candidate"), 0.0))

        # Spread filter
        spread_pts = self._spread_points(symbol)
        if spread_pts > max_spread_points:
            logger.warning(f"BLOCK: spread too high {spread_pts:.1f} > {max_spread_points:.1f}")
            self.tlog.append({"type": "blocked", "symbol": symbol, "reason": "spread", "spread_points": spread_pts})
            return

        # Duplicate protection: open position guard
        if block_if_position_exists and self._positions_exist(symbol):
            logger.warning("BLOCK: position exists (duplicate protection).")
            self.tlog.append({"type": "blocked", "symbol": symbol, "reason": "position_exists"})
            return

        # Stops/freeze check
        ok_levels, why_levels = self._validate_stops_levels(symbol, direction, entry, sl, tp)
        if not ok_levels:
            logger.warning(f"BLOCK: levels invalid: {why_levels}")
            self.tlog.append({"type": "blocked", "symbol": symbol, "reason": "levels", "detail": why_levels})
            return

        # Lot sizing (dynamic if risk configured; fallback to cfg lot)
        fallback_lot = _safe_float(cfg_eff.get("lot", 0.01), 0.01)
        lot, lot_note = self._calc_lot(cfg_eff, symbol, direction, entry, sl, fallback_lot=fallback_lot)
        logger.info(f"LOT: {lot:.2f} ({lot_note})")

        # Execution dedupe
        sig = f"{symbol}|{direction}|{round(entry,2)}|{round(sl,2)}|{round(tp,2)}|{round(score,1)}|{int(decision.get('confidence',0))}"
        if not self._execution_dedupe_ok(cfg_eff, sig):
            logger.warning("BLOCK: execution dedupe/cooldown.")
            self.tlog.append({"type": "blocked", "symbol": symbol, "reason": "execution_dedupe"})
            return

        # 6) Telegram: approved-only
        msg = self._build_mentor_message(symbol, direction, pkg, decision, enable_execution, lot)
        self._safe_send_telegram_trade(cfg_eff, msg, sig)

        # 7) Execute (if enabled) else dry-run
        if not enable_execution:
            logger.info("DRY-RUN: enable_execution=false, skipping order_send.")
            self.tlog.append({"type": "dry_run", "symbol": symbol, "direction": direction, "decision": decision})
            return

        result = self._place_market_order(symbol, direction, lot, sl, tp)
        logger.info(f"EXEC_RESULT: ok={result.get('ok')} retcode={result.get('retcode')} comment={result.get('comment')}")
        self.tlog.append({"type": "execution", "symbol": symbol, "direction": direction, "result": result})

    def run_forever(self, interval_sec: int = 10) -> None:
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                logger.info("Stopped by user.")
                break
            except Exception as e:
                logger.error(f"CRITICAL: executor loop error: {e}")
                logger.error(traceback.format_exc())
                self.tlog.append({"type": "executor_error", "error": str(e), "traceback": traceback.format_exc()})
            time.sleep(interval_sec)


def main():
    ex = MentorExecutor()
    # One-shot by default (safe for commissioning). Use run_forever() if you want continuous loop.
    ex.run_once()


if __name__ == "__main__":
    main()
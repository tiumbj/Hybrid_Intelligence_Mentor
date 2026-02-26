"""
Mentor Executor - Execution Controller (Production Hardening)
Version: 2.4.0

Changelog:
- 2.4.0 (2026-02-26):
  - Add dynamic lot sizing by risk% equity (default 1%) using mt5.order_calc_profit()
  - Add execution safety hardening:
      * spread filter
      * trade_stops_level / trade_freeze_level validation
      * position check (no duplicate positions)
      * duplicate cooldown (loop protection)
  - Enforce ai.enabled:
      * ai.enabled=false -> NO AI CALL, fallback to technical gate
      * ai.enabled=true  -> AI confirm-only, strict response validation
  - Keep locked behavior:
      * Telegram send ONLY after final approval gate (no reject/NONE spam)
      * Engine is pricing owner; Executor enforces risk+safety

Rules enforced:
- Execute ONLY if final_approved==true AND confidence>=threshold AND enable_execution==true
- Telegram: only-on-approved (after confidence gate)
- No silent freeze: critical failures logged (with traceback)
"""

from __future__ import annotations

import os
import json
import time
import math
import logging
import traceback
from typing import Any, Dict, Optional, Tuple

import MetaTrader5 as mt5

from engine import TradingEngine
from ai_mentor import AIMentor
from telegram_notifier import TelegramNotifier
from trade_logger import TradeLogger
from config_resolver import resolve_effective_config, summarize_effective_config

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


def _fmt_float(v: Any, decimals: int = 2, na: str = "NA") -> str:
    try:
        if v is None:
            return na
        x = float(v)
        fmt = "{:." + str(int(decimals)) + "f}"
        return fmt.format(x)
    except Exception:
        return na


def _round_down_to_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return math.floor(x / step) * step


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

        # Telegram dedupe
        self._last_alert_sig: Optional[str] = None
        self._last_alert_ts: float = 0.0

        # Execution dedupe (order spam protection)
        self._last_exec_sig: Optional[str] = None
        self._last_exec_ts: float = 0.0

        self._ensure_mt5()

    # -----------------------------
    # MT5 utilities
    # -----------------------------
    def _ensure_mt5(self) -> None:
        if mt5.terminal_info() is not None:
            return
        if not mt5.initialize():
            err = mt5.last_error()
            raise RuntimeError(f"MT5 initialize failed: {err}")
        logger.info("MT5 Connected")

    def _get_tick(self, symbol: str):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Cannot get tick for symbol={symbol} last_error={mt5.last_error()}")
        return tick

    def _symbol_info(self, symbol: str):
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol not found: {symbol}")
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Cannot select symbol={symbol} last_error={mt5.last_error()}")
        return info

    def _account_equity(self) -> float:
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError(f"account_info None last_error={mt5.last_error()}")
        return float(getattr(acc, "equity", 0.0) or 0.0)

    # -----------------------------
    # Safety checks
    # -----------------------------
    @staticmethod
    def _sanity(direction: str, entry: float, sl: float, tp: float) -> bool:
        if direction == "BUY":
            return sl < entry < tp
        if direction == "SELL":
            return tp < entry < sl
        return False

    def _spread_points(self, symbol: str) -> float:
        info = self._symbol_info(symbol)
        tick = self._get_tick(symbol)
        point = float(getattr(info, "point", 0.0) or 0.0)
        if point <= 0:
            return 0.0
        return float((tick.ask - tick.bid) / point)

    def _has_open_position(self, symbol: str) -> bool:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            # ถ้าอ่านไม่ได้ ให้ถือว่า "มีความเสี่ยง" และ block
            return True
        return len(positions) > 0

    def _validate_stops_levels(
        self,
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
    ) -> Tuple[bool, str]:
        info = self._symbol_info(symbol)
        point = float(getattr(info, "point", 0.0) or 0.0)
        if point <= 0:
            return False, "invalid_point"

        stops_level = int(getattr(info, "trade_stops_level", 0) or 0)  # points
        freeze_level = int(getattr(info, "trade_freeze_level", 0) or 0)  # points

        # ระยะ SL/TP จาก entry เป็น points
        sl_dist_pts = abs(entry - sl) / point
        tp_dist_pts = abs(tp - entry) / point

        # stops_level: SL/TP ต้องห่างอย่างน้อย
        if stops_level > 0:
            if sl_dist_pts < stops_level:
                return False, f"sl_too_close(stops_level={stops_level}pts, sl_dist={sl_dist_pts:.1f}pts)"
            if tp_dist_pts < stops_level:
                return False, f"tp_too_close(stops_level={stops_level}pts, tp_dist={tp_dist_pts:.1f}pts)"

        # freeze_level: บางโบรกเกอร์ห้ามแก้ใกล้ราคา (ที่นี่ใช้เป็น hard gate ก่อนยิง order)
        if freeze_level > 0:
            # ใช้ tick ปัจจุบันประเมิน "ใกล้เกิน" แบบ conservative
            tick = self._get_tick(symbol)
            mkt = tick.ask if direction == "BUY" else tick.bid
            mkt_dist_pts = abs(entry - mkt) / point
            if mkt_dist_pts < freeze_level:
                return False, f"entry_too_close_freeze(freeze_level={freeze_level}pts, dist={mkt_dist_pts:.1f}pts)"

        return True, "ok"

    # -----------------------------
    # Risk sizing (1% equity by default)
    # -----------------------------
    def _calc_loss_per_1lot(
        self,
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
    ) -> Tuple[float, str]:
        """
        Returns: (abs_loss_money_per_1lot, status)
        Uses mt5.order_calc_profit() for best accuracy per broker contract.
        """
        info = self._symbol_info(symbol)
        _ = info  # keep for future extensions

        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

        # order_calc_profit: profit for closing at price_close
        # For BUY: open=entry, close=sl (below) => profit negative
        # For SELL: open=entry, close=sl (above) => profit negative
        try:
            p = mt5.order_calc_profit(order_type, symbol, 1.0, float(entry), float(sl))
            if p is None:
                return 0.0, f"calc_profit_none last_error={mt5.last_error()}"
            loss = abs(float(p))
            return loss, "ok"
        except Exception as e:
            return 0.0, f"calc_profit_exception: {e}"

    def _calc_lot_by_risk(
        self,
        cfg_eff: Dict[str, Any],
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
    ) -> Tuple[float, str]:
        """
        risk_percent default 1.0 (ถ้าไม่ได้ใส่ config)
        ปรับ lot ให้เข้ากับ min/step/max ของ symbol
        """
        risk_cfg = (cfg_eff.get("risk") or {}) if isinstance(cfg_eff, dict) else {}
        risk_percent = _safe_float(risk_cfg.get("risk_percent"), 1.0)  # NEW (default 1.0%)
        risk_percent = max(0.0, min(5.0, risk_percent))  # guardrail: 0–5%

        equity = self._account_equity()
        risk_amount = equity * (risk_percent / 100.0)

        loss_1lot, st = self._calc_loss_per_1lot(symbol, direction, entry, sl)
        if loss_1lot <= 0:
            return float(cfg_eff.get("lot", 0.01) or 0.01), f"fallback_lot(loss_1lot={loss_1lot} st={st})"

        raw_lot = risk_amount / loss_1lot

        info = self._symbol_info(symbol)
        vmin = float(getattr(info, "volume_min", 0.01) or 0.01)
        vmax = float(getattr(info, "volume_max", 100.0) or 100.0)
        vstep = float(getattr(info, "volume_step", vmin) or vmin)

        lot = _round_down_to_step(raw_lot, vstep)
        lot = max(vmin, min(vmax, lot))

        # ถ้าคำนวณแล้วต่ำกว่า min มาก ให้ clamp เป็น min
        return float(lot), f"ok(risk%={risk_percent:.2f} equity={equity:.2f} risk_amt={risk_amount:.2f} loss_1lot={loss_1lot:.2f})"

    # -----------------------------
    # Telegram (locked: approved only)
    # -----------------------------
    def _should_send_approved_alert(self, cfg_eff: Dict[str, Any], symbol: str, direction: str, entry: float, sl: float, tp: float) -> bool:
        tg_cfg = (cfg_eff.get("telegram") or {}) if isinstance(cfg_eff, dict) else {}
        cooldown_sec = int(tg_cfg.get("cooldown_sec", 900))
        sig = f"{symbol}|{direction}|{entry:.2f}|{sl:.2f}|{tp:.2f}"
        now = time.time()
        if self._last_alert_sig == sig and (now - self._last_alert_ts) < cooldown_sec:
            return False
        self._last_alert_sig = sig
        self._last_alert_ts = now
        return True

    @staticmethod
    def _build_mentor_message(
        symbol: str,
        direction: str,
        pkg: Dict[str, Any],
        approved_source: str,
        confidence: int,
        entry: float,
        sl: float,
        tp: float,
        lot: float,
        enable_execution: bool,
        extra_note: str = "",
    ) -> str:
        ctx = pkg.get("context", {}) or {}
        score = _safe_float(pkg.get("score", 0.0), 0.0)
        rr = _safe_float(pkg.get("rr", 0.0), 0.0)

        status = "APPROVED"
        if not enable_execution:
            status += " (DRY-RUN)"
        if extra_note:
            status += f" | {extra_note}"

        msg = []
        msg.append(f"HIM Signal — {status}")
        msg.append(f"Symbol: {symbol}")
        msg.append(f"Direction: {direction}")
        msg.append(f"ApprovedBy: {approved_source}")
        msg.append(f"Confidence: {confidence}%")
        msg.append(f"Lot: {lot:.2f}")
        msg.append(f"Entry: {entry:.2f}")
        msg.append(f"SL: {sl:.2f}")
        msg.append(f"TP: {tp:.2f}")
        msg.append("")
        msg.append("Mentor Explanation")
        msg.append(f"1) HTF: {ctx.get('HTF_trend')} | MTF: {ctx.get('MTF_trend')} | LTF: {ctx.get('LTF_trend')} (bias={ctx.get('bias_source')})")
        msg.append(f"2) BOS(required): {ctx.get('bos')} | Retest: {ctx.get('retest_ok')} | Breakout: {ctx.get('breakout_state')}({ctx.get('breakout_side')})")
        msg.append(f"3) Proximity: score={_fmt_float(ctx.get('proximity_score'),1)} side={ctx.get('proximity_side')} dist={_fmt_float(ctx.get('proximity_best_dist'),2)} thrPts={_fmt_float(ctx.get('breakout_proximity_threshold_points'),2)}")
        msg.append(f"4) Score={score:.1f}/10 | RR={rr:.2f} | blocked_by={ctx.get('blocked_by')}")
        if extra_note:
            msg.append("")
            msg.append(f"Note: {extra_note}")
        return "\n".join(msg)

    def _safe_send_telegram_approved(
        self,
        cfg_eff: Dict[str, Any],
        symbol: str,
        direction: str,
        pkg: Dict[str, Any],
        approved_source: str,
        confidence: int,
        entry: float,
        sl: float,
        tp: float,
        lot: float,
        enable_execution: bool,
        extra_note: str,
    ) -> None:
        try:
            if not self._should_send_approved_alert(cfg_eff, symbol, direction, entry, sl, tp):
                logger.info("Telegram: dedupe/cooldown suppress.")
                return
            msg = self._build_mentor_message(
                symbol=symbol,
                direction=direction,
                pkg=pkg,
                approved_source=approved_source,
                confidence=confidence,
                entry=entry,
                sl=sl,
                tp=tp,
                lot=lot,
                enable_execution=enable_execution,
                extra_note=extra_note,
            )
            ok = self.tg.send_text(msg, event_type="trade")
            if not ok:
                logger.warning("Telegram send failed or disabled.")
                self.tlog.append({"type": "telegram_failed", "symbol": symbol, "direction": direction})
        except Exception:
            logger.error("CRITICAL: telegram build/send failed (isolated).")
            logger.error(traceback.format_exc())
            self.tlog.append({"type": "telegram_exception", "symbol": symbol, "direction": direction, "traceback": traceback.format_exc()})

    # -----------------------------
    # Order send
    # -----------------------------
    def _place_market_order(self, symbol: str, direction: str, lot: float, sl: float, tp: float) -> Dict[str, Any]:
        info = self._symbol_info(symbol)
        tick = self._get_tick(symbol)
        price = float(tick.ask if direction == "BUY" else tick.bid)

        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 30,
            "magic": 20260226,
            "comment": "HIM v2.4.0",
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
    # Main loop
    # -----------------------------
    def _exec_dedupe_ok(self, cfg_eff: Dict[str, Any], symbol: str, direction: str, entry: float, sl: float, tp: float) -> bool:
        ex_cfg = (cfg_eff.get("execution_safety") or {}) if isinstance(cfg_eff, dict) else {}
        cooldown_sec = int(ex_cfg.get("cooldown_sec", 120))  # NEW default 120s
        sig = f"{symbol}|{direction}|{entry:.2f}|{sl:.2f}|{tp:.2f}"
        now = time.time()
        if self._last_exec_sig == sig and (now - self._last_exec_ts) < cooldown_sec:
            return False
        self._last_exec_sig = sig
        self._last_exec_ts = now
        return True

    def run_once(self) -> None:
        # 0) Load effective config
        cfg_eff = self.hcfg.load_effective()

        symbol = str(cfg_eff.get("symbol", "GOLD"))
        enable_execution = bool(cfg_eff.get("enable_execution", False))
        conf_th = int(cfg_eff.get("confidence_threshold", 75))
        min_score = _safe_float(cfg_eff.get("min_score", 0.0), 0.0)

        ai_cfg = (cfg_eff.get("ai") or {}) if isinstance(cfg_eff, dict) else {}
        ai_enabled = bool(ai_cfg.get("enabled", False))

        # Safety config (NEW defaults)
        safety_cfg = (cfg_eff.get("execution_safety") or {}) if isinstance(cfg_eff, dict) else {}
        max_spread_points = _safe_float(safety_cfg.get("max_spread_points"), 50.0)  # default 50 points
        block_if_position_exists = bool(safety_cfg.get("block_if_position_exists", True))

        # Log effective snapshot (compact)
        try:
            logger.info(f"CFG_EFFECTIVE: {summarize_effective_config(cfg_eff)}")
        except Exception:
            logger.info(f"CFG_EFFECTIVE: {cfg_eff}")

        # 1) Engine signal package
        try:
            pkg = self.engine.generate_signal_package()
        except Exception as e:
            logger.error(f"CRITICAL: engine failed: {e}")
            logger.error(traceback.format_exc())
            self.tlog.append({"type": "engine_error", "symbol": symbol, "error": str(e), "traceback": traceback.format_exc()})
            return

        direction = str(pkg.get("direction", "NONE"))
        score = _safe_float(pkg.get("score", 0.0), 0.0)
        rr = _safe_float(pkg.get("rr", 0.0), 0.0)
        confidence_py = _safe_int(pkg.get("confidence_py", 0), 0)

        logger.info(f"SIGNAL: {direction} score={score:.1f} rr={rr:.2f} confidence_py={confidence_py}")
        self.tlog.append({"type": "signal", "symbol": symbol, "package": pkg})

        if direction not in ("BUY", "SELL"):
            logger.info("No clear bias. Skip.")
            return

        # 2) Build trade numbers source (AI or technical)
        approved_source = "TECHNICAL" if not ai_enabled else "AI"
        final_approved = False
        confidence = 0
        entry = 0.0
        sl = 0.0
        tp = 0.0
        extra_note = ""

        if not ai_enabled:
            # Technical gate (fallback)
            # - ใช้ engine candidates เป็นตัวเลข
            # - approved เมื่อ score/confidence_py ผ่าน threshold
            entry = _safe_float(pkg.get("entry_candidate"), 0.0)
            sl = _safe_float(pkg.get("stop_candidate"), 0.0)
            tp = _safe_float(pkg.get("tp_candidate"), 0.0)

            confidence = int(confidence_py)
            if score >= min_score and confidence >= conf_th and self._sanity(direction, entry, sl, tp):
                final_approved = True
                extra_note = "ai.enabled=false -> technical gate"
            else:
                logger.info("Gate: technical rejected (score/conf/sanity).")
                self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "technical_reject", "score": score, "confidence_py": confidence_py})
                return
        else:
            # AI confirm-only
            ai_resp = self.ai.approve_trade(pkg)

            if not AIMentor.validate_response(ai_resp):
                logger.error("CRITICAL: AI response invalid shape. Skip execution.")
                self.tlog.append({"type": "ai_invalid", "symbol": symbol, "package": pkg, "ai_raw": ai_resp})
                return

            approved = bool(ai_resp["approved"])
            confidence = int(ai_resp["confidence"])
            entry = float(ai_resp["entry"])
            sl = float(ai_resp["sl"])
            tp = float(ai_resp["tp"])

            logger.info(f"AI: approved={approved} confidence={confidence}%")
            if ai_resp.get("reasoning"):
                logger.info("REASONING:\n" + str(ai_resp["reasoning"]))
            if ai_resp.get("warnings"):
                logger.info("WARNINGS:\n" + str(ai_resp["warnings"]))

            self.tlog.append({"type": "ai_decision", "symbol": symbol, "direction": direction, "ai": ai_resp})

            # Final gates (AI)
            if not approved:
                logger.info("Gate: rejected by AI.")
                self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "ai_rejected"})
                return
            if confidence < conf_th:
                logger.info(f"Gate: confidence below threshold ({confidence} < {conf_th}).")
                self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "confidence_below_threshold", "confidence": confidence, "threshold": conf_th})
                return
            if not self._sanity(direction, entry, sl, tp):
                logger.error("CRITICAL: price sanity check failed. Skip execution.")
                self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "sanity_failed"})
                return

            final_approved = True

        # 3) Safety gates (spread/levels/position/duplicate)
        sp = self._spread_points(symbol)
        if sp > max_spread_points:
            logger.info(f"Gate: spread too high ({sp:.1f} > {max_spread_points}).")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "spread_high", "spread_points": sp})
            return

        ok_lv, msg_lv = self._validate_stops_levels(symbol, direction, entry, sl, tp)
        if not ok_lv:
            logger.info(f"Gate: stops/freeze validation failed: {msg_lv}")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "stops_level_failed", "detail": msg_lv})
            return

        if block_if_position_exists and self._has_open_position(symbol):
            logger.info("Gate: position exists (no stacking).")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "position_exists"})
            return

        if not self._exec_dedupe_ok(cfg_eff, symbol, direction, entry, sl, tp):
            logger.info("Gate: execution dedupe/cooldown suppress.")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "exec_dedupe"})
            return

        if not final_approved:
            logger.info("Gate: not approved (unexpected).")
            return

        # 4) Risk sizing (dynamic lot)
        try:
            lot, lot_msg = self._calc_lot_by_risk(cfg_eff, symbol, direction, entry, sl)
            logger.info(f"RISK LOT: {lot:.2f} ({lot_msg})")
        except Exception as e:
            lot = float(cfg_eff.get("lot", 0.01) or 0.01)
            lot_msg = f"risk_calc_exception -> fallback lot={lot} err={e}"
            logger.error(lot_msg)
            self.tlog.append({"type": "risk_calc_error", "symbol": symbol, "error": str(e), "traceback": traceback.format_exc()})

        # 5) Telegram (LOCKED): send only after APPROVED+confidence gate
        self._safe_send_telegram_approved(
            cfg_eff=cfg_eff,
            symbol=symbol,
            direction=direction,
            pkg=pkg,
            approved_source=approved_source,
            confidence=confidence,
            entry=entry,
            sl=sl,
            tp=tp,
            lot=lot,
            enable_execution=enable_execution,
            extra_note=lot_msg,
        )

        # 6) Execute (only if enable_execution==true)
        if not enable_execution:
            logger.info("Gate: enable_execution=false (dry-run).")
            self.tlog.append({"type": "gate_reject", "symbol": symbol, "reason": "execution_disabled"})
            return

        order_res = self._place_market_order(symbol=symbol, direction=direction, lot=lot, sl=sl, tp=tp)
        if order_res.get("ok"):
            logger.info(f"ORDER OK: ticket={order_res.get('order')} price={order_res.get('price')}")
        else:
            logger.error(f"ORDER FAILED: {order_res}")

        self.tlog.append(
            {
                "type": "order",
                "symbol": symbol,
                "direction": direction,
                "approved_source": approved_source,
                "confidence_threshold": conf_th,
                "enable_execution": enable_execution,
                "lot": lot,
                "spread_points": sp,
                "order_result": order_res,
            }
        )

    def run_loop(self, interval_sec: int = 15) -> None:
        logger.info("HIM MentorExecutor loop started.")
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"CRITICAL LOOP ERROR: {e}")
                logger.error(traceback.format_exc())
                self.tlog.append({"type": "loop_error", "error": str(e), "traceback": traceback.format_exc()})
            time.sleep(interval_sec)


if __name__ == "__main__":
    ex = MentorExecutor()
    ex.run_loop(interval_sec=15)
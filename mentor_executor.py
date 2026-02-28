"""
Mentor Executor - Execution Controller (Phase 9 Live Stabilization)
Version: 2.6.2

Changelog:
- 2.6.2 (2026-02-27):
  - FIX: Handle positions with SL=0.0 (manual / external) safely
      * Auto-set initial SL using ATR before recording initial_risk
      * Never store initial_risk from SL=0.0
  - FIX: Add MT5 stops/freeze/spread-aware clamp for ALL SL modifications
      * Prevent SL too close to current price (reduce stopouts in sideway/noise)
      * Centralized SL guard for BE / Trail / Emergency / AI tighten-only
  - IMPROVE: More explicit logs for SLTP modify outcomes (retcode/comment)

Frozen decisions:
- mode=sideway_scalp, min_rr=1.5, atr_sl_mult=1.6, require_confirmation=true
- AI cannot change direction, cannot change lot sizing
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5

from ai_mentor import AIMentor
from config_resolver import resolve_effective_config
from engine import TradingEngine
from telegram_notifier import TelegramNotifier
from trade_logger import TradeLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("him_system.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("HIM")


def _sf(v: Any, d: float = 0.0) -> float:
    try:
        if v is None:
            return float(d)
        return float(v)
    except Exception:
        return float(d)


def _si(v: Any, d: int = 0) -> int:
    try:
        if v is None:
            return int(d)
        return int(v)
    except Exception:
        return int(d)


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _sanity(direction: str, entry: float, sl: float, tp: float) -> bool:
    if not (_is_finite(entry) and _is_finite(sl) and _is_finite(tp)):
        return False
    if direction == "BUY":
        return sl < entry < tp
    if direction == "SELL":
        return tp < entry < sl
    return False


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


class PositionStateStore:
    """
    เก็บ initial risk ต่อ ticket เพื่อคำนวณ R (เพราะ SL จะถูกขยับภายหลัง)
    ไฟล์: .state/positions.json
    """

    def __init__(self, base_dir: str):
        self.dir = os.path.join(base_dir, ".state")
        self.path = os.path.join(self.dir, "positions.json")
        os.makedirs(self.dir, exist_ok=True)
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.path):
                return {}
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @staticmethod
    def _sl_is_valid(direction: str, entry: float, sl: float) -> bool:
        if not (_is_finite(entry) and _is_finite(sl)):
            return False
        if sl <= 0:
            return False
        if direction == "BUY":
            return sl < entry
        if direction == "SELL":
            return sl > entry
        return False

    def ensure_ticket(self, ticket: int, direction: str, entry: float, sl: float) -> None:
        """
        IMPORTANT:
        - ห้ามบันทึก initial_risk ถ้า sl=0 หรือ sl ผิดทิศ (จะทำให้ risk เพี้ยน)
        """
        k = str(ticket)
        if k in self.state:
            return
        if not self._sl_is_valid(direction, entry, sl):
            return

        risk = abs(entry - sl)
        if not _is_finite(risk) or risk <= 0:
            return

        self.state[k] = {
            "entry": float(entry),
            "initial_sl": float(sl),
            "initial_risk": float(risk),
            "ts": time.time(),
        }
        self.save()

    def remove_missing(self, live_tickets: List[int]) -> None:
        live = set(str(t) for t in live_tickets)
        removed = False
        for k in list(self.state.keys()):
            if k not in live:
                self.state.pop(k, None)
                removed = True
        if removed:
            self.save()

    def get_initial(self, ticket: int) -> Optional[Dict[str, Any]]:
        return self.state.get(str(ticket))


class MentorExecutor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        self.config_path = os.path.join(base_dir, "config.json")
        self.hcfg = HotConfig(self.config_path)

        self.engine = TradingEngine(config_path=self.config_path)
        self.ai = AIMentor()
        self.tg = TelegramNotifier(config_path=self.config_path)
        self.tlog = TradeLogger(os.path.join(base_dir, "trade_history.json"))
        self.pstore = PositionStateStore(base_dir)

        self._last_exec_sig: Optional[str] = None
        self._last_exec_ts: float = 0.0

        self._ensure_mt5()

    # -----------------------------
    # MT5
    # -----------------------------
    def _ensure_mt5(self) -> None:
        if mt5.terminal_info() is not None:
            return
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        logger.info("MT5 Connected")

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
    def _get_tick(symbol: str):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Cannot get tick for symbol: {symbol} last_error={mt5.last_error()}")
        return tick

    @staticmethod
    def _positions(symbol: str):
        pos = mt5.positions_get(symbol=symbol)
        return list(pos) if pos else []

    def _spread_points(self, symbol: str) -> float:
        info = self._get_info(symbol)
        tick = self._get_tick(symbol)
        if info.point <= 0:
            return float("inf")
        return float((tick.ask - tick.bid) / info.point)

    # -----------------------------
    # Multi-order
    # -----------------------------
    def _multi_order_policy(self, cfg_eff: Dict[str, Any]) -> Dict[str, Any]:
        ex = (cfg_eff.get("execution") or {})
        return {
            "allow_multiple_positions": bool(ex.get("allow_multiple_positions", True)),
            "max_positions_per_symbol": _si(ex.get("max_positions_per_symbol", 2), 2),
            "min_distance_between_entries_atr": _sf(ex.get("min_distance_between_entries_atr", 0.60), 0.60),
            "cooldown_sec": _si(ex.get("cooldown_sec", 120), 120),
        }

    def _execution_dedupe_ok(self, policy: Dict[str, Any], sig: str) -> bool:
        cooldown = int(policy.get("cooldown_sec", 120))
        now = time.time()
        if self._last_exec_sig == sig and (now - self._last_exec_ts) < cooldown:
            return False
        self._last_exec_sig = sig
        self._last_exec_ts = now
        return True

    def _stacking_ok(self, symbol: str, entry: float, atr: float, policy: Dict[str, Any]) -> Tuple[bool, str]:
        pos = self._positions(symbol)
        if not pos:
            return True, "no_positions"

        max_pos = int(policy["max_positions_per_symbol"])
        if len(pos) >= max_pos:
            return False, f"max_positions_reached={len(pos)}/{max_pos}"

        min_dist = float(policy["min_distance_between_entries_atr"]) * max(atr, 1e-9)

        for p in pos:
            p_price = float(getattr(p, "price_open", 0.0) or 0.0)
            if abs(entry - p_price) < min_dist:
                return False, f"stacking_too_close dist={abs(entry - p_price):.5f} min={min_dist:.5f}"

        return True, "stacking_ok"

    # -----------------------------
    # Stops / Freeze helpers (for SL management)
    # -----------------------------
    def _get_min_guard_distance(self, symbol: str) -> Tuple[float, float, float]:
        """
        คืนค่า (point, stop_min_dist_price, freeze_min_dist_price)
        """
        info = self._get_info(symbol)
        point = float(info.point) if info.point else 0.0
        if point <= 0:
            return 0.0, float("inf"), float("inf")

        stops_level_points = int(getattr(info, "trade_stops_level", 0) or 0)
        freeze_level_points = int(getattr(info, "trade_freeze_level", 0) or 0)

        stop_min = float(stops_level_points) * point
        freeze_min = float(freeze_level_points) * point
        return point, stop_min, freeze_min

    def _clamp_sl_to_market(
        self,
        symbol: str,
        direction: str,
        entry: float,
        price_now: float,
        desired_sl: float,
        extra_buffer_points: int = 3,
    ) -> Tuple[Optional[float], str]:
        """
        Clamp SL ให้ไม่ชิดราคาเกินไป (กันโดน noise + กัน MT5 reject)
        - BUY: SL ต้อง <= (bid - min_dist)
        - SELL: SL ต้อง >= (ask + min_dist)
        """
        if not (_is_finite(entry) and _is_finite(price_now) and _is_finite(desired_sl)):
            return None, "clamp_invalid_inputs"

        info = self._get_info(symbol)
        point, stop_min, freeze_min = self._get_min_guard_distance(symbol)
        if point <= 0:
            return None, "clamp_invalid_point"

        tick = self._get_tick(symbol)
        bid = float(tick.bid)
        ask = float(tick.ask)

        # spread buffer: ใช้ spread จริง + buffer points เล็กน้อย
        spread_pts = self._spread_points(symbol)
        spread_buf = max(0.0, spread_pts * point)

        # choose guard: max(stop, freeze) + spread_buffer + extra
        extra = float(max(extra_buffer_points, 0)) * point
        guard = max(stop_min, freeze_min) + spread_buf + extra

        if direction == "BUY":
            # must be below current bid by guard
            max_sl = bid - guard
            new_sl = min(float(desired_sl), max_sl)

            # ต้องต่ำกว่า entry เพื่อความหมาย SL ของ BUY (กันความสับสน)
            # แต่อนุญาตให้ล็อกกำไรได้โดยให้ <= bid-guard อยู่แล้ว
            # ที่นี่เรา clamp ไม่ให้สูงกว่า bid-guard; ถ้าสูงกว่า entry แต่ยังต่ำกว่า bid-guard ก็ถือว่าเป็น "stop-profit"
            if not _is_finite(new_sl):
                return None, "clamp_buy_nan"
            return new_sl, f"clamp_buy guard={guard:.5f}"

        if direction == "SELL":
            min_sl = ask + guard
            new_sl = max(float(desired_sl), min_sl)
            if not _is_finite(new_sl):
                return None, "clamp_sell_nan"
            return new_sl, f"clamp_sell guard={guard:.5f}"

        return None, "clamp_invalid_direction"

    # -----------------------------
    # SLTP modify
    # -----------------------------
    def _modify_position_sltp(
        self,
        symbol: str,
        ticket: int,
        new_sl: Optional[float],
        new_tp: Optional[float],
    ) -> Tuple[bool, str]:
        req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": symbol, "position": int(ticket)}
        if new_sl is not None:
            req["sl"] = float(new_sl)
        if new_tp is not None:
            req["tp"] = float(new_tp)

        res = mt5.order_send(req)
        if res is None:
            return False, f"sltp_none err={mt5.last_error()}"
        ret = getattr(res, "retcode", None)
        if ret not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            return False, f"sltp_retcode={ret} comment={getattr(res,'comment','')}"
        return True, f"sltp_ok retcode={ret}"

    # -----------------------------
    # Hybrid SL Manager (Mode C)
    # -----------------------------
    def _manage_positions(self, cfg_eff: Dict[str, Any], symbol: str, market_ctx: Dict[str, Any]) -> None:
        positions = self._positions(symbol)
        tickets = [int(getattr(p, "ticket", 0) or 0) for p in positions]
        self.pstore.remove_missing(tickets)

        if not positions:
            return

        ex = (cfg_eff.get("execution") or {})

        # Defaults
        be_r = _sf(ex.get("break_even_r", 0.60), 0.60)
        trail_start_r = _sf(ex.get("trail_start_r", 1.00), 1.00)

        # Trail ATR multiplier (เพิ่ม default เพื่อกัน sideway stopout)
        trail_atr_mult = _sf(ex.get("trail_atr_mult", 1.60), 1.60)

        emergency_buffer_atr = _sf(ex.get("emergency_sl_buffer_atr", 0.10), 0.10)

        # Initial SL if missing
        init_sl_atr_mult = _sf(ex.get("pos_init_sl_atr_mult", 1.60), 1.60)

        # ATR from engine context
        atr = _sf(market_ctx.get("atr", 0.0), 0.0)
        if atr <= 0:
            # no ATR -> skip to avoid nonsense
            return

        tick = self._get_tick(symbol)

        flip = bool(market_ctx.get("flip", False))
        event_risk = str(market_ctx.get("event_risk", "NONE")).upper()
        session = str(market_ctx.get("session", "unknown")).lower()

        for p in positions:
            try:
                ticket = int(getattr(p, "ticket", 0) or 0)
                p_type = int(getattr(p, "type", 0) or 0)
                entry = float(getattr(p, "price_open", 0.0) or 0.0)
                sl_cur = float(getattr(p, "sl", 0.0) or 0.0)
                tp_cur = float(getattr(p, "tp", 0.0) or 0.0)

                direction = "BUY" if p_type == mt5.POSITION_TYPE_BUY else "SELL"
                price_now = float(tick.bid if direction == "BUY" else tick.ask)

                # 0) If SL missing / invalid, set an initial SL FIRST (fail-closed on state)
                if sl_cur <= 0 or (direction == "BUY" and not (sl_cur < entry)) or (direction == "SELL" and not (sl_cur > entry)):
                    sl_dist = init_sl_atr_mult * atr
                    desired = (entry - sl_dist) if direction == "BUY" else (entry + sl_dist)
                    clamped, why = self._clamp_sl_to_market(symbol, direction, entry, price_now, desired, extra_buffer_points=5)
                    if clamped is not None:
                        ok, msg = self._modify_position_sltp(symbol, ticket, clamped, None)
                        if ok:
                            self._safe_notify(f"[INIT SL] {symbol} ticket={ticket} dir={direction} sl={clamped:.5f} | {why}")
                            sl_cur = float(clamped)
                        else:
                            logger.warning(f"INIT SL failed ticket={ticket} {msg}")
                            continue
                    else:
                        logger.warning(f"INIT SL clamp failed ticket={ticket} why={why}")
                        continue

                # ensure stored initial risk ONLY after SL valid
                self.pstore.ensure_ticket(ticket, direction, entry, sl_cur)
                init = self.pstore.get_initial(ticket)
                if not init:
                    continue
                init_risk = float(init.get("initial_risk", 0.0) or 0.0)
                if init_risk <= 0 or not _is_finite(init_risk):
                    continue

                # Compute R multiple
                pnl = (price_now - entry) if direction == "BUY" else (entry - price_now)
                r_mult = pnl / init_risk

                # 1) Emergency SL on flip
                if flip:
                    buf = emergency_buffer_atr * atr
                    desired = (price_now - buf) if direction == "BUY" else (price_now + buf)
                    clamped, why = self._clamp_sl_to_market(symbol, direction, entry, price_now, desired, extra_buffer_points=5)
                    if clamped is not None:
                        ok, msg = self._modify_position_sltp(symbol, ticket, clamped, None)
                        if ok:
                            self._safe_notify(f"[EMERGENCY SL] {symbol} ticket={ticket} new_sl={clamped:.5f} R={r_mult:.2f} | {why}")
                    continue

                # 2) Break-even at be_r
                if r_mult >= be_r:
                    # ใช้ buffer อิง spread มากกว่า ATR เพื่อกัน sideway stopout
                    point, _, _ = self._get_min_guard_distance(symbol)
                    spread_pts = self._spread_points(symbol)
                    be_buf = max(2.0 * (point if point > 0 else 0.0), spread_pts * (point if point > 0 else 0.0))

                    desired = (entry + be_buf) if direction == "BUY" else (entry - be_buf)
                    clamped, why = self._clamp_sl_to_market(symbol, direction, entry, price_now, desired, extra_buffer_points=3)
                    if clamped is not None:
                        if (direction == "BUY" and clamped > sl_cur) or (direction == "SELL" and clamped < sl_cur):
                            ok, msg = self._modify_position_sltp(symbol, ticket, clamped, None)
                            if ok:
                                self._safe_notify(f"[BE] {symbol} ticket={ticket} R={r_mult:.2f} new_sl={clamped:.5f} | {why}")

                # 3) ATR trail after trail_start_r
                if r_mult >= trail_start_r:
                    desired = (price_now - trail_atr_mult * atr) if direction == "BUY" else (price_now + trail_atr_mult * atr)
                    clamped, why = self._clamp_sl_to_market(symbol, direction, entry, price_now, desired, extra_buffer_points=3)
                    if clamped is not None:
                        if (direction == "BUY" and clamped > sl_cur) or (direction == "SELL" and clamped < sl_cur):
                            ok, msg = self._modify_position_sltp(symbol, ticket, clamped, None)
                            if ok:
                                self._safe_notify(f"[TRAIL] {symbol} ticket={ticket} R={r_mult:.2f} new_sl={clamped:.5f} | {why}")

                # 4) AI tighten-only (เฉพาะลดความเสี่ยง)
                spread = self._spread_points(symbol)
                max_spread = _sf(((cfg_eff.get("execution_safety") or {}).get("max_spread_points", 35)), 35)
                spread_z = 2.0 if spread > max_spread else 1.6 if spread > 25 else 0.8

                pos_pkg = {
                    "symbol": symbol,
                    "position": {"dir": direction, "entry": entry, "sl": sl_cur, "tp": tp_cur, "price_now": price_now, "atr": atr},
                    "context": {"event_risk": event_risk, "session": session, "regime": market_ctx.get("regime", "unknown")},
                    "execution_context": {"spread_z": spread_z, "slippage_estimate": 0.3},
                    "portfolio": {"open_positions_symbol": len(positions), "correlation_risk": _sf(market_ctx.get("correlation_risk"), 0.0)},
                    "constraints": {
                        "sl_atr_min": _sf(((cfg_eff.get("ai") or {}).get("sl_atr_min", 1.20)), 1.20),
                        "sl_atr_max": _sf(((cfg_eff.get("ai") or {}).get("sl_atr_max", 1.80)), 1.80),
                        "event_conf_cap_high": _si(((cfg_eff.get("ai") or {}).get("event_conf_cap_high", 72)), 72),
                        "pos_sl_tighten_max_atr": _sf(((cfg_eff.get("ai") or {}).get("pos_sl_tighten_max_atr", 0.30)), 0.30),
                        "pos_sl_min_improve_atr": _sf(((cfg_eff.get("ai") or {}).get("pos_sl_min_improve_atr", 0.03)), 0.03),
                    },
                }

                ai_pos = self.ai.evaluate_position_sl(pos_pkg)
                decision = str(((ai_pos.get("position") or {}).get("decision", "NO_CHANGE"))).upper()
                ai_new_sl = (ai_pos.get("position") or {}).get("new_sl", None)

                if decision == "TIGHTEN" and ai_new_sl is not None:
                    desired = float(ai_new_sl)

                    # tighten-only hard check (never widen)
                    if (direction == "BUY" and desired <= sl_cur) or (direction == "SELL" and desired >= sl_cur):
                        continue

                    clamped, why = self._clamp_sl_to_market(symbol, direction, entry, price_now, desired, extra_buffer_points=3)
                    if clamped is None:
                        continue

                    if (direction == "BUY" and clamped > sl_cur) or (direction == "SELL" and clamped < sl_cur):
                        ok, msg = self._modify_position_sltp(symbol, ticket, clamped, None)
                        if ok:
                            mentor = ai_pos.get("mentor") or {}
                            self._safe_notify(f"[AI SL TIGHTEN] {symbol} ticket={ticket} new_sl={clamped:.5f} | {why} | {mentor.get('headline','')}")
            except Exception:
                continue

    def _safe_notify(self, text: str) -> None:
        try:
            self.tg.send_text(text, event_type="info")
        except Exception:
            pass

    # -----------------------------
    # AI package builder (new trades)
    # -----------------------------
    def _build_ai_package(self, cfg_eff: Dict[str, Any], pkg: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(cfg_eff.get("symbol", "GOLD"))
        direction = str(pkg.get("direction", "NONE")).upper()

        entry0 = _sf(pkg.get("entry_candidate"))
        sl0 = _sf(pkg.get("stop_candidate"))
        tp0 = _sf(pkg.get("tp_candidate"))
        rr0 = _sf(pkg.get("rr"), 0.0)

        ctx = (pkg.get("context") or {})
        atr = _sf(ctx.get("atr", pkg.get("atr", 0.0)), 0.0)

        spread_pts = self._spread_points(symbol)
        info = self._get_info(symbol)
        exec_ctx = {
            "spread_current": float(spread_pts),
            "spread_avg_5m": float(spread_pts),
            "spread_z": 0.0,
            "slippage_estimate": 0.3,
            "freeze_level": int(getattr(info, "trade_freeze_level", 0) or 0),
            "stop_level": int(getattr(info, "trade_stops_level", 0) or 0),
        }

        structure = {
            "bos": bool(pkg.get("bos", False)),
            "choch": bool(ctx.get("choch", False)),
            "htf_bias": ctx.get("HTF_trend") or ctx.get("htf_bias") or "unknown",
            "mtf_bias": ctx.get("MTF_trend") or ctx.get("mtf_bias") or "unknown",
            "ltf_bias": ctx.get("LTF_trend") or ctx.get("ltf_bias") or "unknown",
            "proximity_score": ctx.get("proximity_score"),
        }

        technical = {
            "bb_state": ctx.get("bb_state") or ctx.get("bb"),
            "rsi": ctx.get("rsi"),
            "adx": ctx.get("adx"),
            "supertrend": ctx.get("supertrend_dir") or "unknown",
            "volatility_z": ctx.get("vol_z") or ctx.get("vol"),
        }

        context = {
            "regime": ctx.get("regime") or cfg_eff.get("mode") or "unknown",
            "session": ctx.get("session") or "unknown",
            "event_risk": ctx.get("event_risk") or "NONE",
            "time_to_event_min": ctx.get("time_to_event_min"),
            "correlations": ctx.get("correlations") or {},
            "liquidity": ctx.get("liq") or ctx.get("liquidity") or "",
        }

        positions = self._positions(symbol)
        portfolio = {
            "open_positions_symbol": len(positions),
            "daily_pnl_usd": 0.0,
            "equity_drawdown_pct": 0.0,
            "correlation_risk": _sf(ctx.get("correlation_risk"), 0.0),
        }

        ai_cfg = (cfg_eff.get("ai") or {})
        constraints = {
            "min_rr": _sf(cfg_eff.get("min_rr", 1.5), 1.5),
            "entry_shift_max_atr": _sf(ai_cfg.get("entry_shift_max_atr", 0.20), 0.20),
            "sl_atr_min": _sf(ai_cfg.get("sl_atr_min", 1.20), 1.20),
            "sl_atr_max": _sf(ai_cfg.get("sl_atr_max", 1.80), 1.80),
            "conf_execute_threshold": _si(cfg_eff.get("confidence_threshold", 70), 70),
            "event_conf_cap_high": _si(ai_cfg.get("event_conf_cap_high", 72), 72),
        }

        return {
            "symbol": symbol,
            "baseline": {"dir": direction, "entry": entry0, "sl": sl0, "tp": tp0, "rr": rr0, "atr": atr, "spread_points": spread_pts},
            "execution_context": exec_ctx,
            "technical": technical,
            "structure": structure,
            "context": context,
            "portfolio": portfolio,
            "constraints": constraints,
            "ask": "Validate baseline, adjust within constraints, output execution/analysis/mentor",
        }

    # -----------------------------
    # Python-side AI validation (new trades)
    # -----------------------------
    def _validate_ai_execution(self, ai_out: Dict[str, Any], baseline: Dict[str, Any], constraints: Dict[str, Any]) -> Tuple[bool, str]:
        ex = (ai_out.get("execution") or {})
        direction = str(baseline.get("dir", "NONE")).upper()
        entry0, sl0, tp0 = _sf(baseline.get("entry")), _sf(baseline.get("sl")), _sf(baseline.get("tp"))
        atr = max(_sf(baseline.get("atr"), 0.0), 1e-9)

        entry1, sl1, tp1 = _sf(ex.get("entry")), _sf(ex.get("sl")), _sf(ex.get("tp"))
        rr1 = _sf(ex.get("rr"), 0.0)

        if not _sanity(direction, entry1, sl1, tp1):
            return False, "sanity_failed"

        max_shift = _sf(constraints.get("entry_shift_max_atr", 0.20), 0.20) * atr
        if abs(entry1 - entry0) > (max_shift + 1e-9):
            return False, "entry_shift_exceed"

        sl_dist = abs(sl1 - entry1)
        sl_min = _sf(constraints.get("sl_atr_min", 1.20), 1.20) * atr
        sl_max = _sf(constraints.get("sl_atr_max", 1.80), 1.80) * atr
        if sl_dist < (sl_min - 1e-9) or sl_dist > (sl_max + 1e-9):
            return False, "sl_atr_range_fail"

        min_rr = _sf(constraints.get("min_rr", 1.50), 1.50)
        if rr1 < (min_rr - 1e-9):
            return False, "rr_below_min"

        return True, "ok"

    # -----------------------------
    # Stops / Freeze (new trade pre-check)
    # -----------------------------
    def _validate_stops_levels(self, symbol: str, direction: str, entry: float, sl: float, tp: float) -> Tuple[bool, str]:
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
            if (entry - sl) < min_dist or (tp - entry) < min_dist:
                return False, "stop_level_failed"
        elif direction == "SELL":
            if not (tp < entry < sl):
                return False, "sanity_failed_sell"
            if (sl - entry) < min_dist or (entry - tp) < min_dist:
                return False, "stop_level_failed"
        else:
            return False, "invalid_direction"

        if freeze_level_points > 0:
            tick = self._get_tick(symbol)
            freeze_dist = float(freeze_level_points) * point
            cur = float(tick.ask if direction == "BUY" else tick.bid)
            if abs(cur - entry) <= freeze_dist:
                return False, "freeze_level_too_close"

        return True, "ok"

    # -----------------------------
    # Order execution
    # -----------------------------
    def _place_market_order(self, symbol: str, direction: str, lot: float, sl: float, tp: float) -> Tuple[bool, str]:
        info = self._get_info(symbol)
        tick = self._get_tick(symbol)

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": info.trade_fill_mode,
            "comment": "HIM v2.6.2",
        }

        res = mt5.order_send(request)
        if res is None:
            return False, f"order_send_none err={mt5.last_error()}"
        ret = getattr(res, "retcode", None)
        if ret not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            return False, f"retcode={ret} comment={getattr(res,'comment','')}"
        return True, f"ok retcode={ret}"

    # -----------------------------
    # Main loop
    # -----------------------------
    def run_forever(self) -> None:
        logger.info("============================================================")
        logger.info("HIM Mentor Executor v2.6.2 | Hybrid SL Manager (C) | AI Spec v2 Mode D | Multi-order")
        logger.info("============================================================")

        while True:
            try:
                cfg_eff = self.hcfg.load_effective()
                symbol = str(cfg_eff.get("symbol", "GOLD"))

                # 1) Get engine package (current market context)
                pkg = self.engine.generate_signal_package()
                if not isinstance(pkg, dict):
                    time.sleep(1.0)
                    continue

                # 2) Hybrid SL manager runs every loop (manages OPEN positions)
                market_ctx = (pkg.get("context") or {})
                self._manage_positions(cfg_eff, symbol, market_ctx)

                # 3) New trade decision
                direction = str(pkg.get("direction", "NONE")).upper()
                if direction not in ("BUY", "SELL"):
                    time.sleep(0.5)
                    continue

                # Frozen gate: BOS + SuperTrend OK required
                if not bool(pkg.get("bos", False)) or not bool(pkg.get("supertrend_ok", False)):
                    time.sleep(0.5)
                    continue

                entry0 = _sf(pkg.get("entry_candidate"))
                sl0 = _sf(pkg.get("stop_candidate"))
                tp0 = _sf(pkg.get("tp_candidate"))
                rr0 = _sf(pkg.get("rr"), 0.0)
                if not _sanity(direction, entry0, sl0, tp0):
                    time.sleep(0.5)
                    continue

                policy = self._multi_order_policy(cfg_eff)
                sig = f"{symbol}:{direction}:{entry0:.5f}:{sl0:.5f}:{tp0:.5f}"
                if not self._execution_dedupe_ok(policy, sig=sig):
                    time.sleep(0.5)
                    continue

                atr = _sf((pkg.get("context") or {}).get("atr", 0.0), 0.0)
                if policy["allow_multiple_positions"]:
                    ok_stack, why = self._stacking_ok(symbol, entry0, atr, policy)
                    if not ok_stack:
                        time.sleep(0.5)
                        continue
                else:
                    if len(self._positions(symbol)) > 0:
                        time.sleep(0.5)
                        continue

                # AI (new trade)
                ai_pkg = self._build_ai_package(cfg_eff, pkg)
                ai_out = self.ai.evaluate(ai_pkg)

                baseline = ai_pkg.get("baseline") or {}
                constraints = ai_pkg.get("constraints") or {}

                ok_ai, why_ai = self._validate_ai_execution(ai_out, baseline, constraints)
                if not ok_ai:
                    time.sleep(0.5)
                    continue

                ex = ai_out.get("execution") or {}
                entry1 = _sf(ex.get("entry"))
                sl1 = _sf(ex.get("sl"))
                tp1 = _sf(ex.get("tp"))
                conf = _si(ex.get("conf"), 0)
                conf_th = _si(cfg_eff.get("confidence_threshold", 70), 70)

                ok_lv, why_lv = self._validate_stops_levels(symbol, direction, entry1, sl1, tp1)
                if not ok_lv:
                    time.sleep(0.5)
                    continue

                spread = self._spread_points(symbol)
                max_spread = _sf(((cfg_eff.get("execution_safety") or {}).get("max_spread_points", 35)), 35)
                if spread > max_spread:
                    time.sleep(0.5)
                    continue

                if conf < conf_th:
                    time.sleep(0.5)
                    continue

                lot = _sf(cfg_eff.get("lot", 0.01), 0.01)
                ok, msg = self._place_market_order(symbol, direction, lot, sl1, tp1)

                mentor = ai_out.get("mentor") or {}
                mentor_text = (
                    f"{mentor.get('headline','')}\n"
                    f"{mentor.get('explanation','')}\n"
                    f"{mentor.get('action_guidance','')}\n"
                    f"{mentor.get('confidence_reasoning','')}\n"
                    f"EXEC={ok} | {msg}"
                ).strip()

                try:
                    self.tg.send_text(mentor_text, event_type="trade")
                except Exception:
                    pass

                try:
                    self.tlog.log_trade(
                        {
                            "ts": time.time(),
                            "symbol": symbol,
                            "dir": direction,
                            "baseline": {"entry": entry0, "sl": sl0, "tp": tp0, "rr": rr0},
                            "ai": {"entry": entry1, "sl": sl1, "tp": tp1, "conf": conf},
                            "ok": ok,
                            "msg": msg,
                            "mentor_headline": mentor.get("headline", ""),
                            "risk_flags": ((ai_out.get("analysis") or {}).get("risk_flags") or []),
                        }
                    )
                except Exception:
                    pass

                time.sleep(0.8)

            except KeyboardInterrupt:
                logger.warning("Stopped by user.")
                return
            except Exception:
                logger.error("CRITICAL LOOP ERROR")
                logger.error(traceback.format_exc())
                time.sleep(2.0)


if __name__ == "__main__":
    MentorExecutor().run_forever()
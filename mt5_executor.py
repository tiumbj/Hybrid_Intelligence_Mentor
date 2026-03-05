"""
mt5_executor.py
Version: v1.3.1
Purpose: MT5 Execution Guard + AI Final Confirm Gate + Request Dedup + Execution Logging
         (Production Safe Execution for HIM)

========================================================
CHANGELOG (v1.3.1)
========================================================
- FIX: Directional stops validation (prevents MT5 retcode=10016 'Invalid stops')
  BUY requires: SL < price and TP > price
  SELL requires: SL > price and TP < price
- KEEP: v1.3.0 features (AI gate, dedup, JSONL logs, SLTP enforcement, ATR spread)
- Fail-closed: invalid stop side => SKIP("invalid_stops_side")

========================================================
INPUT CONTRACT (AI -> mt5_executor)
========================================================
Required minimal fields:
{
  "request_id": "unique_string",
  "decision": "BUY" | "SELL" | "HOLD" | ...,
  "plan": {"entry": <num>, "sl": <num>, "tp": <num>},
  "ai_confirm": {"approved": true|false, "reason": "...", "confidence": 0..1}
}
"""

from __future__ import annotations

import os
import time
import json
from typing import Any, Dict, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np

VERSION = "v1.3.1"

COOLDOWN_SECONDS = 30
MAGIC_NUMBER = 202603

SLTP_VERIFY_TIMEOUT_SEC = 3.0
SLTP_VERIFY_RETRY_INTERVAL_SEC = 0.25

DEDUP_STATE_FILE = ".execution_dedup_state.json"
EXEC_LOG_FILE = os.path.join("logs", "execution_orders.jsonl")


class MT5Executor:
    def __init__(
        self,
        symbol: str = "GOLD",
        lot: float = 0.01,
        max_spread_points: int = 80,
        atr_period: int = 14,
        atr_multiplier: float = 0.2,
        timeframe: int = mt5.TIMEFRAME_M5,
        deviation_min: int = 20,
        deviation_spread_mult: float = 2.0,
        sltp_verify_timeout_sec: float = SLTP_VERIFY_TIMEOUT_SEC,
        sltp_verify_retry_interval_sec: float = SLTP_VERIFY_RETRY_INTERVAL_SEC,
        dedup_state_file: str = DEDUP_STATE_FILE,
        exec_log_file: str = EXEC_LOG_FILE,
    ) -> None:
        self.symbol = symbol
        self.lot = lot

        self.max_spread_points = int(max_spread_points)
        self.atr_period = int(atr_period)
        self.atr_multiplier = float(atr_multiplier)
        self.timeframe = timeframe

        self.deviation_min = int(deviation_min)
        self.deviation_spread_mult = float(deviation_spread_mult)

        self.sltp_verify_timeout_sec = float(sltp_verify_timeout_sec)
        self.sltp_verify_retry_interval_sec = float(sltp_verify_retry_interval_sec)

        self.dedup_state_file = str(dedup_state_file)
        self.exec_log_file = str(exec_log_file)

        self.last_trade_time = 0.0

        os.makedirs(os.path.dirname(self.exec_log_file), exist_ok=True)

        self._dedup = self._load_dedup_state()

        if not mt5.initialize():
            raise RuntimeError("MT5 initialize failed")

    # -----------------------------
    # Helpers
    # -----------------------------

    @staticmethod
    def _is_number(x: Any) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    @staticmethod
    def _now() -> float:
        return time.time()

    @staticmethod
    def _round_to_digits(price: float, digits: int) -> float:
        return float(round(float(price), int(digits)))

    @staticmethod
    def _safe_json(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, default=str)

    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        line = self._safe_json(record)
        with open(self.exec_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # -----------------------------
    # Dedup State
    # -----------------------------

    def _load_dedup_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.dedup_state_file):
            return {"version": VERSION, "executed": {}}

        try:
            with open(self.dedup_state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"version": VERSION, "executed": {}}
            if "executed" not in data or not isinstance(data["executed"], dict):
                data["executed"] = {}
            return data
        except Exception:
            return {"version": VERSION, "executed": {}, "warning": "dedup_state_load_failed"}

    def _save_dedup_state(self) -> None:
        tmp = self.dedup_state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._dedup, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.dedup_state_file)

    def _dedup_is_done(self, request_id: str) -> bool:
        return str(request_id) in self._dedup.get("executed", {})

    def _dedup_mark_done(self, request_id: str, payload: Dict[str, Any]) -> None:
        self._dedup.setdefault("executed", {})[str(request_id)] = payload
        self._dedup["version"] = VERSION
        self._save_dedup_state()

    # -----------------------------
    # AI Confirm Gate
    # -----------------------------

    def ai_confirm_check(self, signal: Dict[str, Any]) -> Tuple[bool, str]:
        ai_confirm = signal.get("ai_confirm", None)
        if not isinstance(ai_confirm, dict):
            return False, "ai_confirm_missing"
        if ai_confirm.get("approved", None) is not True:
            return False, "ai_denied"
        return True, "ai_approved"

    # -----------------------------
    # Symbol / Environment Checks
    # -----------------------------

    def symbol_check(self) -> Tuple[bool, Any]:
        info = mt5.symbol_info(self.symbol)
        if info is None:
            return False, "symbol_not_found"

        if not info.visible:
            ok = mt5.symbol_select(self.symbol, True)
            if not ok:
                return False, "symbol_select_failed"

        if getattr(info, "trade_mode", None) == mt5.SYMBOL_TRADE_MODE_DISABLED:
            return False, "trade_disabled"

        return True, info

    def cooldown_check(self) -> Tuple[bool, Optional[str]]:
        if self._now() - self.last_trade_time < COOLDOWN_SECONDS:
            return False, "cooldown_active"
        return True, None

    def margin_check(self) -> Tuple[bool, Optional[str]]:
        acc = mt5.account_info()
        if acc is None:
            return False, "account_error"
        if acc.margin_free <= 0:
            return False, "no_margin"
        return True, None

    # -----------------------------
    # Duplicate / Pending Guards
    # -----------------------------

    def duplicate_position_check(self, direction: str) -> Tuple[bool, Optional[str]]:
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for p in positions:
                if getattr(p, "magic", None) != MAGIC_NUMBER:
                    continue
                if direction == "BUY" and p.type == 0:
                    return False, "duplicate_buy_magic"
                if direction == "SELL" and p.type == 1:
                    return False, "duplicate_sell_magic"
        return True, None

    def pending_orders_check(self) -> Tuple[bool, Optional[str]]:
        orders = mt5.orders_get(symbol=self.symbol)
        if not orders:
            return True, None
        for o in orders:
            if getattr(o, "magic", None) != MAGIC_NUMBER:
                continue
            return False, "pending_order_exists_magic"
        return True, None

    # -----------------------------
    # ATR-based Spread Filter
    # -----------------------------

    def get_atr_points(self, point: float) -> Optional[float]:
        n = self.atr_period + 1
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, n)
        if rates is None or len(rates) < n:
            return None

        try:
            highs = np.asarray(rates["high"], dtype=float)
            lows = np.asarray(rates["low"], dtype=float)
            closes = np.asarray(rates["close"], dtype=float)
        except Exception:
            return None

        if highs.size < n or lows.size < n or closes.size < n:
            return None

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )

        atr_price = float(np.mean(tr))
        if point <= 0:
            return None
        return atr_price / point

    def get_live_spread_points(self, info: Any) -> Tuple[Optional[int], Optional[str]]:
        point = float(getattr(info, "point", 0.0))
        if point <= 0:
            return None, "invalid_point"
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None, "tick_error"
        spread_price = float(tick.ask) - float(tick.bid)
        spread_points = int(round(spread_price / point))
        if spread_points < 0:
            spread_points = 0
        return spread_points, None

    def spread_check(self, info: Any) -> Tuple[bool, Optional[str]]:
        point = float(getattr(info, "point", 0.0))
        if point <= 0:
            return False, "invalid_point"

        spread_points, _ = self.get_live_spread_points(info)
        if spread_points is None:
            spread_points = int(getattr(info, "spread", 0))

        atr_points = self.get_atr_points(point)
        if atr_points is not None and atr_points > 0:
            dynamic_limit = atr_points * float(self.atr_multiplier)
            if spread_points > dynamic_limit:
                return False, f"spread>ATR_limit:{spread_points}>{dynamic_limit:.1f}"
            return True, None

        if spread_points > int(self.max_spread_points):
            return False, f"spread_too_high:{spread_points}"
        return True, None

    # -----------------------------
    # Stops / Levels (Directional + Distance)
    # -----------------------------

    def stops_check(self, direction: str, info: Any, exec_price: float, sl: float, tp: float) -> Tuple[bool, Optional[str]]:
        point = float(getattr(info, "point", 0.0))
        digits = int(getattr(info, "digits", 2))
        stops_level_points = int(getattr(info, "trade_stops_level", 0))
        stop_level_price = float(stops_level_points) * point

        if point <= 0:
            return False, "invalid_point"

        if not (self._is_number(exec_price) and self._is_number(sl) and self._is_number(tp)):
            return False, "invalid_plan_numbers"

        px = self._round_to_digits(float(exec_price), digits)
        slx = self._round_to_digits(float(sl), digits)
        tpx = self._round_to_digits(float(tp), digits)

        if slx <= 0 or tpx <= 0:
            return False, "sl_tp_must_be_positive"

        # Directional stop-side validation (KEY FIX for retcode=10016)
        if direction == "BUY":
            if not (slx < px and tpx > px):
                return False, "invalid_stops_side"
        elif direction == "SELL":
            if not (slx > px and tpx < px):
                return False, "invalid_stops_side"

        # Distance validation
        if abs(px - slx) < stop_level_price:
            return False, "sl_too_close"
        if abs(tpx - px) < stop_level_price:
            return False, "tp_too_close"

        return True, None

    # -----------------------------
    # SL/TP Post-Trade Enforcement (P0)
    # -----------------------------

    def _position_has_sltp(self, p: Any) -> bool:
        slv = float(getattr(p, "sl", 0.0) or 0.0)
        tpv = float(getattr(p, "tp", 0.0) or 0.0)
        return (slv > 0.0) and (tpv > 0.0)

    def _find_latest_our_position(self, direction: str) -> Optional[Any]:
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return None

        candidates = []
        for p in positions:
            if getattr(p, "magic", None) != MAGIC_NUMBER:
                continue
            if direction == "BUY" and p.type != 0:
                continue
            if direction == "SELL" and p.type != 1:
                continue
            candidates.append(p)

        if not candidates:
            return None

        candidates.sort(key=lambda x: getattr(x, "time", 0), reverse=True)
        return candidates[0]

    def _sltp_modify(self, position_ticket: int, sl: float, tp: float) -> Tuple[bool, str]:
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(position_ticket),
            "sl": float(sl),
            "tp": float(tp),
            "magic": MAGIC_NUMBER,
            "comment": "HIM_SLTP_FALLBACK",
        }
        res = mt5.order_send(req)
        if res is None:
            return False, "sltp_modify_none"
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"sltp_modify_fail:{res.retcode}"
        return True, "sltp_modified"

    def enforce_sltp_after_send(self, direction: str, sl: float, tp: float) -> Tuple[bool, str, Optional[int]]:
        deadline = self._now() + self.sltp_verify_timeout_sec

        position = None
        while self._now() < deadline:
            position = self._find_latest_our_position(direction)
            if position is not None:
                break
            time.sleep(self.sltp_verify_retry_interval_sec)

        if position is None:
            return False, "position_not_found_after_send", None

        pos_ticket = int(getattr(position, "ticket", 0) or 0)
        if pos_ticket <= 0:
            return False, "invalid_position_ticket", None

        if self._position_has_sltp(position):
            return True, "sltp_ok", pos_ticket

        ok, msg = self._sltp_modify(pos_ticket, sl, tp)
        if not ok:
            return False, msg, pos_ticket

        deadline2 = self._now() + self.sltp_verify_timeout_sec
        while self._now() < deadline2:
            pos2 = mt5.positions_get(ticket=pos_ticket)
            if pos2 and self._position_has_sltp(pos2[0]):
                return True, "sltp_enforced", pos_ticket
            time.sleep(self.sltp_verify_retry_interval_sec)

        return False, "sltp_verify_failed_after_modify", pos_ticket

    # -----------------------------
    # Execution
    # -----------------------------

    def execute(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        ts = self._now()

        request_id = signal.get("request_id", None)
        if not isinstance(request_id, str) or not request_id.strip():
            out = {"status": "SKIP", "reason": "missing_request_id"}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "status": out["status"], "reason": out["reason"]})
            return out
        request_id = request_id.strip()

        if self._dedup_is_done(request_id):
            out = {"status": "SKIP", "reason": "duplicate_request_id", "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out})
            return out

        ok_ai, ai_reason = self.ai_confirm_check(signal)
        if not ok_ai:
            out = {"status": "SKIP", "reason": ai_reason, "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out, "ai_confirm": signal.get("ai_confirm", None)})
            return out

        direction = str(signal.get("decision", "")).upper()
        if direction not in ("BUY", "SELL"):
            out = {"status": "SKIP", "reason": "no_trade_signal", "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out, "decision": direction})
            return out

        plan = signal.get("plan", {}) if isinstance(signal.get("plan", {}), dict) else {}
        entry = plan.get("entry", None)
        sl = plan.get("sl", None)
        tp = plan.get("tp", None)

        ok, info_or_reason = self.symbol_check()
        if not ok:
            out = {"status": "SKIP", "reason": str(info_or_reason), "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out})
            return out
        info = info_or_reason

        ok, reason = self.spread_check(info)
        if not ok:
            out = {"status": "SKIP", "reason": reason or "spread_blocked", "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out})
            return out

        ok, reason = self.cooldown_check()
        if not ok:
            out = {"status": "SKIP", "reason": reason or "cooldown_blocked", "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out})
            return out

        ok, reason = self.duplicate_position_check(direction)
        if not ok:
            out = {"status": "SKIP", "reason": reason or "duplicate_blocked", "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out})
            return out

        ok, reason = self.pending_orders_check()
        if not ok:
            out = {"status": "SKIP", "reason": reason or "pending_blocked", "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out})
            return out

        ok, reason = self.margin_check()
        if not ok:
            out = {"status": "SKIP", "reason": reason or "margin_blocked", "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out})
            return out

        if not (self._is_number(entry) and self._is_number(sl) and self._is_number(tp)):
            out = {"status": "SKIP", "reason": "plan_missing_or_invalid", "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out, "plan": plan})
            return out

        sl_f = float(sl)
        tp_f = float(tp)
        if sl_f <= 0 or tp_f <= 0:
            out = {"status": "SKIP", "reason": "plan_sl_tp_must_be_positive", "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out, "plan": plan})
            return out

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            out = {"status": "SKIP", "reason": "tick_error", "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out})
            return out

        digits = int(getattr(info, "digits", 2))
        point = float(getattr(info, "point", 0.0))
        if point <= 0:
            out = {"status": "SKIP", "reason": "invalid_point", "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out})
            return out

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)

        price = self._round_to_digits(price, digits)
        sl_f = self._round_to_digits(sl_f, digits)
        tp_f = self._round_to_digits(tp_f, digits)

        ok, reason = self.stops_check(direction, info, price, sl_f, tp_f)
        if not ok:
            out = {"status": "SKIP", "reason": reason or "stops_blocked", "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out, "price": price, "sl": sl_f, "tp": tp_f})
            return out

        spread_points, _ = self.get_live_spread_points(info)
        if spread_points is None:
            spread_points = int(getattr(info, "spread", 0))
        deviation = max(self.deviation_min, int(round(spread_points * self.deviation_spread_mult)))

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(self.lot),
            "type": order_type,
            "price": price,
            "sl": sl_f,
            "tp": tp_f,
            "deviation": int(deviation),
            "magic": MAGIC_NUMBER,
            "comment": "HIM_MT5_EXECUTOR",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            out = {"status": "SKIP", "reason": "order_send_none", "request_id": request_id}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out})
            return out

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            out = {"status": "SKIP", "reason": f"order_fail:{result.retcode}", "request_id": request_id, "mt5_comment": getattr(result, "comment", "")}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out})
            return out

        self.last_trade_time = self._now()

        ok_sltp, sltp_reason, pos_ticket = self.enforce_sltp_after_send(direction, sl_f, tp_f)
        if not ok_sltp:
            out = {"status": "ORDER_SENT_BUT_UNSAFE", "request_id": request_id, "price": price, "position_ticket": pos_ticket, "reason": sltp_reason}
            self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out, "ai_confirm": signal.get("ai_confirm", None), "plan": plan})
            return out

        out = {"status": "ORDER_SENT", "request_id": request_id, "price": price, "position_ticket": pos_ticket, "sltp": sltp_reason}
        self._append_jsonl({"ts": ts, "version": VERSION, "symbol": self.symbol, "request_id": request_id, **out, "ai_confirm": signal.get("ai_confirm", None), "plan": plan})
        self._dedup_mark_done(request_id, {"ts": ts, "status": out["status"], "position_ticket": pos_ticket, "price": price, "sltp": sltp_reason})
        return out

    def skip(self, reason: str) -> Dict[str, Any]:
        return {"status": "SKIP", "reason": reason}


if __name__ == "__main__":
    print(f"[MT5Executor] file={os.path.abspath(__file__)} version={VERSION}")

    executor = MT5Executor()

    example_signal = {
        "request_id": "TEST_DENY_001",
        "decision": "BUY",
        "plan": {"entry": 0, "sl": 5154.1, "tp": 5175.1},
        "ai_confirm": {"approved": False, "reason": "deny_test", "confidence": 0.0},
    }

    print("[MT5Executor] example_signal =", json.dumps(example_signal, ensure_ascii=False))
    result = executor.execute(example_signal)
    print(json.dumps(result, indent=2, ensure_ascii=False))
"""
mt5_executor.py
Version: v1.1.1
Purpose: MT5 Execution Guard with ATR-based Spread Filter (Production Safe Execution)

Project Rules
- Add module without changing existing production structure (does not touch mentor_executor.py)
- Does NOT modify strategy/logic (engine owns signal logic)
- Only applies execution safety checks before MT5 order_send

Safety Guards
- symbol visibility / trade_mode check
- spread guard (dynamic: ATR-based, fallback: max_spread)
- stop level validation (trade_stops_level)
- duplicate position guard
- cooldown guard
- margin check
"""

from __future__ import annotations

import time
import json
from typing import Any, Dict, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np

COOLDOWN_SECONDS = 30
MAGIC_NUMBER = 202603


class MT5Executor:
    def __init__(
        self,
        symbol: str = "GOLD",
        lot: float = 0.01,
        max_spread: int = 80,
        atr_period: int = 14,
        atr_multiplier: float = 0.2,
        timeframe: int = mt5.TIMEFRAME_M5,
    ) -> None:
        self.symbol = symbol
        self.lot = lot
        self.max_spread = max_spread
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.timeframe = timeframe

        self.last_trade_time = 0.0

        if not mt5.initialize():
            raise RuntimeError("MT5 initialize failed")

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    @staticmethod
    def _is_number(x: Any) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    # --------------------------------------------------
    # Symbol / Environment Checks
    # --------------------------------------------------

    def symbol_check(self) -> Tuple[bool, Any]:
        info = mt5.symbol_info(self.symbol)
        if info is None:
            return False, "symbol_not_found"

        if not info.visible:
            ok = mt5.symbol_select(self.symbol, True)
            if not ok:
                return False, "symbol_select_failed"

        # MetaTrader5 python: trade permission via trade_mode
        if getattr(info, "trade_mode", None) == mt5.SYMBOL_TRADE_MODE_DISABLED:
            return False, "trade_disabled"

        return True, info

    def cooldown_check(self) -> Tuple[bool, Optional[str]]:
        if time.time() - self.last_trade_time < COOLDOWN_SECONDS:
            return False, "cooldown_active"
        return True, None

    def duplicate_position_check(self, direction: str) -> Tuple[bool, Optional[str]]:
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for p in positions:
                # p.type: 0=BUY, 1=SELL
                if direction == "BUY" and p.type == 0:
                    return False, "duplicate_buy"
                if direction == "SELL" and p.type == 1:
                    return False, "duplicate_sell"
        return True, None

    def margin_check(self) -> Tuple[bool, Optional[str]]:
        acc = mt5.account_info()
        if acc is None:
            return False, "account_error"
        if acc.margin_free <= 0:
            return False, "no_margin"
        return True, None

    # --------------------------------------------------
    # ATR-based Spread Filter
    # --------------------------------------------------

    def get_atr_points(self, point: float) -> Optional[float]:
        """
        Returns ATR in 'points' (not price).
        Uses simple ATR mean of True Range over atr_period.
        """
        n = self.atr_period + 1
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, n)

        if rates is None:
            return None
        if len(rates) < n:
            return None

        # rates is numpy structured array -> use field access
        try:
            highs = np.asarray(rates["high"], dtype=float)
            lows = np.asarray(rates["low"], dtype=float)
            closes = np.asarray(rates["close"], dtype=float)
        except Exception:
            return None

        if highs.size < n or lows.size < n or closes.size < n:
            return None

        # True Range for bars 1..n-1 using prev close
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

    def spread_check(self, info: Any) -> Tuple[bool, Optional[str]]:
        spread_points = int(getattr(info, "spread", 0))
        point = float(getattr(info, "point", 0.0))

        # Dynamic ATR-based threshold
        atr_points = self.get_atr_points(point) if point > 0 else None
        if atr_points is not None and atr_points > 0:
            dynamic_limit = atr_points * float(self.atr_multiplier)
            if spread_points > dynamic_limit:
                return False, f"spread>ATR_limit:{spread_points}>{dynamic_limit:.1f}"
            return True, None

        # Fallback fixed threshold
        if spread_points > int(self.max_spread):
            return False, f"spread_too_high:{spread_points}"
        return True, None

    # --------------------------------------------------
    # MT5 Constraints
    # --------------------------------------------------

    def stops_check(self, info: Any, entry: float, sl: float, tp: float) -> Tuple[bool, Optional[str]]:
        point = float(getattr(info, "point", 0.0))
        stops_level_points = int(getattr(info, "trade_stops_level", 0))
        stop_level = stops_level_points * point

        if point <= 0:
            return False, "invalid_point"

        if not (self._is_number(entry) and self._is_number(sl) and self._is_number(tp)):
            return False, "invalid_plan_numbers"

        # Require minimum distance from entry
        if abs(entry - sl) < stop_level:
            return False, "sl_too_close"
        if abs(tp - entry) < stop_level:
            return False, "tp_too_close"

        return True, None

    # --------------------------------------------------
    # Execution
    # --------------------------------------------------

    def execute(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects:
        signal["decision"] in {"BUY","SELL",None}
        signal["plan"]["entry"], ["sl"], ["tp"]
        """
        direction = str(signal.get("decision", "")).upper()
        plan = signal.get("plan", {}) if isinstance(signal.get("plan", {}), dict) else {}

        entry = plan.get("entry", None)
        sl = plan.get("sl", None)
        tp = plan.get("tp", None)

        ok, info_or_reason = self.symbol_check()
        if not ok:
            return self.skip(str(info_or_reason))

        info = info_or_reason

        ok, reason = self.spread_check(info)
        if not ok:
            return self.skip(reason or "spread_blocked")

        ok, reason = self.cooldown_check()
        if not ok:
            return self.skip(reason or "cooldown_blocked")

        ok, reason = self.duplicate_position_check(direction)
        if not ok:
            return self.skip(reason or "duplicate_blocked")

        ok, reason = self.margin_check()
        if not ok:
            return self.skip(reason or "margin_blocked")

        # If no actionable decision -> skip (does not change strategy, just avoids sending)
        if direction not in ("BUY", "SELL"):
            return self.skip("no_trade_signal")

        # Stops validation needs numeric plan
        if not (self._is_number(entry) and self._is_number(sl) and self._is_number(tp)):
            return self.skip("plan_missing_or_invalid")

        entry_f = float(entry)
        sl_f = float(sl)
        tp_f = float(tp)

        ok, reason = self.stops_check(info, entry_f, sl_f, tp_f)
        if not ok:
            return self.skip(reason or "stops_blocked")

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return self.skip("tick_error")

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(self.lot),
            "type": order_type,
            "price": price,
            "sl": sl_f,
            "tp": tp_f,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "HIM_MT5_EXECUTOR",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return self.skip("order_send_none")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return self.skip(f"order_fail:{result.retcode}")

        self.last_trade_time = time.time()

        return {
            "status": "ORDER_SENT",
            "ticket": result.order,
            "price": price,
        }

    def skip(self, reason: str) -> Dict[str, Any]:
        return {"status": "SKIP", "reason": reason}


# ---------------------------------------------------------
# TEST MODE (does not place order due to no_trade_signal)
# ---------------------------------------------------------

if __name__ == "__main__":
    executor = MT5Executor()

    # Example signal: will SKIP unless you fill a valid plan + decision
    example_signal = {
        "decision": "BUY",
        "plan": {"entry": 0, "sl": 0, "tp": 0},
    }

    result = executor.execute(example_signal)
    print(json.dumps(result, indent=2, ensure_ascii=False))
"""
Hybrid Intelligence Mentor (HIM) Trading Engine
Version: 2.9.0

Changelog:
- 2.9.0 (2026-02-26):
  - ADD: Continuation Mode (No BOS) driven by config:
      mode == "continuation" AND continuation.enabled == True
    Rules:
      - Require MTF + LTF aligned with direction_bias (user choice #1)
      - Allow HTF ranging, but block if HTF is opposite (require_htf_not_opposite=True)
      - Require supertrend_ok, proximity, rr>=min_rr
  - KEEP: Option C (BOS+Proximity+Retest) for precision/balanced/frequent modes
  - KEEP: BOS confirm uses ATR buffer (breakout.confirm_buffer_atr)
  - KEEP: contract: top-level bos + supertrend_ok booleans always
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import MetaTrader5 as mt5

from config_resolver import resolve_effective_config


class TradingEngine:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.cfg = self.load_config()

    @staticmethod
    def safe_float(v: Any, default: float = 0.0) -> float:
        if v is None:
            return float(default)
        try:
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def safe_int(v: Any, default: int = 0) -> int:
        if v is None:
            return int(default)
        try:
            return int(v)
        except Exception:
            return int(default)

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, x)))

    def load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if not isinstance(cfg, dict):
                return {}
            return resolve_effective_config(cfg)
        except Exception:
            return {}

    def reload_config(self) -> Dict[str, Any]:
        self.cfg = self.load_config()
        return self.cfg

    def ensure_mt5(self) -> Tuple[bool, str]:
        try:
            if mt5.terminal_info() is not None:
                return True, "already_initialized"
            ok = mt5.initialize()
            if not ok:
                return False, f"initialize_failed: {mt5.last_error()}"
            return True, "initialized"
        except Exception as e:
            return False, f"initialize_exception: {e}"

    def get_tick(self, symbol: str):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"tick_none: {symbol} last_error={mt5.last_error()}")
        return tick

    def tf(self, name: str):
        mapping = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        key = str(name).upper().strip()
        return mapping.get(key, mt5.TIMEFRAME_M15)

    def get_data(self, symbol: str, timeframe, bars: int):
        ok, msg = self.ensure_mt5()
        if not ok:
            raise RuntimeError(msg)

        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select_failed: {symbol} last_error={mt5.last_error()}")

        bars = int(bars)

        def fetch_once():
            return mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

        rates = fetch_once()
        if rates is None or len(rates) == 0:
            time.sleep(0.2)
            rates = fetch_once()

        if rates is None:
            raise RuntimeError(f"copy_rates_none: {symbol} tf={timeframe} bars={bars} last_error={mt5.last_error()}")
        if len(rates) < 200:
            raise RuntimeError(f"not_enough_bars: got={len(rates)} need>=200 symbol={symbol}")
        return rates

    def atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        period = max(1, int(period))
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        out = np.zeros(len(close), dtype=float)
        alpha = 1.0 / period
        for i in range(1, len(close)):
            out[i] = out[i - 1] * (1 - alpha) + tr[i] * alpha
        return out

    def supertrend(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, mult: float):
        n = len(close)
        period = max(1, int(period))
        mult = float(max(mult, 0.1))

        atr = self.atr(high, low, close, period)
        hl2 = (high + low) / 2.0
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr

        f_upper = np.copy(upper)
        f_lower = np.copy(lower)

        dir_arr = np.ones(n, dtype=int)
        st = np.zeros(n, dtype=float)

        for i in range(1, n):
            if upper[i] < f_upper[i - 1] or close[i - 1] > f_upper[i - 1]:
                f_upper[i] = upper[i]
            else:
                f_upper[i] = f_upper[i - 1]

            if lower[i] > f_lower[i - 1] or close[i - 1] < f_lower[i - 1]:
                f_lower[i] = lower[i]
            else:
                f_lower[i] = f_lower[i - 1]

            if dir_arr[i - 1] == 1:
                dir_arr[i] = -1 if close[i] < f_lower[i] else 1
            else:
                dir_arr[i] = 1 if close[i] > f_upper[i] else -1

            st[i] = f_lower[i] if dir_arr[i] == 1 else f_upper[i]

        st[0] = hl2[0]
        return st, dir_arr

    def structure(self, data, sens: int) -> Tuple[str, Optional[float], Optional[float]]:
        high = data["high"]
        low = data["low"]
        sens = max(1, int(sens))

        piv_hi = []
        piv_lo = []
        for i in range(sens, len(high) - sens):
            if high[i] == max(high[i - sens : i + sens + 1]):
                piv_hi.append(float(high[i]))
            if low[i] == min(low[i - sens : i + sens + 1]):
                piv_lo.append(float(low[i]))

        last_hi = piv_hi[-1] if piv_hi else None
        last_lo = piv_lo[-1] if piv_lo else None

        trend = "ranging"
        if len(piv_hi) >= 2 and len(piv_lo) >= 2:
            if piv_hi[-1] > piv_hi[-2] and piv_lo[-1] > piv_lo[-2]:
                trend = "bullish"
            elif piv_hi[-1] < piv_hi[-2] and piv_lo[-1] < piv_lo[-2]:
                trend = "bearish"

        return trend, last_hi, last_lo

    def proximity_score(self, dist_buy: Optional[float], dist_sell: Optional[float], threshold_points: float):
        thr = max(self.safe_float(threshold_points, 0.0), 1e-9)
        best: Optional[float] = None
        side: Optional[str] = None

        if dist_buy is not None:
            db = float(dist_buy)
            if db >= 0:
                best = db
                side = "BUY"

        if dist_sell is not None:
            ds = float(dist_sell)
            if ds >= 0:
                if best is None or ds < best:
                    best = ds
                    side = "SELL"

        if best is None:
            return 0.0, None, None

        score = 100.0 * max(0.0, min(1.0, (thr - best) / thr))
        return float(score), side, float(best)

    def _get_breakout_knobs(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        b = cfg.get("breakout", {}) or {}
        confirm_buffer_atr = self.safe_float(b.get("confirm_buffer_atr"), 0.05)
        require_retest = bool(b.get("require_retest", True))
        retest_band_atr = self.safe_float(b.get("retest_band_atr"), 0.30)
        prox_thr_atr = self.safe_float(b.get("proximity_threshold_atr"), 1.50)
        prox_min_score = self.safe_float(b.get("proximity_min_score"), 10.0)
        return {
            "confirm_buffer_atr": max(confirm_buffer_atr, 0.0),
            "require_retest": require_retest,
            "retest_band_atr": max(retest_band_atr, 0.0),
            "proximity_threshold_atr": max(prox_thr_atr, 0.0),
            "proximity_min_score": self.clamp(prox_min_score, 0.0, 100.0),
        }

    def _get_continuation_knobs(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        c = cfg.get("continuation", {}) or {}
        enabled = bool(c.get("enabled", False))
        req_align = bool(c.get("require_mtf_ltf_align", True))
        req_htf_not_opposite = bool(c.get("require_htf_not_opposite", True))
        prox_thr_atr = self.safe_float(c.get("proximity_threshold_atr"), 1.80)
        prox_min_score = float(self.safe_int(c.get("proximity_min_score"), 20))
        return {
            "enabled": enabled,
            "require_mtf_ltf_align": req_align,
            "require_htf_not_opposite": req_htf_not_opposite,
            "proximity_threshold_atr": max(prox_thr_atr, 0.0),
            "proximity_min_score": self.clamp(float(prox_min_score), 0.0, 100.0),
        }

    def generate_signal_package(self) -> Dict[str, Any]:
        cfg = self.reload_config()

        symbol = str(cfg.get("symbol", "GOLD"))
        mode = str(cfg.get("mode", "balanced")).strip().lower()

        tf_cfg = cfg.get("timeframes", {}) or {}
        risk_cfg = cfg.get("risk", {}) or {}
        st_cfg = cfg.get("supertrend", {}) or {}
        sens_cfg = cfg.get("structure_sensitivity", {}) or {}

        atr_period = self.safe_int(risk_cfg.get("atr_period", 14), 14)
        atr_sl_mult = self.safe_float(risk_cfg.get("atr_sl_mult", 1.8), 1.8)
        min_rr = self.safe_float(cfg.get("min_rr", 2.0), 2.0)

        st_period = self.safe_int(st_cfg.get("period", 10), 10)
        st_mult = st_cfg.get("multiplier", st_cfg.get("mult", 3.0))
        st_mult = self.safe_float(st_mult, 3.0)

        htf = str(tf_cfg.get("htf", "H4"))
        mtf = str(tf_cfg.get("mtf", "H1"))
        ltf = str(tf_cfg.get("ltf", "M15"))

        sens_htf = self.safe_int(sens_cfg.get("htf", 5), 5)
        sens_mtf = self.safe_int(sens_cfg.get("mtf", 4), 4)
        sens_ltf = self.safe_int(sens_cfg.get("ltf", 3), 3)

        breakout_knobs = self._get_breakout_knobs(cfg)
        cont_knobs = self._get_continuation_knobs(cfg)

        blocked_reasons = []
        watch_state = "NONE"

        try:
            htf_data = self.get_data(symbol, self.tf(htf), 600)
            mtf_data = self.get_data(symbol, self.tf(mtf), 800)
            ltf_data = self.get_data(symbol, self.tf(ltf), 1200)
        except Exception as e:
            return {
                "symbol": symbol,
                "direction": "NONE",
                "entry_candidate": None,
                "stop_candidate": None,
                "tp_candidate": None,
                "rr": 0.0,
                "score": 0.0,
                "confidence_py": 0,
                "bos": False,
                "supertrend_ok": False,
                "context": {"blocked_by": "no_data", "error": str(e)},
            }

        htf_trend, _, _ = self.structure(htf_data, sens_htf)
        mtf_trend, _, _ = self.structure(mtf_data, sens_mtf)
        ltf_trend, bos_hi, bos_lo = self.structure(ltf_data, sens_ltf)

        # Bias owner (engine)
        direction_bias = "NONE"
        bias_source = "NO_CLEAR_BIAS"
        if htf_trend == "bullish":
            direction_bias = "BUY"
            bias_source = "HTF"
        elif htf_trend == "bearish":
            direction_bias = "SELL"
            bias_source = "HTF"
        elif mtf_trend == "bullish":
            direction_bias = "BUY"
            bias_source = "MTF_FALLBACK"
        elif mtf_trend == "bearish":
            direction_bias = "SELL"
            bias_source = "MTF_FALLBACK"
        else:
            blocked_reasons.append("no_clear_bias")

        close = np.asarray(ltf_data["close"], dtype=float)
        high = np.asarray(ltf_data["high"], dtype=float)
        low = np.asarray(ltf_data["low"], dtype=float)
        ltf_close = float(close[-1])

        atr_arr = self.atr(high, low, close, atr_period)
        atr_val = float(atr_arr[-1]) if len(atr_arr) else 0.0

        st_line, st_dir_arr = self.supertrend(high, low, close, st_period, st_mult)
        st_dir = "bullish" if int(st_dir_arr[-1]) == 1 else "bearish"
        st_value = float(st_line[-1])

        # SuperTrend gate relative to bias
        supertrend_ok = False
        if direction_bias == "BUY" and st_dir == "bullish":
            supertrend_ok = True
        elif direction_bias == "SELL" and st_dir == "bearish":
            supertrend_ok = True

        if direction_bias in ("BUY", "SELL") and not supertrend_ok:
            blocked_reasons.append("supertrend_conflict")

        # Distances to refs
        dist_buy = (float(bos_hi) - ltf_close) if bos_hi is not None else None
        dist_sell = (ltf_close - float(bos_lo)) if bos_lo is not None else None

        # Choose proximity threshold by mode:
        # - continuation uses continuation.proximity_threshold_atr
        # - otherwise uses breakout.proximity_threshold_atr
        prox_thr_atr = cont_knobs["proximity_threshold_atr"] if (mode == "continuation" and cont_knobs["enabled"]) else breakout_knobs["proximity_threshold_atr"]
        threshold_points = float(atr_val * prox_thr_atr) if (atr_val > 0 and prox_thr_atr > 0) else 1e-9
        threshold_points = max(threshold_points, 1e-9)

        prox_score, prox_side, prox_best_dist = self.proximity_score(dist_buy, dist_sell, threshold_points)

        # Watch state
        if prox_side in ("BUY", "SELL") and prox_score >= 10:
            watch_state = "WATCH_PROXIMITY"

        # BOS confirm (Option C)
        confirm_buffer_points = float(max(atr_val * breakout_knobs["confirm_buffer_atr"], 0.0)) if atr_val > 0 else 0.0

        breakout_state = "NONE"
        breakout_side = None
        breakout_overshoot_points = None

        buy_overshoot = float(ltf_close - (float(bos_hi) + confirm_buffer_points)) if bos_hi is not None else None
        sell_overshoot = float((float(bos_lo) - confirm_buffer_points) - ltf_close) if bos_lo is not None else None

        if buy_overshoot is not None and buy_overshoot > 0:
            breakout_state = "BREAKOUT_BUY_CONFIRMED"
            breakout_side = "BUY"
            breakout_overshoot_points = float(buy_overshoot)
        elif sell_overshoot is not None and sell_overshoot > 0:
            breakout_state = "BREAKOUT_SELL_CONFIRMED"
            breakout_side = "SELL"
            breakout_overshoot_points = float(sell_overshoot)

        bos = bool(breakout_side == direction_bias and breakout_state.startswith("BREAKOUT_"))

        # Retest (Option C)
        bos_ref = bos_hi if direction_bias == "BUY" else (bos_lo if direction_bias == "SELL" else None)
        retest_band = float(max(atr_val * breakout_knobs["retest_band_atr"], 1e-9)) if atr_val > 0 else 1e-9
        retest_ok = False
        if bos_ref is not None and direction_bias in ("BUY", "SELL"):
            retest_ok = bool(abs(ltf_close - float(bos_ref)) <= retest_band)

        # -------------------------
        # Mode decision
        # -------------------------
        direction_out = "NONE"

        # Common RR calc (only if we decide to trade)
        entry_candidate = None
        stop_candidate = None
        tp_candidate = None
        rr = 0.0

        # Guard: must have BUY/SELL bias
        if direction_bias not in ("BUY", "SELL"):
            blocked_reasons.append("direction_none")

        # Continuation Mode (No BOS)
        if mode == "continuation" and cont_knobs["enabled"]:
            # Rule #1: require MTF + LTF aligned with bias
            if cont_knobs["require_mtf_ltf_align"]:
                if not (mtf_trend == ("bearish" if direction_bias == "SELL" else "bullish") and
                        ltf_trend == ("bearish" if direction_bias == "SELL" else "bullish")):
                    blocked_reasons.append("no_mtf_ltf_align")

            # HTF not opposite (allow ranging)
            if cont_knobs["require_htf_not_opposite"]:
                if direction_bias == "SELL" and htf_trend == "bullish":
                    blocked_reasons.append("htf_opposite")
                if direction_bias == "BUY" and htf_trend == "bearish":
                    blocked_reasons.append("htf_opposite")

            # Proximity required (aligned side)
            if not (prox_side == direction_bias and prox_score >= cont_knobs["proximity_min_score"]):
                blocked_reasons.append("no_proximity")

            # supertrend already checked above; keep as requirement
            # rr requirement checked after candidate build

            blocked = len(blocked_reasons) > 0
            if not blocked:
                direction_out = direction_bias

        # Option C modes (default)
        else:
            # BOS required
            if direction_bias in ("BUY", "SELL") and not bos:
                blocked_reasons.append("no_bos")

            # Proximity required (aligned side)
            if direction_bias in ("BUY", "SELL"):
                if not (prox_side == direction_bias and prox_score >= breakout_knobs["proximity_min_score"]):
                    blocked_reasons.append("no_proximity")

            # Retest required
            if direction_bias in ("BUY", "SELL") and breakout_knobs["require_retest"]:
                if bos and not retest_ok:
                    blocked_reasons.append("no_retest")

            blocked = len(blocked_reasons) > 0
            if not blocked:
                direction_out = direction_bias

        # Build candidates if signal allowed
        if direction_out in ("BUY", "SELL"):
            tick = self.get_tick(symbol)
            entry_candidate = float(tick.ask if direction_out == "BUY" else tick.bid)

            sl_dist = float(max(atr_val * atr_sl_mult, 1e-9))
            if direction_out == "BUY":
                stop_candidate = float(entry_candidate - sl_dist)
                tp_candidate = float(entry_candidate + sl_dist * float(max(min_rr, 0.1)))
                risk = entry_candidate - stop_candidate
                reward = tp_candidate - entry_candidate
                rr = float(reward / risk) if risk > 1e-9 else 0.0
            else:
                stop_candidate = float(entry_candidate + sl_dist)
                tp_candidate = float(entry_candidate - sl_dist * float(max(min_rr, 0.1)))
                risk = stop_candidate - entry_candidate
                reward = entry_candidate - tp_candidate
                rr = float(reward / risk) if risk > 1e-9 else 0.0

            # RR gate
            if rr < min_rr:
                direction_out = "NONE"
                entry_candidate = stop_candidate = tp_candidate = None
                rr = 0.0
                blocked_reasons.append("rr_too_low")

        blocked_by = ",".join(blocked_reasons) if blocked_reasons else None

        return {
            "symbol": symbol,
            "direction": direction_out,
            "entry_candidate": entry_candidate,
            "stop_candidate": stop_candidate,
            "tp_candidate": tp_candidate,
            "rr": float(rr),
            "score": 0.0,
            "confidence_py": 0,
            "bos": bool(bos),
            "supertrend_ok": bool(supertrend_ok),
            "context": {
                "engine_version": "2.9.0",
                "mode": mode,
                "HTF_trend": htf_trend,
                "MTF_trend": mtf_trend,
                "LTF_trend": ltf_trend,
                "bias_source": bias_source,
                "direction_bias": direction_bias,

                "bos_ref_high": bos_hi,
                "bos_ref_low": bos_lo,
                "bos_ref": bos_ref,
                "distance_buy": dist_buy,
                "distance_sell": dist_sell,

                "atr": atr_val,
                "atr_period": atr_period,
                "atr_sl_mult": atr_sl_mult,
                "min_rr": min_rr,

                "breakout_confirm_buffer_atr": breakout_knobs["confirm_buffer_atr"],
                "breakout_confirm_buffer_points": confirm_buffer_points,
                "breakout_proximity_min_score": breakout_knobs["proximity_min_score"],
                "breakout_proximity_threshold_atr": breakout_knobs["proximity_threshold_atr"],

                "continuation_enabled": cont_knobs["enabled"],
                "continuation_require_mtf_ltf_align": cont_knobs["require_mtf_ltf_align"],
                "continuation_require_htf_not_opposite": cont_knobs["require_htf_not_opposite"],
                "continuation_proximity_threshold_atr": cont_knobs["proximity_threshold_atr"],
                "continuation_proximity_min_score": cont_knobs["proximity_min_score"],

                "breakout_state": breakout_state,
                "breakout_side": breakout_side,
                "breakout_overshoot_points": breakout_overshoot_points,

                "retest_required": bool(breakout_knobs["require_retest"]),
                "retest_band_atr": breakout_knobs["retest_band_atr"],
                "retest_band": retest_band,
                "retest_ok": retest_ok,

                "supertrend_period": st_period,
                "supertrend_mult": st_mult,
                "supertrend_dir": st_dir,
                "supertrend_value": st_value,
                "supertrend_ok": bool(supertrend_ok),

                "proximity_score": prox_score,
                "proximity_side": prox_side,
                "proximity_best_dist": prox_best_dist,
                "proximity_threshold_points": threshold_points,

                "watch_state": watch_state,
                "blocked_by": blocked_by,
                "timeframes": {"htf": htf, "mtf": mtf, "ltf": ltf},
                "structure_sensitivity": {"htf": sens_htf, "mtf": sens_mtf, "ltf": sens_ltf},
            },
        }


if __name__ == "__main__":
    if not mt5.initialize():
        print("MT5 init failed:", mt5.last_error())
        raise SystemExit(1)

    e = TradingEngine("config.json")
    pkg = e.generate_signal_package()
    print(json.dumps(pkg, indent=2, ensure_ascii=False))
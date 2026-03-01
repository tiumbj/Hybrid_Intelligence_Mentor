"""
Hybrid Intelligence Mentor (HIM) Trading Engine
Version: 3.0.0

Changelog:
- 3.0.0 (2026-02-28):
  - REWRITE: สร้างใหม่ทั้งหมดตามโครงสร้างเดิม
  - KEEP: ระบบ proximity_score_min สำหรับ sideway_scalp
  - KEEP: adaptive trigger (near_trigger_atr, allow_soft_trigger, rsi_soft_band)
  - KEEP: sideway_scalp NEUTRAL bias (bias_source="SIDEWAY_MODE")
  - KEEP: regime gate (ADX + BB width normalized by ATR)
  - KEEP: debug bag สำหรับวินิจฉัย rr_too_low/tick anomalies
  - IMPROVE: ปรับปรุงโครงสร้างโค้ดให้อ่านง่ายขึ้น
  - IMPROVE: เพิ่มความแม่นยำในการคำนวณ
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import MetaTrader5 as mt5

from config_resolver import resolve_effective_config


ENGINE_VERSION = "3.0.0"


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
                raw = json.load(f)
            return resolve_effective_config(raw)
        except Exception:
            return {}

    def reload_config(self) -> Dict[str, Any]:
        self.cfg = self.load_config()
        return self.cfg

    @staticmethod
    def ensure_mt5() -> None:
        if mt5.initialize():
            return
        time.sleep(0.2)
        if not mt5.initialize():
            raise RuntimeError("MT5 initialize failed")

    @staticmethod
    def get_tick(symbol: str):
        TradingEngine.ensure_mt5()
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise RuntimeError(f"no tick for symbol={symbol}")
        return t

    @staticmethod
    def tf(tf_str: str) -> int:
        t = (tf_str or "").strip().upper()
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M2": mt5.TIMEFRAME_M2,
            "M3": mt5.TIMEFRAME_M3,
            "M4": mt5.TIMEFRAME_M4,
            "M5": mt5.TIMEFRAME_M5,
            "M6": mt5.TIMEFRAME_M6,
            "M10": mt5.TIMEFRAME_M10,
            "M12": mt5.TIMEFRAME_M12,
            "M15": mt5.TIMEFRAME_M15,
            "M20": mt5.TIMEFRAME_M20,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H2": mt5.TIMEFRAME_H2,
            "H3": mt5.TIMEFRAME_H3,
            "H4": mt5.TIMEFRAME_H4,
            "H6": mt5.TIMEFRAME_H6,
            "H8": mt5.TIMEFRAME_H8,
            "H12": mt5.TIMEFRAME_H12,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        if t not in mapping:
            raise ValueError(f"unknown timeframe: {tf_str}")
        return mapping[t]

    def get_data(self, symbol: str, timeframe: int, n: int) -> Dict[str, np.ndarray]:
        TradingEngine.ensure_mt5()
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, int(n))
        if rates is None or len(rates) < 50:
            raise RuntimeError(f"no rates (symbol={symbol}, timeframe={timeframe})")
        return {
            "time": np.asarray(rates["time"], dtype=np.int64),
            "open": np.asarray(rates["open"], dtype=float),
            "high": np.asarray(rates["high"], dtype=float),
            "low": np.asarray(rates["low"], dtype=float),
            "close": np.asarray(rates["close"], dtype=float),
            "tick_volume": np.asarray(rates["tick_volume"], dtype=np.int64),
            "spread": np.asarray(rates["spread"], dtype=np.int64),
        }

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)
        n = len(c)
        out = np.full(n, np.nan, dtype=float)
        if n < period + 1 or period <= 0:
            return out

        tr = np.maximum.reduce(
            [
                h[1:] - l[1:],
                np.abs(h[1:] - c[:-1]),
                np.abs(l[1:] - c[:-1]),
            ]
        )
        out[period] = np.mean(tr[:period])
        for i in range(period + 1, n):
            out[i] = (out[i - 1] * (period - 1) + tr[i - 1]) / period
        return out

    @staticmethod
    def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 10, multiplier: float = 3.0):
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)

        atr = TradingEngine.atr(h, l, c, period)
        hl2 = (h + l) / 2.0
        upperband = hl2 + multiplier * atr
        lowerband = hl2 - multiplier * atr

        n = len(c)
        st = np.full(n, np.nan, dtype=float)
        direction = np.full(n, 0, dtype=int)

        for i in range(1, n):
            if np.isnan(atr[i]) or np.isnan(upperband[i]) or np.isnan(lowerband[i]):
                continue

            if np.isnan(st[i - 1]):
                st[i] = lowerband[i]
                direction[i] = 1
                continue

            prev_st = st[i - 1]
            prev_dir = direction[i - 1] if direction[i - 1] != 0 else 1

            fub = upperband[i] if (upperband[i] < upperband[i - 1] or c[i - 1] > upperband[i - 1]) else upperband[i - 1]
            flb = lowerband[i] if (lowerband[i] > lowerband[i - 1] or c[i - 1] < lowerband[i - 1]) else lowerband[i - 1]

            if prev_dir == 1:
                if c[i] <= fub:
                    direction[i] = -1
                    st[i] = fub
                else:
                    direction[i] = 1
                    st[i] = flb
            else:
                if c[i] >= flb:
                    direction[i] = 1
                    st[i] = flb
                else:
                    direction[i] = -1
                    st[i] = fub

            if np.isnan(st[i]):
                st[i] = prev_st

        direction = np.where(direction == 0, 1, direction)
        direction = np.where(direction > 0, 1, -1)
        return st, direction

    @staticmethod
    def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        c = np.asarray(close, dtype=float)
        n = len(c)
        out = np.full(n, np.nan, dtype=float)
        if n < period + 1 or period <= 0:
            return out

        delta = np.diff(c)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        avg_gain = np.full(n, np.nan, dtype=float)
        avg_loss = np.full(n, np.nan, dtype=float)

        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])

        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

        rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
        out = 100.0 - (100.0 / (1.0 + rs))
        return out

    @staticmethod
    def bollinger(close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = np.asarray(close, dtype=float)
        n = len(c)
        upper = np.full(n, np.nan, dtype=float)
        mid = np.full(n, np.nan, dtype=float)
        lower = np.full(n, np.nan, dtype=float)
        if n < period or period <= 1:
            return upper, mid, lower

        for i in range(period - 1, n):
            window = c[i - period + 1 : i + 1]
            mu = float(np.mean(window))
            sd = float(np.std(window, ddof=0))
            mid[i] = mu
            upper[i] = mu + sd * float(std_mult)
            lower[i] = mu - sd * float(std_mult)
        return upper, mid, lower

    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)
        n = len(c)
        out = np.full(n, np.nan, dtype=float)
        if n < period + 2 or period <= 0:
            return out

        up_move = h[1:] - h[:-1]
        down_move = l[:-1] - l[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = np.maximum.reduce(
            [
                h[1:] - l[1:],
                np.abs(h[1:] - c[:-1]),
                np.abs(l[1:] - c[:-1]),
            ]
        )

        atr = np.full(n, np.nan, dtype=float)
        p_dm = np.full(n, np.nan, dtype=float)
        m_dm = np.full(n, np.nan, dtype=float)

        atr[period] = np.sum(tr[:period])
        p_dm[period] = np.sum(plus_dm[:period])
        m_dm[period] = np.sum(minus_dm[:period])

        for i in range(period + 1, n):
            atr[i] = atr[i - 1] - (atr[i - 1] / period) + tr[i - 1]
            p_dm[i] = p_dm[i - 1] - (p_dm[i - 1] / period) + plus_dm[i - 1]
            m_dm[i] = m_dm[i - 1] - (m_dm[i - 1] / period) + minus_dm[i - 1]

        plus_di = 100.0 * (p_dm / np.where(atr == 0, np.nan, atr))
        minus_di = 100.0 * (m_dm / np.where(atr == 0, np.nan, atr))
        dx = 100.0 * (np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di)))

        start = period * 2
        if start < n and np.any(np.isfinite(dx[period + 1 : start + 1])):
            out[start] = np.nanmean(dx[period + 1 : start + 1])
            for i in range(start + 1, n):
                out[i] = (out[i - 1] * (period - 1) + dx[i]) / period
        return out

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

    def _get_sideway_knobs(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        c = cfg.get("sideway_scalp", {}) or {}
        adx_period = self.safe_int(c.get("adx_period", 14), 14)
        bb_period = self.safe_int(c.get("bb_period", 20), 20)
        rsi_period = self.safe_int(c.get("rsi_period", 14), 14)

        return {
            "enabled": bool(c.get("enabled", True)),
            "adx_period": max(adx_period, 2),
            "adx_max": max(self.safe_float(c.get("adx_max", 22.0), 22.0), 0.0),
            "bb_period": max(bb_period, 5),
            "bb_std": max(self.safe_float(c.get("bb_std", 2.0), 2.0), 0.1),
            "bb_width_atr_max": max(self.safe_float(c.get("bb_width_atr_max", 6.0), 6.0), 0.1),
            "rsi_period": max(rsi_period, 2),
            "rsi_overbought": self.clamp(self.safe_float(c.get("rsi_overbought", 70.0), 70.0), 50.0, 95.0),
            "rsi_oversold": self.clamp(self.safe_float(c.get("rsi_oversold", 30.0), 30.0), 5.0, 50.0),
            "require_confirmation": bool(c.get("require_confirmation", True)),
            "touch_buffer_atr": self.clamp(self.safe_float(c.get("touch_buffer_atr", 0.10), 0.10), 0.0, 0.50),
            "near_trigger_atr": self.clamp(self.safe_float(c.get("near_trigger_atr", 0.50), 0.50), 0.0, 2.0),
            "allow_soft_trigger": bool(c.get("allow_soft_trigger", True)),
            "rsi_soft_band": self.clamp(self.safe_float(c.get("rsi_soft_band", 10.0), 10.0), 0.0, 25.0),
            "proximity_window_atr": self.clamp(self.safe_float(c.get("proximity_window_atr", 1.00), 1.00), 0.10, 5.0),
            "proximity_score_min": self.clamp(self.safe_float(c.get("proximity_score_min", 0.70), 0.70), 0.0, 1.0),
        }

    def generate_signal_package(self) -> Dict[str, Any]:
        cfg = self.reload_config()

        symbol = str(cfg.get("symbol", "GOLD"))
        mode = str(cfg.get("mode", "balanced")).strip().lower()
        tf_cfg = cfg.get("timeframes", {}) or {}

        risk_cfg = cfg.get("risk", {}) or {}
        if not risk_cfg:
            legacy_atr = cfg.get("atr", {}) or {}
            risk_cfg = {"atr_period": legacy_atr.get("period", 14), "atr_sl_mult": legacy_atr.get("sl_mult", 1.8)}

        atr_period = self.safe_int(risk_cfg.get("atr_period", 14), 14)
        atr_sl_mult = self.safe_float(risk_cfg.get("atr_sl_mult", 1.8), 1.8)
        min_rr = self.safe_float(cfg.get("min_rr", 2.0), 2.0)

        st_cfg = cfg.get("supertrend", {}) or {}
        st_period = self.safe_int(st_cfg.get("period", 10), 10)
        st_mult = self.safe_float(st_cfg.get("multiplier", st_cfg.get("mult", 3.0)), 3.0)

        sens_cfg = cfg.get("structure_sensitivity", {}) or {}
        sens_htf = self.safe_int(sens_cfg.get("htf", 5), 5)
        sens_mtf = self.safe_int(sens_cfg.get("mtf", 4), 4)
        sens_ltf = self.safe_int(sens_cfg.get("ltf", 3), 3)

        htf = str(tf_cfg.get("htf", "H4"))
        mtf = str(tf_cfg.get("mtf", "H1"))
        ltf = str(tf_cfg.get("ltf", "M15"))

        blocked_reasons = []
        watch_state = "NONE"
        debug: Dict[str, Any] = {}

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
                "context": {"blocked_by": "no_data", "error": str(e), "engine_version": ENGINE_VERSION},
            }

        htf_trend, _, _ = self.structure(htf_data, sens_htf)
        mtf_trend, _, _ = self.structure(mtf_data, sens_mtf)
        ltf_trend, bos_hi, bos_lo = self.structure(ltf_data, sens_ltf)

        close = np.asarray(ltf_data["close"], dtype=float)
        high = np.asarray(ltf_data["high"], dtype=float)
        low = np.asarray(ltf_data["low"], dtype=float)
        open_ = np.asarray(ltf_data["open"], dtype=float)

        mtf_close = np.asarray(mtf_data["close"], dtype=float)
        mtf_high = np.asarray(mtf_data["high"], dtype=float)
        mtf_low = np.asarray(mtf_data["low"], dtype=float)

        ltf_close = float(close[-1])

        atr_arr = self.atr(high, low, close, atr_period)
        atr_val = float(atr_arr[-1]) if len(atr_arr) and np.isfinite(atr_arr[-1]) else 0.0

        st_line, st_dir_arr = self.supertrend(high, low, close, st_period, st_mult)
        st_dir = "bullish" if int(st_dir_arr[-1]) == 1 else "bearish"
        st_value = float(st_line[-1]) if len(st_line) else float("nan")

        if mode == "sideway_scalp":
            direction_bias = "NEUTRAL"
            bias_source = "SIDEWAY_MODE"
        else:
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

        supertrend_ok = True if mode == "sideway_scalp" else False
        if mode != "sideway_scalp":
            if direction_bias == "BUY" and st_dir == "bullish":
                supertrend_ok = True
            elif direction_bias == "SELL" and st_dir == "bearish":
                supertrend_ok = True
            else:
                if direction_bias in ("BUY", "SELL"):
                    blocked_reasons.append("supertrend_conflict")

        direction_out = "NONE"
        entry_candidate = None
        stop_candidate = None
        tp_candidate = None
        rr = 0.0

        sideway_ctx: Dict[str, Any] = {}

        distance_buy = float("nan")
        distance_sell = float("nan")
        proximity_side = "NONE"
        proximity_best_dist = float("nan")
        proximity_score = 0.0

        if mode == "sideway_scalp":
            k = self._get_sideway_knobs(cfg)

            adx_arr = self.adx(mtf_high, mtf_low, mtf_close, k["adx_period"])
            adx_val = float(adx_arr[-1]) if (len(adx_arr) and np.isfinite(adx_arr[-1])) else float("nan")

            bb_u, bb_m, bb_l = self.bollinger(close, k["bb_period"], k["bb_std"])
            bb_upper = float(bb_u[-1]) if (len(bb_u) and np.isfinite(bb_u[-1])) else float("nan")
            bb_mid = float(bb_m[-1]) if (len(bb_m) and np.isfinite(bb_m[-1])) else float("nan")
            bb_lower = float(bb_l[-1]) if (len(bb_l) and np.isfinite(bb_l[-1])) else float("nan")

            bb_width = float(bb_upper - bb_lower) if (np.isfinite(bb_upper) and np.isfinite(bb_lower)) else float("nan")
            bb_width_atr = float(bb_width / atr_val) if (atr_val > 0 and np.isfinite(bb_width)) else float("nan")

            rsi_arr = self.rsi(close, k["rsi_period"])
            rsi_val = float(rsi_arr[-1]) if (len(rsi_arr) and np.isfinite(rsi_arr[-1])) else float("nan")

            last_open = float(open_[-1])
            last_close = float(close[-1])
            prev_open = float(open_[-2]) if len(open_) >= 2 else last_open
            prev_close = float(close[-2]) if len(close) >= 2 else last_close

            bullish_reversal = (last_close > last_open) and (prev_close < prev_open)
            bearish_reversal = (last_close < last_open) and (prev_close > prev_open)

            if not (np.isfinite(adx_val) and adx_val <= k["adx_max"]):
                blocked_reasons.append("not_sideway_adx")
            if not (np.isfinite(bb_width_atr) and bb_width_atr <= k["bb_width_atr_max"]):
                blocked_reasons.append("not_sideway_bbwidth")

            touch_buffer_points = float(k["touch_buffer_atr"] * atr_val) if atr_val > 0 else 0.0
            near_points = float(k["near_trigger_atr"] * atr_val) if atr_val > 0 else 0.0

            buy_trigger_level = (bb_lower + touch_buffer_points) if np.isfinite(bb_lower) else float("nan")
            sell_trigger_level = (bb_upper - touch_buffer_points) if np.isfinite(bb_upper) else float("nan")

            if np.isfinite(buy_trigger_level):
                distance_buy = float(ltf_close - buy_trigger_level)
            if np.isfinite(sell_trigger_level):
                distance_sell = float(sell_trigger_level - ltf_close)

            d_buy_abs = abs(distance_buy) if np.isfinite(distance_buy) else float("inf")
            d_sell_abs = abs(distance_sell) if np.isfinite(distance_sell) else float("inf")

            if d_buy_abs < d_sell_abs:
                proximity_side = "BUY"
                proximity_best_dist = float(d_buy_abs)
            else:
                proximity_side = "SELL"
                proximity_best_dist = float(d_sell_abs)

            window = float(max(k["proximity_window_atr"] * atr_val, 1e-9))
            if np.isfinite(proximity_best_dist) and window > 0:
                proximity_score = float(max(0.0, 1.0 - (proximity_best_dist / window)))
            else:
                proximity_score = 0.0

            if len(blocked_reasons) == 0 and k["enabled"]:
                buy_touch = np.isfinite(buy_trigger_level) and (ltf_close <= buy_trigger_level)
                sell_touch = np.isfinite(sell_trigger_level) and (ltf_close >= sell_trigger_level)

                buy_soft = np.isfinite(buy_trigger_level) and (ltf_close <= (buy_trigger_level + near_points))
                sell_soft = np.isfinite(sell_trigger_level) and (ltf_close >= (sell_trigger_level - near_points))

                buy_confirm_hard = (np.isfinite(rsi_val) and rsi_val <= k["rsi_oversold"]) or bullish_reversal
                sell_confirm_hard = (np.isfinite(rsi_val) and rsi_val >= k["rsi_overbought"]) or bearish_reversal

                buy_soft_rsi = (np.isfinite(rsi_val) and rsi_val <= (k["rsi_oversold"] + k["rsi_soft_band"]))
                sell_soft_rsi = (np.isfinite(rsi_val) and rsi_val >= (k["rsi_overbought"] - k["rsi_soft_band"]))
                buy_confirm_soft = bullish_reversal or buy_soft_rsi
                sell_confirm_soft = bearish_reversal or sell_soft_rsi

                if buy_touch:
                    if (not k["require_confirmation"]) or buy_confirm_hard:
                        direction_out = "BUY"
                    else:
                        watch_state = "WATCH_PROXIMITY"
                        blocked_reasons.append("no_confirmation")
                elif sell_touch:
                    if (not k["require_confirmation"]) or sell_confirm_hard:
                        direction_out = "SELL"
                    else:
                        watch_state = "WATCH_PROXIMITY"
                        blocked_reasons.append("no_confirmation")
                else:
                    in_soft_zone = bool(buy_soft or sell_soft)
                    if in_soft_zone:
                        watch_state = "WATCH_PROXIMITY"

                        if proximity_score < float(k["proximity_score_min"]):
                            blocked_reasons.append("low_proximity_score")
                        else:
                            if k["allow_soft_trigger"]:
                                if buy_soft and ((not k["require_confirmation"]) or buy_confirm_soft):
                                    direction_out = "BUY"
                                elif sell_soft and ((not k["require_confirmation"]) or sell_confirm_soft):
                                    direction_out = "SELL"
                                else:
                                    blocked_reasons.append("no_confirmation")
                            else:
                                blocked_reasons.append("no_mean_reversion_trigger")
                    else:
                        blocked_reasons.append("no_mean_reversion_trigger")
            elif not k["enabled"]:
                blocked_reasons.append("sideway_disabled")

            if direction_out in ("BUY", "SELL"):
                tick = self.get_tick(symbol)

                debug["debug_direction_attempted"] = direction_out
                debug["debug_tick_bid"] = float(getattr(tick, "bid", float("nan")))
                debug["debug_tick_ask"] = float(getattr(tick, "ask", float("nan")))
                spread = float(getattr(tick, "ask", 0.0)) - float(getattr(tick, "bid", 0.0))
                debug["debug_spread"] = float(spread)

                entry_candidate = float(tick.ask if direction_out == "BUY" else tick.bid)

                sl_dist = float(max(atr_val * atr_sl_mult, 1e-9))
                debug["debug_sl_dist"] = float(sl_dist)
                debug["debug_min_rr_used"] = float(min_rr)
                debug["debug_min_rr_check"] = float(min_rr)

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

                debug["debug_entry"] = float(entry_candidate)
                debug["debug_stop"] = float(stop_candidate)
                debug["debug_tp"] = float(tp_candidate)
                debug["debug_rr"] = float(rr)

                eps = 1e-6
                if rr < (min_rr - eps):
                    debug["debug_rr_fail_reason"] = "rr < (min_rr - eps)"
                    direction_out = "NONE"
                    entry_candidate = stop_candidate = tp_candidate = None
                    rr = 0.0
                    blocked_reasons.append("rr_too_low")

            sideway_ctx = {
                "adx_period": k["adx_period"],
                "adx": adx_val,
                "adx_max": k["adx_max"],
                "bb_period": k["bb_period"],
                "bb_std": k["bb_std"],
                "bb_upper": bb_upper,
                "bb_mid": bb_mid,
                "bb_lower": bb_lower,
                "bb_width": bb_width,
                "bb_width_atr": bb_width_atr,
                "bb_width_atr_max": k["bb_width_atr_max"],
                "rsi_period": k["rsi_period"],
                "rsi": rsi_val,
                "rsi_overbought": k["rsi_overbought"],
                "rsi_oversold": k["rsi_oversold"],
                "bullish_reversal": bool(bullish_reversal),
                "bearish_reversal": bool(bearish_reversal),
                "touch_buffer_atr": k["touch_buffer_atr"],
                "touch_buffer_points": touch_buffer_points,
                "near_trigger_atr": k["near_trigger_atr"],
                "near_trigger_points": near_points,
                "allow_soft_trigger": bool(k["allow_soft_trigger"]),
                "require_confirmation": bool(k["require_confirmation"]),
                "rsi_soft_band": k["rsi_soft_band"],
                "proximity_window_atr": k["proximity_window_atr"],
                "proximity_score_min": k["proximity_score_min"],
                "buy_trigger_level": buy_trigger_level,
                "sell_trigger_level": sell_trigger_level,
                "distance_buy": distance_buy,
                "distance_sell": distance_sell,
                "proximity_side": proximity_side,
                "proximity_best_dist": proximity_best_dist,
                "proximity_score": proximity_score,
            }

        blocked_by = ",".join(blocked_reasons) if blocked_reasons else None

        if debug:
            sideway_ctx["debug"] = debug

        return {
            "symbol": symbol,
            "direction": direction_out,
            "entry_candidate": entry_candidate,
            "stop_candidate": stop_candidate,
            "tp_candidate": tp_candidate,
            "rr": float(rr),
            "score": float(proximity_score) if mode == "sideway_scalp" else 0.0,
            "confidence_py": 0,
            "bos": False,
            "supertrend_ok": bool(supertrend_ok),
            "context": {
                "engine_version": ENGINE_VERSION,
                "mode": mode,
                "HTF_trend": htf_trend,
                "MTF_trend": mtf_trend,
                "LTF_trend": ltf_trend,
                "bias_source": bias_source,
                "direction_bias": direction_bias,
                "watch_state": watch_state,
                "supertrend_dir": st_dir,
                "supertrend_value": st_value,
                "bos_ref_high": bos_hi,
                "bos_ref_low": bos_lo,
                "atr": atr_val,
                "atr_period": atr_period,
                "atr_sl_mult": atr_sl_mult,
                "min_rr": min_rr,
                **sideway_ctx,
                "blocked_by": blocked_by,
            },
        }
    
    

# ==================== HELPER FUNCTIONS ====================

def create_engine(config_path: str = "config.json") -> TradingEngine:
    """
    สร้าง TradingEngine instance
    
    Args:
        config_path: path ไปยังไฟล์ config.json
        
    Returns:
        TradingEngine instance
    """
    return TradingEngine(config_path=config_path)


def get_signal(engine: TradingEngine) -> Dict[str, Any]:
    """
    ดึงสัญญาณการเทรดจาก engine
    
    Args:
        engine: TradingEngine instance
        
    Returns:
        Dictionary ที่มีข้อมูลสัญญาณครบถ้วน
    """
    return engine.generate_signal_package()


def format_signal_summary(signal: Dict[str, Any]) -> str:
    """
    แปลงสัญญาณเป็นข้อความสรุปที่อ่านง่าย
    
    Args:
        signal: signal package จาก generate_signal_package()
        
    Returns:
        ข้อความสรุปสัญญาณ
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"TRADING SIGNAL - {signal.get('symbol', 'N/A')}")
    lines.append("=" * 60)
    
    direction = signal.get("direction", "NONE")
    lines.append(f"Direction: {direction}")
    
    if direction in ("BUY", "SELL"):
        entry = signal.get("entry_candidate")
        stop = signal.get("stop_candidate")
        tp = signal.get("tp_candidate")
        rr = signal.get("rr", 0.0)
        
        lines.append(f"Entry: {entry:.5f}" if entry else "Entry: N/A")
        lines.append(f"Stop Loss: {stop:.5f}" if stop else "Stop Loss: N/A")
        lines.append(f"Take Profit: {tp:.5f}" if tp else "Take Profit: N/A")
        lines.append(f"Risk/Reward: {rr:.2f}")
        lines.append(f"Score: {signal.get('score', 0.0):.3f}")
    
    ctx = signal.get("context", {})
    lines.append(f"\nEngine Version: {ctx.get('engine_version', 'N/A')}")
    lines.append(f"Mode: {ctx.get('mode', 'N/A')}")
    lines.append(f"HTF Trend: {ctx.get('HTF_trend', 'N/A')}")
    lines.append(f"MTF Trend: {ctx.get('MTF_trend', 'N/A')}")
    lines.append(f"LTF Trend: {ctx.get('LTF_trend', 'N/A')}")
    lines.append(f"SuperTrend: {ctx.get('supertrend_dir', 'N/A')} (OK: {signal.get('supertrend_ok', False)})")
    
    blocked = ctx.get("blocked_by")
    if blocked:
        lines.append(f"\n⚠️  BLOCKED BY: {blocked}")
    
    watch = ctx.get("watch_state", "NONE")
    if watch != "NONE":
        lines.append(f"📊 Watch State: {watch}")
    
    # Sideway scalp details
    if ctx.get("mode") == "sideway_scalp":
        lines.append("\n--- SIDEWAY SCALP DETAILS ---")
        lines.append(f"ADX: {ctx.get('adx', float('nan')):.2f} (max: {ctx.get('adx_max', 0):.2f})")
        lines.append(f"BB Width/ATR: {ctx.get('bb_width_atr', float('nan')):.2f} (max: {ctx.get('bb_width_atr_max', 0):.2f})")
        lines.append(f"RSI: {ctx.get('rsi', float('nan')):.2f}")
        lines.append(f"Proximity Score: {ctx.get('proximity_score', 0.0):.3f} (min: {ctx.get('proximity_score_min', 0.0):.2f})")
        lines.append(f"Proximity Side: {ctx.get('proximity_side', 'NONE')}")
        lines.append(f"Distance to Trigger: {ctx.get('proximity_best_dist', float('nan')):.5f}")
        
        if ctx.get("bullish_reversal"):
            lines.append("🔄 Bullish Reversal Detected")
        if ctx.get("bearish_reversal"):
            lines.append("🔄 Bearish Reversal Detected")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def validate_signal(signal: Dict[str, Any]) -> Tuple[bool, str]:
    """
    ตรวจสอบความถูกต้องของสัญญาณก่อนเทรด
    
    Args:
        signal: signal package จาก generate_signal_package()
        
    Returns:
        (is_valid, message) - True ถ้าสัญญาณพร้อมเทรด, False ถ้ายังไม่พร้อม พร้อมข้อความอธิบาย
    """
    direction = signal.get("direction", "NONE")
    
    if direction == "NONE":
        ctx = signal.get("context", {})
        blocked = ctx.get("blocked_by", "unknown")
        return False, f"No trade signal. Blocked by: {blocked}"
    
    if direction not in ("BUY", "SELL"):
        return False, f"Invalid direction: {direction}"
    
    entry = signal.get("entry_candidate")
    stop = signal.get("stop_candidate")
    tp = signal.get("tp_candidate")
    
    if entry is None or stop is None or tp is None:
        return False, "Missing entry, stop loss, or take profit levels"
    
    rr = signal.get("rr", 0.0)
    ctx = signal.get("context", {})
    min_rr = ctx.get("min_rr", 2.0)
    
    if rr < min_rr:
        return False, f"Risk/Reward too low: {rr:.2f} < {min_rr:.2f}"
    
    if not signal.get("supertrend_ok", False):
        mode = ctx.get("mode", "")
        if mode != "sideway_scalp":
            return False, "SuperTrend conflict with trade direction"
    
    return True, f"Valid {direction} signal with R:R {rr:.2f}"


def get_trade_params(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    แปลงสัญญาณเป็นพารามิเตอร์สำหรับส่งออเดอร์
    
    Args:
        signal: signal package จาก generate_signal_package()
        
    Returns:
        Dictionary ที่มีพารามิเตอร์การเทรด หรือ None ถ้าสัญญาณไม่ถูกต้อง
    """
    is_valid, msg = validate_signal(signal)
    
    if not is_valid:
        return None
    
    direction = signal.get("direction")
    entry = signal.get("entry_candidate")
    stop = signal.get("stop_candidate")
    tp = signal.get("tp_candidate")
    symbol = signal.get("symbol")
    
    return {
        "symbol": symbol,
        "action": direction,
        "entry_price": entry,
        "stop_loss": stop,
        "take_profit": tp,
        "risk_reward": signal.get("rr", 0.0),
        "score": signal.get("score", 0.0),
        "context": signal.get("context", {}),
    }


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    """
    ตัวอย่างการใช้งาน Trading Engine
    """
    print("Initializing HIM Trading Engine v3.0.0...")
    print("-" * 60)
    
    try:
        # สร้าง engine
        engine = create_engine("config.json")
        print("✓ Engine initialized successfully")
        
        # ดึงสัญญาณ
        print("\nGenerating signal package...")
        signal = get_signal(engine)
        
        # แสดงสรุปสัญญาณ
        print("\n" + format_signal_summary(signal))
        
        # ตรวจสอบความถูกต้อง
        is_valid, validation_msg = validate_signal(signal)
        print(f"\nValidation: {validation_msg}")
        
        # ถ้าสัญญาณถูกต้อง แสดงพารามิเตอร์การเทรด
        if is_valid:
            trade_params = get_trade_params(signal)
            if trade_params:
                print("\n📈 READY TO TRADE")
                print(f"Action: {trade_params['action']}")
                print(f"Symbol: {trade_params['symbol']}")
                print(f"Entry: {trade_params['entry_price']:.5f}")
                print(f"Stop Loss: {trade_params['stop_loss']:.5f}")
                print(f"Take Profit: {trade_params['take_profit']:.5f}")
                print(f"R:R Ratio: {trade_params['risk_reward']:.2f}")
        else:
            print("\n⏸️  NO TRADE - Waiting for valid signal")
        
        # แสดง raw signal data (สำหรับ debug)
        print("\n" + "=" * 60)
        print("RAW SIGNAL DATA (for debugging):")
        print("=" * 60)
        import json
        print(json.dumps(signal, indent=2, default=str))
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Engine execution completed")
    print("=" * 60)

    

# ==================== ADVANCED FEATURES ====================

class SignalHistory:
    """
    เก็บประวัติสัญญาณการเทรด
    """
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.history = []
    
    def add_signal(self, signal: Dict[str, Any]) -> None:
        """เพิ่มสัญญาณลงในประวัติ"""
        timestamp = time.time()
        signal_with_time = {
            "timestamp": timestamp,
            "signal": signal
        }
        self.history.append(signal_with_time)
        
        # จำกัดจำนวนประวัติ
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_last_signal(self) -> Optional[Dict[str, Any]]:
        """ดึงสัญญาณล่าสุด"""
        if not self.history:
            return None
        return self.history[-1]["signal"]
    
    def get_signal_count(self, direction: str, minutes: int = 60) -> int:
        """
        นับจำนวนสัญญาณในช่วงเวลาที่กำหนด
        
        Args:
            direction: "BUY", "SELL", หรือ "ANY"
            minutes: ช่วงเวลาย้อนหลัง (นาที)
        """
        cutoff_time = time.time() - (minutes * 60)
        count = 0
        
        for item in self.history:
            if item["timestamp"] < cutoff_time:
                continue
            
            sig_dir = item["signal"].get("direction", "NONE")
            if direction == "ANY":
                if sig_dir in ("BUY", "SELL"):
                    count += 1
            elif sig_dir == direction:
                count += 1
        
        return count
    
    def get_win_rate(self) -> float:
        """
        คำนวณ win rate (ต้องมีการอัปเดตผลการเทรด)
        """
        if not self.history:
            return 0.0
        
        total = 0
        wins = 0
        
        for item in self.history:
            result = item.get("result")
            if result is not None:
                total += 1
                if result == "WIN":
                    wins += 1
        
        return (wins / total * 100.0) if total > 0 else 0.0
    
    def update_result(self, index: int, result: str) -> None:
        """
        อัปเดตผลการเทรด
        
        Args:
            index: index ของสัญญาณในประวัติ
            result: "WIN", "LOSS", หรือ "BREAKEVEN"
        """
        if 0 <= index < len(self.history):
            self.history[index]["result"] = result


class RiskManager:
    """
    จัดการความเสี่ยงและการคำนวณ lot size
    """
    def __init__(self, account_balance: float, risk_percent: float = 1.0):
        """
        Args:
            account_balance: ยอดเงินในบัญชี
            risk_percent: เปอร์เซ็นต์ความเสี่ยงต่อเทรด (1.0 = 1%)
        """
        self.account_balance = account_balance
        self.risk_percent = risk_percent
    
    def calculate_lot_size(self, entry: float, stop: float, pip_value: float = 10.0) -> float:
        """
        คำนวณ lot size ตามความเสี่ยงที่กำหนด
        
        Args:
            entry: ราคาเข้า
            stop: ราคา stop loss
            pip_value: มูลค่าต่อ pip (ขึ้นกับคู่เงิน)
            
        Returns:
            lot size ที่เหมาะสม
        """
        risk_amount = self.account_balance * (self.risk_percent / 100.0)
        stop_distance = abs(entry - stop)
        
        if stop_distance == 0:
            return 0.01  # lot size ต่ำสุด
        
        # คำนวณ lot size
        lot_size = risk_amount / (stop_distance * pip_value)
        
        # ปัดเศษให้เป็น 0.01
        lot_size = round(lot_size, 2)
        
        # จำกัด lot size ต่ำสุด
        return max(0.01, lot_size)
    
    def check_daily_loss_limit(self, daily_loss: float, max_daily_loss_percent: float = 5.0) -> bool:
        """
        ตรวจสอบว่าขาดทุนเกินลิมิตรายวันหรือไม่
        
        Args:
            daily_loss: ยอดขาดทุนสะสมในวัน (บวก = ขาดทุน)
            max_daily_loss_percent: เปอร์เซ็นต์ขาดทุนสูงสุดต่อวัน
            
        Returns:
            True ถ้ายังเทรดได้, False ถ้าเกินลิมิต
        """
        max_loss = self.account_balance * (max_daily_loss_percent / 100.0)
        return daily_loss < max_loss
    
    def check_max_trades(self, current_trades: int, max_trades: int = 5) -> bool:
        """
        ตรวจสอบจำนวนเทรดพร้อมกัน
        
        Args:
            current_trades: จำนวนเทรดที่เปิดอยู่
            max_trades: จำนวนเทรดสูงสุดที่อนุญาต
            
        Returns:
            True ถ้าเปิดเทรดใหม่ได้, False ถ้าเต็มแล้ว
        """
        return current_trades < max_trades


class MarketConditionFilter:
    """
    กรองสภาวะตลาดที่ไม่เหมาะสมกับการเทรด
    """
    @staticmethod
    def is_high_impact_news_time() -> bool:
        """
        ตรวจสอบว่าเป็นเวลาข่าวสำคัญหรือไม่
        (ต้องเชื่อมต่อกับ news calendar API)
        
        Returns:
            True ถ้าเป็นเวลาข่าวสำคัญ
        """
        # TODO: Implement news calendar check
        return False
    
    @staticmethod
    def is_market_open(symbol: str = "GOLD") -> bool:
        """
        ตรวจสอบว่าตลาดเปิดหรือไม่
        
        Args:
            symbol: ชื่อตราสาร
            
        Returns:
            True ถ้าตลาดเปิด
        """
        try:
            TradingEngine.ensure_mt5()
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
            return symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED
        except Exception:
            return False
    
    @staticmethod
    def check_spread(symbol: str, max_spread_points: int = 50) -> bool:
        """
        ตรวจสอบ spread ว่าอยู่ในเกณฑ์หรือไม่
        
        Args:
            symbol: ชื่อตราสาร
            max_spread_points: spread สูงสุดที่ยอมรับได้ (points)
            
        Returns:
            True ถ้า spread อยู่ในเกณฑ์
        """
        try:
            tick = TradingEngine.get_tick(symbol)
            spread = tick.ask - tick.bid
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
            
            spread_points = spread / symbol_info.point
            return spread_points <= max_spread_points
        except Exception:
            return False
    
    @staticmethod
    def check_volatility(atr: float, min_atr: float = 0.0, max_atr: float = float('inf')) -> bool:
        """
        ตรวจสอบความผันผวนว่าอยู่ในช่วงที่เหมาะสม
        
        Args:
            atr: ค่า ATR ปัจจุบัน
            min_atr: ATR ต่ำสุด
            max_atr: ATR สูงสุด
            
        Returns:
            True ถ้า volatility เหมาะสม
        """
        return min_atr <= atr <= max_atr


class AdvancedTradingEngine(TradingEngine):
    """
    Trading Engine เวอร์ชันขยาย พร้อมฟีเจอร์เพิ่มเติม
    """
    def __init__(self, config_path: str = "config.json", account_balance: float = 10000.0):
        super().__init__(config_path)
        self.signal_history = SignalHistory()
        self.risk_manager = RiskManager(account_balance)
        self.market_filter = MarketConditionFilter()
    
    def generate_signal_with_filters(self) -> Dict[str, Any]:
        """
        สร้างสัญญาณพร้อมกรองเงื่อนไขเพิ่มเติม
        
        Returns:
            Signal package พร้อมข้อมูลการกรอง
        """
        signal = self.generate_signal_package()
        
        # เพิ่มการกรองเพิ่มเติม
        additional_blocks = []
        
        # ตรวจสอบตลาดเปิด
        if not self.market_filter.is_market_open(signal.get("symbol", "GOLD")):
            additional_blocks.append("market_closed")
        
        # ตรวจสอบข่าวสำคัญ
        if self.market_filter.is_high_impact_news_time():
            additional_blocks.append("high_impact_news")
        
        # ตรวจสอบ spread
        if not self.market_filter.check_spread(signal.get("symbol", "GOLD")):
            additional_blocks.append("spread_too_wide")
        
        # ตรวจสอบ volatility
        ctx = signal.get("context", {})
        atr = ctx.get("atr", 0.0)
        if not self.market_filter.check_volatility(atr, min_atr=0.5):
            additional_blocks.append("volatility_too_low")
        
        # อัปเดต blocked_by
        if additional_blocks:
            current_blocks = ctx.get("blocked_by", "")
            if current_blocks:
                additional_blocks.insert(0, current_blocks)
            ctx["blocked_by"] = ",".join(additional_blocks)
            
            # ยกเลิกสัญญาณถ้ามีการบล็อกเพิ่มเติม
            if signal.get("direction") in ("BUY", "SELL"):
                signal["direction"] = "NONE"
                signal["entry_candidate"] = None
                signal["stop_candidate"] = None
                signal["tp_candidate"] = None
                signal["rr"] = 0.0
        
        # บันทึกสัญญาณ
        self.signal_history.add_signal(signal)
        
        return signal
    
    def generate_trade_order(self, signal: Dict[str, Any], lot_size: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        สร้างคำสั่งเทรดพร้อมคำนวณ lot size
        
        Args:
            signal: signal package
            lot_size: lot size (ถ้าไม่ระบุจะคำนวณอัตโนมัติ)
            
        Returns:
            Trade order dictionary หรือ None ถ้าไม่สามารถสร้างได้
        """
        is_valid, msg = validate_signal(signal)
        if not is_valid:
            return None
        
        entry = signal.get("entry_candidate")
        stop = signal.get("stop_candidate")
        tp = signal.get("tp_candidate")
        direction = signal.get("direction")
        symbol = signal.get("symbol")
        
        # คำนวณ lot size ถ้าไม่ได้ระบุ
        if lot_size is None:
            lot_size = self.risk_manager.calculate_lot_size(entry, stop)
        
        # สร้าง order
        order = {
            "symbol": symbol,
            "action": "BUY" if direction == "BUY" else "SELL",
            "lot_size": lot_size,
            "entry_price": entry,
            "stop_loss": stop,
            "take_profit": tp,
            "magic_number": 123456,  # เลขประจำตัว EA
            "comment": f"HIM_v{ENGINE_VERSION}",
            "signal_data": signal,
        }
        
        return order
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        ดึงสถิติการทำงานของ engine
        
        Returns:
            Dictionary ที่มีสถิติต่างๆ
        """
        total_signals = len(self.signal_history.history)
        buy_signals = self.signal_history.get_signal_count("BUY", minutes=1440)  # 24 ชั่วโมง
        sell_signals = self.signal_history.get_signal_count("SELL", minutes=1440)
        win_rate = self.signal_history.get_win_rate()
        
        return {
            "total_signals": total_signals,
            "buy_signals_24h": buy_signals,
            "sell_signals_24h": sell_signals,
            "win_rate": win_rate,
            "last_signal_time": self.signal_history.history[-1]["timestamp"] if self.signal_history.history else None,
        }


# ==================== UTILITY FUNCTIONS ====================

def export_signal_to_json(signal: Dict[str, Any], filename: str = "signal.json") -> None:
    """
    ส่งออกสัญญาณเป็นไฟล์ JSON
    
    Args:
        signal: signal package
        filename: ชื่อไฟล์
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(signal, f, indent=2, default=str, ensure_ascii=False)
        print(f"✓ Signal exported to {filename}")
    except Exception as e:
        print(f"✗ Failed to export signal: {str(e)}")


def import_signal_from_json(filename: str = "signal.json") -> Optional[Dict[str, Any]]:
    """
    นำเข้าสัญญาณจากไฟล์ JSON
    
    Args:
        filename: ชื่อไฟล์
        
    Returns:
        Signal package หรือ None ถ้าไม่สำเร็จ
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            signal = json.load(f)
        print(f"✓ Signal imported from {filename}")
        return signal
    except Exception as e:
        print(f"✗ Failed to import signal: {str(e)}")
        return None


def compare_signals(signal1: Dict[str, Any], signal2: Dict[str, Any]) -> Dict[str, Any]:
    """
    เปรียบเทียบสัญญาณ 2 ตัว
    
    Args:
        signal1: สัญญาณแรก
        signal2: สัญญาณที่สอง
        
    Returns:
        Dictionary ที่มีผลการเปรียบเทียบ
    """
    comparison = {
        "direction_match": signal1.get("direction") == signal2.get("direction"),
        "direction_1": signal1.get("direction"),
        "direction_2": signal2.get("direction"),
        "rr_1": signal1.get("rr", 0.0),
        "rr_2": signal2.get("rr", 0.0),
        "score_1": signal1.get("score", 0.0),
        "score_2": signal2.get("score", 0.0),
        "mode_1": signal1.get("context", {}).get("mode"),
        "mode_2": signal2.get("context", {}).get("mode"),
    }
    
    return comparison


def backtest_signal_logic(historical_data: Dict[str, np.ndarray], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    ทดสอบ logic การสร้างสัญญาณกับข้อมูลในอดีต
    
    Args:
        historical_data: ข้อมูลราคาย้อนหลัง
        config: configuration dictionary
        
    Returns:
        ผลการ backtest
    """
    # TODO: Implement comprehensive backtest logic
    results = {
        "total_signals": 0,
        "buy_signals": 0,
        "sell_signals": 0,
        "avg_rr": 0.0,
        "signals": [],
    }
    
    return results


def calculate_position_size_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    คำนวณ position size ตาม Kelly Criterion
    
    Args:
        win_rate: อัตราชนะ (0.0 - 1.0)
        avg_win: กำไรเฉลี่ย
        avg_loss: ขาดทุนเฉลี่ย
        
    Returns:
        เปอร์เซ็นต์ของเงินทุนที่ควรเสี่ยง
    """
    if avg_loss == 0:
        return 0.0
    
    win_loss_ratio = avg_win / avg_loss
    kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    
    # ใช้ Half Kelly เพื่อความปลอดภัย
    kelly_half = kelly / 2.0
    
    # จำกัดไม่เกิน 5%
    return max(0.0, min(kelly_half, 0.05))


def send_notification(message: str, method: str = "print") -> None:
    """
    ส่งการแจ้งเตือน
    
    Args:
        message: ข้อความที่ต้องการส่ง
        method: วิธีการส่ง ("print", "telegram", "email", "line")
    """
    if method == "print":
        print(f"📢 NOTIFICATION: {message}")
    elif method == "telegram":
        # TODO: Implement Telegram notification
        pass
    elif method == "email":
        # TODO: Implement Email notification
        pass
    elif method == "line":
        # TODO: Implement LINE notification
        pass


def format_price(price: float, decimals: int = 5) -> str:
    """
    จัดรูปแบบราคาให้อ่านง่าย
    
    Args:
        price: ราคา
        decimals: จำนวนทศนิยม
        
    Returns:
        ราคาที่จัดรูปแบบแล้ว
    """
    return f"{price:.{decimals}f}"


def calculate_pip_value(symbol: str, lot_size: float = 1.0) -> float:
    """
    คำนวณมูลค่าต่อ pip
    
    Args:
        symbol: ชื่อตราสาร
        lot_size: ขนาด lot
        
    Returns:
        มูลค่าต่อ pip
    """
    try:
        TradingEngine.ensure_mt5()
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 10.0  # ค่าเริ่มต้น
        
        # คำนวณมูลค่า pip
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        
        if tick_size > 0:
            pip_value = (tick_value / tick_size) * lot_size
            return pip_value
        
        return 10.0
    except Exception:
        return 10.0


# ==================== MONITORING & LOGGING ====================

class PerformanceMonitor:
    """
    ติดตามประสิทธิภาพการทำงานของ engine
    """
    def __init__(self):
        self.metrics = {
            "signal_generation_time": [],
            "data_fetch_time": [],
            "indicator_calculation_time": [],
        }
    
    def record_metric(self, metric_name: str, value: float) -> None:
        """บันทึกค่าประสิทธิภาพ"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name: str) -> float:
        """คำนวณค่าเฉลี่ย"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return 0.0
        return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
    
    def get_summary(self) -> Dict[str, float]:
        """ดึงสรุปประสิทธิภาพ"""
        summary = {}
        for metric_name in self.metrics:
            summary[f"{metric_name}_avg"] = self.get_average(metric_name)
            if self.metrics[metric_name]:
                summary[f"{metric_name}_max"] = max(self.metrics[metric_name])
                summary[f"{metric_name}_min"] = min(self.metrics[metric_name])
        return summary


class TradeLogger:
    """
    บันทึกข้อมูลการเทรด
    """
    def __init__(self, log_file: str = "trades.log"):
        self.log_file = log_file
    
    def log_signal(self, signal: Dict[str, Any]) -> None:
        """บันทึกสัญญาณ"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        direction = signal.get("direction", "NONE")
        entry = signal.get("entry_candidate")
        
        log_entry = f"[{timestamp}] SIGNAL: {direction}"
        if entry:
            log_entry += f" @ {entry:.5f}"
        
        self._write_log(log_entry)
    
    def log_trade_open(self, order: Dict[str, Any]) -> None:
        """บันทึกการเปิดเทรด"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        action = order.get("action", "UNKNOWN")
        symbol = order.get("symbol", "UNKNOWN")
        lot_size = order.get("lot_size", 0.0)
        entry = order.get("entry_price", 0.0)
        
        log_entry = f"[{timestamp}] OPEN: {action} {symbol} {lot_size} lots @ {entry:.5f}"
        self._write_log(log_entry)
    
    def log_trade_close(self, symbol: str, profit: float, result: str) -> None:
        """บันทึกการปิดเทรด"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] CLOSE: {symbol} | Profit: {profit:.2f} | Result: {result}"
        self._write_log(log_entry)
    
    def _write_log(self, message: str) -> None:
        """เขียนข้อความลง log file"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception as e:
            print(f"✗ Failed to write log: {str(e)}")


# ==================== CONFIGURATION TEMPLATES ====================

def create_default_config() -> Dict[str, Any]:
    """
    สร้าง configuration เริ่มต้น
    
    Returns:
        Default configuration dictionary
    """
    config = {
        "symbol": "GOLD",
        "mode": "sideway_scalp",
        "min_rr": 2.0,
        "timeframes": {
            "htf": "H4",
            "mtf": "H1",
            "ltf": "M15"
        },
        "risk": {
            "atr_period": 14,
            "atr_sl_mult": 1.8
        },
        "supertrend": {
            "period": 10,
            "multiplier": 3.0
        },
        "structure_sensitivity": {
            "htf": 5,
            "mtf": 4,
            "ltf": 3
        },
        "sideway_scalp": {
            "enabled": True,
            "adx_period": 14,
            "adx_max": 22.0,
            "bb_period": 20,
            "bb_std": 2.0,
            "bb_width_atr_max": 6.0,
            "rsi_period": 14,
            "rsi_overbought": 70.0,
            "rsi_oversold": 30.0,
            "require_confirmation": True,
            "touch_buffer_atr": 0.10,
            "near_trigger_atr": 0.50,
            "allow_soft_trigger": True,
            "rsi_soft_band": 10.0,
            "proximity_window_atr": 1.00,
            "proximity_score_min": 0.70
        }
    }
    return config


def save_config(config: Dict[str, Any], filename: str = "config.json") -> None:
    """
    บันทึก configuration ลงไฟล์
    
    Args:
        config: configuration dictionary
        filename: ชื่อไฟล์
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"✓ Configuration saved to {filename}")
    except Exception as e:
        print(f"✗ Failed to save configuration: {str(e)}")


def load_config_from_file(filename: str = "config.json") -> Dict[str, Any]:
    """
    โหลด configuration จากไฟล์
    
    Args:
        filename: ชื่อไฟล์
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"✓ Configuration loaded from {filename}")
        return config
    except Exception as e:
        print(f"✗ Failed to load configuration: {str(e)}")
        return create_default_config()


# ==================== TESTING UTILITIES ====================

def run_engine_test(config_path: str = "config.json", iterations: int = 1) -> None:
    """
    ทดสอบการทำงานของ engine
    
    Args:
        config_path: path ไปยัง config file
        iterations: จำนวนรอบที่ต้องการทดสอบ
    """
    print("=" * 60)
    print("RUNNING ENGINE TEST")
    print("=" * 60)
    
    try:
        engine = AdvancedTradingEngine(config_path)
        monitor = PerformanceMonitor()
        
        for i in range(iterations):
            print(f"\n--- Iteration {i + 1}/{iterations} ---")
            
            start_time = time.time()
            signal = engine.generate_signal_with_filters()
            end_time = time.time()
            
            generation_time = end_time - start_time
            monitor.record_metric("signal_generation_time", generation_time)
            
            print(f"Generation Time: {generation_time:.4f}s")
            print(f"Direction: {signal.get('direction')}")
            print(f"Score: {signal.get('score', 0.0):.3f}")
            
            ctx = signal.get("context", {})
            blocked = ctx.get("blocked_by")
            if blocked:
                print(f"Blocked: {blocked}")
            
            time.sleep(1)  # รอ 1 วินาที
        
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        summary = monitor.get_summary()
        for key, value in summary.items():
                        print(f"{key}: {value:.4f}s")
        
        print("\n" + "=" * 60)
        print("ENGINE STATISTICS")
        print("=" * 60)
        stats = engine.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("\n✓ Test completed successfully")
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def validate_config(config: Dict[str, Any]) -> Tuple[bool, list]:
    """
    ตรวจสอบความถูกต้องของ configuration
    
    Args:
        config: configuration dictionary
        
    Returns:
        (is_valid, errors) - True ถ้า config ถูกต้อง, list ของข้อผิดพลาด
    """
    errors = []
    
    # ตรวจสอบ required fields
    required_fields = ["symbol", "mode", "timeframes", "risk"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # ตรวจสอบ mode
    valid_modes = ["balanced", "sideway_scalp"]
    if config.get("mode") not in valid_modes:
        errors.append(f"Invalid mode: {config.get('mode')}. Must be one of {valid_modes}")
    
    # ตรวจสอบ timeframes
    tf = config.get("timeframes", {})
    required_tf = ["htf", "mtf", "ltf"]
    for tf_name in required_tf:
        if tf_name not in tf:
            errors.append(f"Missing timeframe: {tf_name}")
    
    # ตรวจสอบ risk parameters
    risk = config.get("risk", {})
    if "atr_period" in risk and risk["atr_period"] < 1:
        errors.append("atr_period must be >= 1")
    if "atr_sl_mult" in risk and risk["atr_sl_mult"] <= 0:
        errors.append("atr_sl_mult must be > 0")
    
    # ตรวจสอบ min_rr
    if "min_rr" in config and config["min_rr"] < 0:
        errors.append("min_rr must be >= 0")
    
    # ตรวจสอบ sideway_scalp parameters
    if config.get("mode") == "sideway_scalp":
        ss = config.get("sideway_scalp", {})
        if "proximity_score_min" in ss:
            if not (0.0 <= ss["proximity_score_min"] <= 1.0):
                errors.append("proximity_score_min must be between 0.0 and 1.0")
        if "adx_max" in ss and ss["adx_max"] < 0:
            errors.append("adx_max must be >= 0")
        if "rsi_overbought" in ss and "rsi_oversold" in ss:
            if ss["rsi_overbought"] <= ss["rsi_oversold"]:
                errors.append("rsi_overbought must be > rsi_oversold")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def create_config_wizard() -> Dict[str, Any]:
    """
    สร้าง configuration ผ่าน interactive wizard
    
    Returns:
        Configuration dictionary
    """
    print("=" * 60)
    print("CONFIGURATION WIZARD")
    print("=" * 60)
    
    config = {}
    
    # Symbol
    symbol = input("\nEnter symbol (default: GOLD): ").strip().upper()
    config["symbol"] = symbol if symbol else "GOLD"
    
    # Mode
    print("\nSelect trading mode:")
    print("1. Balanced (trend-following)")
    print("2. Sideway Scalp (mean-reversion)")
    mode_choice = input("Enter choice (1 or 2, default: 2): ").strip()
    config["mode"] = "balanced" if mode_choice == "1" else "sideway_scalp"
    
    # Timeframes
    print("\nTimeframe configuration:")
    htf = input("Higher Timeframe (default: H4): ").strip().upper()
    mtf = input("Middle Timeframe (default: H1): ").strip().upper()
    ltf = input("Lower Timeframe (default: M15): ").strip().upper()
    
    config["timeframes"] = {
        "htf": htf if htf else "H4",
        "mtf": mtf if mtf else "H1",
        "ltf": ltf if ltf else "M15"
    }
    
    # Risk parameters
    print("\nRisk management:")
    atr_period = input("ATR Period (default: 14): ").strip()
    atr_sl_mult = input("ATR Stop Loss Multiplier (default: 1.8): ").strip()
    min_rr = input("Minimum Risk/Reward (default: 2.0): ").strip()
    
    config["risk"] = {
        "atr_period": int(atr_period) if atr_period else 14,
        "atr_sl_mult": float(atr_sl_mult) if atr_sl_mult else 1.8
    }
    config["min_rr"] = float(min_rr) if min_rr else 2.0
    
    # SuperTrend
    print("\nSuperTrend configuration:")
    st_period = input("SuperTrend Period (default: 10): ").strip()
    st_mult = input("SuperTrend Multiplier (default: 3.0): ").strip()
    
    config["supertrend"] = {
        "period": int(st_period) if st_period else 10,
        "multiplier": float(st_mult) if st_mult else 3.0
    }
    
    # Structure sensitivity
    config["structure_sensitivity"] = {
        "htf": 5,
        "mtf": 4,
        "ltf": 3
    }
    
    # Sideway scalp settings
    if config["mode"] == "sideway_scalp":
        print("\nSideway Scalp configuration:")
        print("(Press Enter to use defaults)")
        
        adx_max = input("ADX Maximum (default: 22.0): ").strip()
        proximity_min = input("Proximity Score Minimum (default: 0.70): ").strip()
        
        config["sideway_scalp"] = {
            "enabled": True,
            "adx_period": 14,
            "adx_max": float(adx_max) if adx_max else 22.0,
            "bb_period": 20,
            "bb_std": 2.0,
            "bb_width_atr_max": 6.0,
            "rsi_period": 14,
            "rsi_overbought": 70.0,
            "rsi_oversold": 30.0,
            "require_confirmation": True,
            "touch_buffer_atr": 0.10,
            "near_trigger_atr": 0.50,
            "allow_soft_trigger": True,
            "rsi_soft_band": 10.0,
            "proximity_window_atr": 1.00,
            "proximity_score_min": float(proximity_min) if proximity_min else 0.70
        }
    
    print("\n" + "=" * 60)
    print("Configuration created successfully!")
    print("=" * 60)
    
    return config


# ==================== ADVANCED ANALYSIS ====================

class SignalAnalyzer:
    """
    วิเคราะห์คุณภาพและรูปแบบของสัญญาณ
    """
    @staticmethod
    def analyze_signal_quality(signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        วิเคราะห์คุณภาพของสัญญาณ
        
        Args:
            signal: signal package
            
        Returns:
            Dictionary ที่มีผลการวิเคราะห์
        """
        ctx = signal.get("context", {})
        
        quality_score = 0.0
        quality_factors = []
        
        # ตรวจสอบ R:R ratio
        rr = signal.get("rr", 0.0)
        if rr >= 3.0:
            quality_score += 30
            quality_factors.append("excellent_rr")
        elif rr >= 2.0:
            quality_score += 20
            quality_factors.append("good_rr")
        elif rr >= 1.5:
            quality_score += 10
            quality_factors.append("acceptable_rr")
        
        # ตรวจสอบ trend alignment
        htf_trend = ctx.get("HTF_trend", "ranging")
        mtf_trend = ctx.get("MTF_trend", "ranging")
        ltf_trend = ctx.get("LTF_trend", "ranging")
        direction = signal.get("direction", "NONE")
        
        trend_aligned = False
        if direction == "BUY":
            if htf_trend == "bullish" and mtf_trend == "bullish":
                quality_score += 25
                quality_factors.append("strong_trend_alignment")
                trend_aligned = True
            elif htf_trend == "bullish" or mtf_trend == "bullish":
                quality_score += 15
                quality_factors.append("partial_trend_alignment")
        elif direction == "SELL":
            if htf_trend == "bearish" and mtf_trend == "bearish":
                quality_score += 25
                quality_factors.append("strong_trend_alignment")
                trend_aligned = True
            elif htf_trend == "bearish" or mtf_trend == "bearish":
                quality_score += 15
                quality_factors.append("partial_trend_alignment")
        
        # ตรวจสอบ SuperTrend confirmation
        if signal.get("supertrend_ok", False):
            quality_score += 15
            quality_factors.append("supertrend_confirmed")
        
        # ตรวจสอบ proximity score (สำหรับ sideway_scalp)
        if ctx.get("mode") == "sideway_scalp":
            proximity_score = ctx.get("proximity_score", 0.0)
            if proximity_score >= 0.90:
                quality_score += 20
                quality_factors.append("excellent_proximity")
            elif proximity_score >= 0.70:
                quality_score += 10
                quality_factors.append("good_proximity")
        
        # ตรวจสอบ confirmation
        if ctx.get("bullish_reversal") or ctx.get("bearish_reversal"):
            quality_score += 10
            quality_factors.append("reversal_pattern")
        
        # คำนวณ quality grade
        if quality_score >= 80:
            quality_grade = "A"
        elif quality_score >= 60:
            quality_grade = "B"
        elif quality_score >= 40:
            quality_grade = "C"
        else:
            quality_grade = "D"
        
        return {
            "quality_score": quality_score,
            "quality_grade": quality_grade,
            "quality_factors": quality_factors,
            "trend_aligned": trend_aligned,
            "recommendation": "STRONG" if quality_score >= 70 else ("MODERATE" if quality_score >= 50 else "WEAK")
        }
    
    @staticmethod
    def detect_signal_patterns(history: list) -> Dict[str, Any]:
        """
        ตรวจจับรูปแบบของสัญญาณ
        
        Args:
            history: รายการสัญญาณย้อนหลัง
            
        Returns:
            Dictionary ที่มีรูปแบบที่พบ
        """
        if len(history) < 3:
            return {"pattern": "insufficient_data"}
        
        recent_signals = [h["signal"].get("direction") for h in history[-5:]]
        
        # ตรวจจับ consecutive signals
        buy_streak = 0
        sell_streak = 0
        for sig in reversed(recent_signals):
            if sig == "BUY":
                buy_streak += 1
                sell_streak = 0
            elif sig == "SELL":
                sell_streak += 1
                buy_streak = 0
            else:
                break
        
        pattern = {
            "buy_streak": buy_streak,
            "sell_streak": sell_streak,
            "pattern_detected": None
        }
        
        if buy_streak >= 3:
            pattern["pattern_detected"] = "strong_bullish_momentum"
        elif sell_streak >= 3:
            pattern["pattern_detected"] = "strong_bearish_momentum"
        elif len(set(recent_signals)) == 1 and recent_signals[0] == "NONE":
            pattern["pattern_detected"] = "no_clear_direction"
        
        return pattern


# ==================== INTEGRATION HELPERS ====================

def create_mt5_order(order_dict: Dict[str, Any]) -> bool:
    """
    ส่งคำสั่งเทรดไปยัง MT5
    
    Args:
        order_dict: order dictionary จาก generate_trade_order()
        
    Returns:
        True ถ้าส่งคำสั่งสำเร็จ
    """
    try:
        TradingEngine.ensure_mt5()
        
        symbol = order_dict.get("symbol")
        action = order_dict.get("action")
        lot_size = order_dict.get("lot_size")
        entry = order_dict.get("entry_price")
        sl = order_dict.get("stop_loss")
        tp = order_dict.get("take_profit")
        magic = order_dict.get("magic_number", 123456)
        comment = order_dict.get("comment", "HIM")
        
        # เตรียมคำสั่ง
        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "magic": int(magic),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # ส่งคำสั่ง
        result = mt5.order_send(request)
        
        if result is None:
            print(f"✗ Order send failed: No result")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"✗ Order send failed: {result.retcode} - {result.comment}")
            return False
        
        print(f"✓ Order sent successfully: Ticket #{result.order}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to send order: {str(e)}")
        return False


def close_mt5_position(ticket: int) -> bool:
    """
    ปิดออเดอร์ใน MT5
    
    Args:
        ticket: หมายเลขออเดอร์
        
    Returns:
        True ถ้าปิดสำเร็จ
    """
    try:
        TradingEngine.ensure_mt5()
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            print(f"✗ Position not found: {ticket}")
            return False
        
        position = position[0]
        
        # เตรียมคำสั่งปิด
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": mt5.symbol_info_tick(position.symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask,
            "magic": position.magic,
            "comment": "Close by HIM",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"✗ Failed to close position: {result.comment if result else 'No result'}")
            return False
        
        print(f"✓ Position closed successfully: {ticket}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to close position: {str(e)}")
        return False


def get_open_positions(symbol: Optional[str] = None, magic: Optional[int] = None) -> list:
    """
    ดึงรายการออเดอร์ที่เปิดอยู่
    
    Args:
        symbol: กรองตาม symbol (ถ้าระบุ)
        magic: กรองตาม magic number (ถ้าระบุ)
        
    Returns:
        รายการออเดอร์
    """
    try:
        TradingEngine.ensure_mt5()
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        result = []
        for pos in positions:
            if magic is not None and pos.magic != magic:
                continue
            
            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": pos.volume,
                "price_open": pos.price_open,
                "price_current": pos.price_current,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "magic": pos.magic,
                "comment": pos.comment,
            })
        
        return result
        
    except Exception as e:
        print(f"✗ Failed to get positions: {str(e)}")
        return []


def modify_position(ticket: int, new_sl: Optional[float] = None, new_tp: Optional[float] = None) -> bool:
    """
    แก้ไข SL/TP ของออเดอร์
    
    Args:
        ticket: หมายเลขออเดอร์
        new_sl: Stop Loss ใหม่
        new_tp: Take Profit ใหม่
        
    Returns:
        True ถ้าแก้ไขสำเร็จ
    """
    try:
        TradingEngine.ensure_mt5()
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            print(f"✗ Position not found: {ticket}")
            return False
        
        position = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": float(new_sl) if new_sl is not None else position.sl,
            "tp": float(new_tp) if new_tp is not None else position.tp,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"✗ Failed to modify position: {result.comment if result else 'No result'}")
            return False
        
        print(f"✓ Position modified successfully: {ticket}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to modify position: {str(e)}")
        return False


# ==================== AUTO TRADING SYSTEM ====================

class AutoTrader:
    """
    ระบบเทรดอัตโนมัติ
    """
    def __init__(self, config_path: str = "config.json", account_balance: float = 10000.0):
        self.engine = AdvancedTradingEngine(config_path, account_balance)
        self.logger = TradeLogger("auto_trader.log")
        self.is_running = False
        self.max_daily_trades = 10
        self.daily_trade_count = 0
        self.last_trade_date = None
    
    def start(self, check_interval: int = 60) -> None:
        """
        เริ่มระบบเทรดอัตโนมัติ
        
        Args:
            check_interval: ช่วงเวลาตรวจสอบสัญญาณ (วินาที)
        """
        print("=" * 60)
        print("AUTO TRADER STARTED")
        print("=" * 60)
        
        self.is_running = True
        
        try:
            while self.is_running:
                # ตรวจสอบวันใหม่
                current_date = time.strftime("%Y-%m-%d")
                if self.last_trade_date != current_date:
                    self.daily_trade_count = 0
                    self.last_trade_date = current_date
                    print(f"\n📅 New trading day: {current_date}")
                
                # ตรวจสอบจำนวนเทรดรายวัน
                if self.daily_trade_count >= self.max_daily_trades:
                    print(f"\n⚠️  Daily trade limit reached ({self.max_daily_trades}). Waiting for next day...")
                    time.sleep(check_interval)
                    continue
                
                # สร้างสัญญาณ
                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking for signals...")
                signal = self.engine.generate_signal_with_filters()
                
                direction = signal.get("direction", "NONE")
                
                if direction in ("BUY", "SELL"):
                    print(f"✓ Signal detected: {direction}")
                    
                    # วิเคราะห์คุณภาพ
                    analyzer = SignalAnalyzer()
                    quality = analyzer.analyze_signal_quality(signal)
                    
                    print(f"Quality Score: {quality['quality_score']}/100 (Grade: {quality['quality_grade']})")
                    print(f"Recommendation: {quality['recommendation']}")
                    
                    # ตัดสินใจเทรด
                    if quality['quality_score'] >= 50:  # เทรดเฉพาะสัญญาณคุณภาพดี
                        order = self.engine.generate_trade_order(signal)
                        
                        if order:
                            print(f"\n📊 Preparing to trade:")
                            print(f"   Symbol: {order['symbol']}")
                            print(f"   Action: {order['action']}")
                            print(f"   Lot Size: {order['lot_size']}")
                            print(f"   Entry: {order['entry_price']:.5f}")
                            print(f"   Stop Loss: {order['stop_loss']:.5f}")
                            print(f"   Take Profit: {order['take_profit']:.5f}")
                            
                            # ส่งออเดอร์
                            success = create_mt5_order(order)
                            
                            if success:
                                self.daily_trade_count += 1
                                self.logger.log_trade_open(order)
                                print(f"✓ Trade opened successfully ({self.daily_trade_count}/{self.max_daily_trades} today)")
                            else:
                                print("✗ Failed to open trade")
                    else:
                        print(f"⏸️  Signal quality too low. Skipping trade.")
                else:
                    ctx = signal.get("context", {})
                    blocked = ctx.get("blocked_by", "unknown")
                    print(f"No trade signal. Blocked by: {blocked}")
                
                # รอช่วงเวลาถัดไป
                print(f"\nWaiting {check_interval} seconds for next check...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Auto trader stopped by user")
        except Exception as e:
            print(f"\n✗ Auto trader error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            print("\n" + "=" * 60)
            print("AUTO TRADER STOPPED")
            print("=" * 60)
    
    def stop(self) -> None:
        """หยุดระบบเทรดอัตโนมัติ"""
        self.is_running = False


class TrailingStopManager:
    """
    จัดการ Trailing Stop
    """
    def __init__(self, trailing_atr_mult: float = 1.5):
        self.trailing_atr_mult = trailing_atr_mult
    
    def update_trailing_stops(self, symbol: str, magic: int = 123456) -> None:
        """
        อัปเดต trailing stop สำหรับออเดอร์ที่เปิดอยู่
        
        Args:
            symbol: ชื่อตราสาร
            magic: magic number
        """
        try:
            positions = get_open_positions(symbol=symbol, magic=magic)
            
            if not positions:
                return
            
            # ดึงข้อมูลราคา
            engine = TradingEngine()
            data = engine.get_data(symbol, mt5.TIMEFRAME_M15, 100)
            
            high = data["high"]
            low = data["low"]
            close = data["close"]
            
            atr_arr = engine.atr(high, low, close, 14)
            atr = float(atr_arr[-1]) if len(atr_arr) and np.isfinite(atr_arr[-1]) else 0.0
            
            if atr == 0:
                return
            
            trailing_distance = atr * self.trailing_atr_mult
            
            for pos in positions:
                ticket = pos["ticket"]
                pos_type = pos["type"]
                price_open = pos["price_open"]
                price_current = pos["price_current"]
                current_sl = pos["sl"]
                
                new_sl = None
                
                if pos_type == "BUY":
                    # คำนวณ trailing stop สำหรับ BUY
                    potential_sl = price_current - trailing_distance
                    
                    # อัปเดตเฉพาะเมื่อ SL ใหม่สูงกว่า SL เดิม
                    if potential_sl > current_sl and potential_sl > price_open:
                        new_sl = potential_sl
                
                elif pos_type == "SELL":
                    # คำนวณ trailing stop สำหรับ SELL
                    potential_sl = price_current + trailing_distance
                    
                    # อัปเดตเฉพาะเมื่อ SL ใหม่ต่ำกว่า SL เดิม
                    if (current_sl == 0 or potential_sl < current_sl) and potential_sl < price_open:
                        new_sl = potential_sl
                
                if new_sl is not None:
                    print(f"📍 Updating trailing stop for {ticket}: {current_sl:.5f} -> {new_sl:.5f}")
                    modify_position(ticket, new_sl=new_sl)
                    
        except Exception as e:
            print(f"✗ Failed to update trailing stops: {str(e)}")


# ==================== DASHBOARD & REPORTING ====================

class TradingDashboard:
    """
    แดชบอร์ดสำหรับแสดงข้อมูลการเทรด
    """
    @staticmethod
    def print_dashboard(engine: AdvancedTradingEngine, symbol: str = "GOLD") -> None:
        """
        แสดงแดชบอร์ดข้อมูลการเทรด
        
        Args:
            engine: AdvancedTradingEngine instance
            symbol: ชื่อตราสาร
        """
        print("\n" + "=" * 80)
        print(" " * 30 + "TRADING DASHBOARD")
        print("=" * 80)
        
        # ข้อมูลทั่วไป
        print(f"\n📊 GENERAL INFO")
        print("-" * 80)
        print(f"Symbol: {symbol}")
        print(f"Current Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Engine Version: {ENGINE_VERSION}")
        
        # สัญญาณปัจจุบัน
        signal = engine.generate_signal_with_filters()
        print(f"\n🎯 CURRENT SIGNAL")
        print("-" * 80)
        print(f"Direction: {signal.get('direction', 'NONE')}")
        print(f"Score: {signal.get('score', 0.0):.3f}")
        print(f"R:R Ratio: {signal.get('rr', 0.0):.2f}")
        
        ctx = signal.get("context", {})
        print(f"Mode: {ctx.get('mode', 'N/A')}")
        print(f"HTF Trend: {ctx.get('HTF_trend', 'N/A')}")
        print(f"MTF Trend: {ctx.get('MTF_trend', 'N/A')}")
        print(f"LTF Trend: {ctx.get('LTF_trend', 'N/A')}")
        
        blocked = ctx.get("blocked_by")
        if blocked:
            print(f"⚠️  Blocked by: {blocked}")
        
        # คุณภาพสัญญาณ
        if signal.get("direction") in ("BUY", "SELL"):
            analyzer = SignalAnalyzer()
            quality = analyzer.analyze_signal_quality(signal)
            
            print(f"\n⭐ SIGNAL QUALITY")
            print("-" * 80)
            print(f"Quality Score: {quality['quality_score']}/100")
            print(f"Quality Grade: {quality['quality_grade']}")
            print(f"Recommendation: {quality['recommendation']}")
            print(f"Quality Factors: {', '.join(quality['quality_factors'])}")
        
        # ออเดอร์ที่เปิดอยู่
        positions = get_open_positions(symbol=symbol)
        print(f"\n💼 OPEN POSITIONS ({len(positions)})")
        print("-" * 80)
        
        if positions:
            for pos in positions:
                print(f"Ticket: {pos['ticket']} | {pos['type']} {pos['volume']} lots")
                print(f"   Entry: {pos['price_open']:.5f} | Current: {pos['price_current']:.5f}")
                print(f"   SL: {pos['sl']:.5f} | TP: {pos['tp']:.5f}")
                print(f"   Profit: ${pos['profit']:.2f}")
                print()
        else:
            print("No open positions")
        
        # สถิติ
        stats = engine.get_statistics()
        print(f"\n📈 STATISTICS")
        print("-" * 80)
        print(f"Total Signals: {stats.get('total_signals', 0)}")
        print(f"Buy Signals (24h): {stats.get('buy_signals_24h', 0)}")
        print(f"Sell Signals (24h): {stats.get('sell_signals_24h', 0)}")
        print(f"Win Rate: {stats.get('win_rate', 0.0):.2f}%")
        
        # ข้อมูลตลาด
        print(f"\n🌍 MARKET CONDITIONS")
        print("-" * 80)
        
        market_filter = MarketConditionFilter()
        market_open = market_filter.is_market_open(symbol)
        spread_ok = market_filter.check_spread(symbol)
        
        print(f"Market Open: {'✓ Yes' if market_open else '✗ No'}")
        print(f"Spread OK: {'✓ Yes' if spread_ok else '✗ No'}")
        print(f"ATR: {ctx.get('atr', 0.0):.5f}")
        
        if ctx.get("mode") == "sideway_scalp":
            print(f"ADX: {ctx.get('adx', float('nan')):.2f} (max: {ctx.get('adx_max', 0):.2f})")
            print(f"RSI: {ctx.get('rsi', float('nan')):.2f}")
            print(f"Proximity Score: {ctx.get('proximity_score', 0.0):.3f}")
        
        print("\n" + "=" * 80)


def generate_daily_report(start_date: str, end_date: str) -> str:
    """
    สร้างรายงานประจำวัน
    
    Args:
        start_date: วันที่เริ่มต้น (YYYY-MM-DD)
        end_date: วันที่สิ้นสุด (YYYY-MM-DD)
        
    Returns:
        รายงานในรูปแบบข้อความ
    """
    report = []
    report.append("=" * 80)
    report.append(" " * 25 + "DAILY TRADING REPORT")
    report.append("=" * 80)
    report.append(f"\nReport Period: {start_date} to {end_date}")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # TODO: ดึงข้อมูลจากประวัติการเทรด
    report.append("\n📊 TRADING SUMMARY")
    report.append("-" * 80)
    report.append("Total Trades: 0")
    report.append("Winning Trades: 0")
    report.append("Losing Trades: 0")
    report.append("Win Rate: 0.00%")
    report.append("Total Profit/Loss: $0.00")
    report.append("Average R:R: 0.00")
    
    report.append("\n📈 PERFORMANCE METRICS")
    report.append("-" * 80)
    report.append("Best Trade: $0.00")
    report.append("Worst Trade: $0.00")
    report.append("Average Win: $0.00")
    report.append("Average Loss: $0.00")
    report.append("Profit Factor: 0.00")
    report.append("Max Drawdown: 0.00%")
    
    report.append("\n🎯 SIGNAL ANALYSIS")
    report.append("-" * 80)
    report.append("Total Signals Generated: 0")
    report.append("Signals Taken: 0")
    report.append("Signals Skipped: 0")
    report.append("Average Signal Quality: 0.00/100")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def export_report_to_file(report: str, filename: str = "daily_report.txt") -> None:
    """
    ส่งออกรายงานเป็นไฟล์
    
    Args:
        report: ข้อความรายงาน
        filename: ชื่อไฟล์
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✓ Report exported to {filename}")
    except Exception as e:
        print(f"✗ Failed to export report: {str(e)}")


# ==================== CLI INTERFACE ====================

def print_menu() -> None:
    """แสดงเมนูหลัก"""
    print("\n" + "=" * 60)
    print(" " * 20 + "HIM TRADING ENGINE")
    print(" " * 22 + f"Version {ENGINE_VERSION}")
    print("=" * 60)
    print("\n1. Generate Signal")
    print("2. View Dashboard")
    print("3. Start Auto Trading")
    print("4. View Open Positions")
    print("5. Close All Positions")
    print("6. Configuration Wizard")
    print("7. Run Engine Test")
    print("8. Generate Daily Report")
    print("9. Export Signal to JSON")
    print("0. Exit")
    print("\n" + "=" * 60)


def cli_main() -> None:
    """
    Command Line Interface หลัก
    """
    engine = None
    auto_trader = None
    
    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip()
        
        try:
            if choice == "1":
                # Generate Signal
                if engine is None:
                    engine = AdvancedTradingEngine()
                
                print("\n🔄 Generating signal...")
                signal = engine.generate_signal_with_filters()
                print(format_signal_summary(signal))
                
                is_valid, msg = validate_signal(signal)
                print(f"\nValidation: {msg}")
                
                if is_valid:
                    analyzer = SignalAnalyzer()
                    quality = analyzer.analyze_signal_quality(signal)
                    print(f"\nQuality Analysis:")
                    print(f"Score: {quality['quality_score']}/100 (Grade: {quality['quality_grade']})")
                    print(f"Recommendation: {quality['recommendation']}")
            
            elif choice == "2":
                # View Dashboard
                if engine is None:
                    engine = AdvancedTradingEngine()
                
                TradingDashboard.print_dashboard(engine)
            
            elif choice == "3":
                # Start Auto Trading
                if auto_trader is None:
                    balance = input("\nEnter account balance (default: 10000): ").strip()
                    balance = float(balance) if balance else 10000.0
                    
                    auto_trader = AutoTrader(account_balance=balance)
                
                interval = input("Enter check interval in seconds (default: 60): ").strip()
                interval = int(interval) if interval else 60
                
                print("\n⚠️  Starting auto trading. Press Ctrl+C to stop.")
                input("Press Enter to confirm...")
                
                auto_trader.start(check_interval=interval)
            
            elif choice == "4":
                # View Open Positions
                symbol = input("\nEnter symbol (default: GOLD): ").strip().upper()
                symbol = symbol if symbol else "GOLD"
                
                positions = get_open_positions(symbol=symbol)
                
                print(f"\n💼 Open Positions for {symbol}: {len(positions)}")
                print("-" * 60)
                
                if positions:
                    for pos in positions:
                        print(f"\nTicket: {pos['ticket']}")
                        print(f"Type: {pos['type']}")
                        print(f"Volume: {pos['volume']} lots")
                        print(f"Entry: {pos['price_open']:.5f}")
                        print(f"Current: {pos['price_current']:.5f}")
                        print(f"SL: {pos['sl']:.5f}")
                        print(f"TP: {pos['tp']:.5f}")
                        print(f"Profit: ${pos['profit']:.2f}")
                else:
                    print("No open positions")
            
            elif choice == "5":
                # Close All Positions
                symbol = input("\nEnter symbol (default: GOLD): ").strip().upper()
                symbol = symbol if symbol else "GOLD"
                
                positions = get_open_positions(symbol=symbol)
                
                if not positions:
                    print("\nNo positions to close")
                    continue
                
                confirm = input(f"\n⚠️  Close {len(positions)} position(s)? (yes/no): ").strip().lower()
                
                if confirm == "yes":
                    for pos in positions:
                        print(f"Closing position {pos['ticket']}...")
                        close_mt5_position(pos['ticket'])
                    print("\n✓ All positions closed")
                else:
                    print("\n✗ Operation cancelled")
            
            elif choice == "6":
                # Configuration Wizard
                config = create_config_wizard()
                
                save_choice = input("\nSave configuration? (yes/no): ").strip().lower()
                if save_choice == "yes":
                    filename = input("Enter filename (default: config.json): ").strip()
                    filename = filename if filename else "config.json"
                    save_config(config, filename)
                    
                    # Reload engine with new config
                    engine = AdvancedTradingEngine(filename)
            
            elif choice == "7":
                # Run Engine Test
                iterations = input("\nEnter number of iterations (default: 5): ").strip()
                iterations = int(iterations) if iterations else 5
                
                run_engine_test(iterations=iterations)
            
            elif choice == "8":
                # Generate Daily Report
                start = input("\nEnter start date (YYYY-MM-DD, default: today): ").strip()
                end = input("Enter end date (YYYY-MM-DD, default: today): ").strip()
                
                today = time.strftime("%Y-%m-%d")
                start = start if start else today
                end = end if end else today
                
                report = generate_daily_report(start, end)
                print(report)
                
                export_choice = input("\nExport to file? (yes/no): ").strip().lower()
                if export_choice == "yes":
                    filename = input("Enter filename (default: daily_report.txt): ").strip()
                    filename = filename if filename else "daily_report.txt"
                    export_report_to_file(report, filename)
            
            elif choice == "9":
                # Export Signal to JSON
                if engine is None:
                    engine = AdvancedTradingEngine()
                
                signal = engine.generate_signal_with_filters()
                
                filename = input("\nEnter filename (default: signal.json): ").strip()
                filename = filename if filename else "signal.json"
                
                export_signal_to_json(signal, filename)
            
            elif choice == "0":
                # Exit
                print("\n👋 Goodbye!")
                break
            
            else:
                print("\n✗ Invalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Operation interrupted by user")
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")


# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    import sys
    
    # ตรวจสอบ command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "signal":
            # สร้างสัญญาณเดียว
            engine = create_engine()
            signal = get_signal(engine)
            print(format_signal_summary(signal))
        
        elif command == "dashboard":
            # แสดงแดชบอร์ด
            engine = AdvancedTradingEngine()
            TradingDashboard.print_dashboard(engine)
        
        elif command == "auto":
            # เริ่ม auto trading
            balance = float(sys.argv[2]) if len(sys.argv) > 2 else 10000.0
            interval = int(sys.argv[3]) if len(sys.argv) > 3 else 60
            
            auto_trader = AutoTrader(account_balance=balance)
            auto_trader.start(check_interval=interval)
        
        elif command == "test":
            # ทดสอบ engine
            iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            run_engine_test(iterations=iterations)
        
        elif command == "config":
            # สร้าง config ใหม่
            config = create_config_wizard()
            save_config(config)
        
        elif command == "cli":
            # เปิด CLI interface
            cli_main()
        
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  signal     - Generate a single signal")
            print("  dashboard  - Show trading dashboard")
            print("  auto [balance] [interval] - Start auto trading")
            print("  test [iterations] - Run engine test")
            print("  config     - Create new configuration")
            print("  cli        - Open interactive CLI")
    
    else:
        # ไม่มี arguments - เปิด CLI interface
        cli_main()


# ==================== ADDITIONAL UTILITIES ====================

def calculate_lot_from_risk_percent(
    account_balance: float,
    risk_percent: float,
    entry: float,
    stop: float,
    pip_value: float = 10.0
) -> float:
    """
    คำนวณ lot size จากเปอร์เซ็นต์ความเสี่ยง
    
    Args:
        account_balance: ยอดเงินในบัญชี
        risk_percent: เปอร์เซ็นต์ความเสี่ยง (1.0 = 1%)
        entry: ราคาเข้า
        stop: ราคา stop loss
        pip_value: มูลค่าต่อ pip
        
    Returns:
        lot size ที่เหมาะสม
    """
    risk_amount = account_balance * (risk_percent / 100.0)
    stop_distance = abs(entry - stop)
    
    if stop_distance == 0:
        return 0.01
    
    lot_size = risk_amount / (stop_distance * pip_value)
    lot_size = round(lot_size, 2)
    
    return max(0.01, lot_size)


def calculate_position_value(lot_size: float, price: float, contract_size: float = 100.0) -> float:
    """
    คำนวณมูลค่าของ position
    
    Args:
        lot_size: ขนาด lot
        price: ราคาปัจจุบัน
        contract_size: ขนาด contract (100 oz สำหรับ GOLD)
        
    Returns:
        มูลค่า position
    """
    return lot_size * contract_size * price


def calculate_margin_required(
    lot_size: float,
    price: float,
    leverage: int = 100,
    contract_size: float = 100.0
) -> float:
    """
    คำนวณ margin ที่ต้องใช้
    
    Args:
        lot_size: ขนาด lot
        price: ราคาปัจจุบัน
        leverage: leverage (เช่น 100 = 1:100)
        contract_size: ขนาด contract
        
    Returns:
        margin ที่ต้องใช้
    """
    position_value = calculate_position_value(lot_size, price, contract_size)
    return position_value / leverage


def calculate_profit_loss(
    entry: float,
    exit: float,
    lot_size: float,
    position_type: str,
    contract_size: float = 100.0
) -> float:
    """
    คำนวณกำไร/ขาดทุน
    
    Args:
        entry: ราคาเข้า
        exit: ราคาออก
        lot_size: ขนาด lot
        position_type: "BUY" หรือ "SELL"
        contract_size: ขนาด contract
        
    Returns:
        กำไร/ขาดทุน (บวก = กำไร, ลบ = ขาดทุน)
    """
    if position_type == "BUY":
        profit = (exit - entry) * lot_size * contract_size
    else:  # SELL
        profit = (entry - exit) * lot_size * contract_size
    
    return profit


def calculate_break_even_price(
    entry: float,
    spread: float,
    commission: float,
    lot_size: float,
    position_type: str,
    contract_size: float = 100.0
) -> float:
    """
    คำนวณราคา break even
    
    Args:
        entry: ราคาเข้า
        spread: spread
        commission: ค่าคอมมิชชั่น
        lot_size: ขนาด lot
        position_type: "BUY" หรือ "SELL"
        contract_size: ขนาด contract
        
    Returns:
        ราคา break even
    """
    total_cost = spread + (commission / (lot_size * contract_size))
    
    if position_type == "BUY":
        return entry + total_cost
    else:  # SELL
        return entry - total_cost


def format_timeframe(tf: int) -> str:
    """
    แปลง MT5 timeframe constant เป็นข้อความ
    
    Args:
        tf: MT5 timeframe constant
        
    Returns:
        ข้อความ timeframe (เช่น "M15", "H1")
    """
    timeframe_map = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M2: "M2",
        mt5.TIMEFRAME_M3: "M3",
        mt5.TIMEFRAME_M4: "M4",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M6: "M6",
        mt5.TIMEFRAME_M10: "M10",
        mt5.TIMEFRAME_M12: "M12",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M20: "M20",
        mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H2: "H2",
        mt5.TIMEFRAME_H3: "H3",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_H6: "H6",
        mt5.TIMEFRAME_H8: "H8",
        mt5.TIMEFRAME_H12: "H12",
        mt5.TIMEFRAME_D1: "D1",
        mt5.TIMEFRAME_W1: "W1",
        mt5.TIMEFRAME_MN1: "MN1",
    }
    return timeframe_map.get(tf, f"TF{tf}")


def parse_timeframe(tf_str: str) -> int:
    """
    แปลงข้อความ timeframe เป็น MT5 constant
    
    Args:
        tf_str: ข้อความ timeframe (เช่น "M15", "H1")
        
    Returns:
        MT5 timeframe constant
    """
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M2": mt5.TIMEFRAME_M2,
        "M3": mt5.TIMEFRAME_M3,
        "M4": mt5.TIMEFRAME_M4,
        "M5": mt5.TIMEFRAME_M5,
        "M6": mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,
        "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H2": mt5.TIMEFRAME_H2,
        "H3": mt5.TIMEFRAME_H3,
        "H4": mt5.TIMEFRAME_H4,
        "H6": mt5.TIMEFRAME_H6,
        "H8": mt5.TIMEFRAME_H8,
        "H12": mt5.TIMEFRAME_H12,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    return timeframe_map.get(tf_str.upper(), mt5.TIMEFRAME_M15)


# ==================== WEBHOOK INTEGRATION ====================

class WebhookServer:
    """
    เซิร์ฟเวอร์รับ webhook สำหรับสัญญาณจากภายนอก
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self.engine = None
        self.auto_trader = None
    
    def start(self) -> None:
        """เริ่มเซิร์ฟเวอร์ webhook"""
        try:
            from flask import Flask, request, jsonify
            
            app = Flask(__name__)
            
            @app.route("/webhook", methods=["POST"])
            def webhook():
                """รับสัญญาณจาก webhook"""
                try:
                    data = request.get_json()
                    
                    # ตรวจสอบข้อมูล
                    required_fields = ["symbol", "action", "entry", "stop", "tp"]
                    for field in required_fields:
                        if field not in data:
                            return jsonify({"error": f"Missing field: {field}"}), 400
                    
                    # สร้างออเดอร์
                    order = {
                        "symbol": data["symbol"],
                        "action": data["action"],
                        "lot_size": data.get("lot_size", 0.01),
                        "entry_price": float(data["entry"]),
                        "stop_loss": float(data["stop"]),
                        "take_profit": float(data["tp"]),
                        "magic_number": data.get("magic", 123456),
                        "comment": data.get("comment", "Webhook"),
                    }
                    
                    # ส่งออเดอร์
                    success = create_mt5_order(order)
                    
                    if success:
                        return jsonify({"status": "success", "message": "Order placed"}), 200
                    else:
                        return jsonify({"status": "error", "message": "Failed to place order"}), 500
                
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            
            @app.route("/signal", methods=["GET"])
            def get_signal_endpoint():
                """ดึงสัญญาณปัจจุบัน"""
                try:
                    if self.engine is None:
                        self.engine = AdvancedTradingEngine()
                    
                    signal = self.engine.generate_signal_with_filters()
                    
                    return jsonify(signal), 200
                
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            
            @app.route("/positions", methods=["GET"])
            def get_positions_endpoint():
                """ดึงรายการออเดอร์ที่เปิดอยู่"""
                try:
                    symbol = request.args.get("symbol", "GOLD")
                    positions = get_open_positions(symbol=symbol)
                    
                    return jsonify({"positions": positions}), 200
                
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            
            @app.route("/health", methods=["GET"])
            def health_check():
                """ตรวจสอบสถานะเซิร์ฟเวอร์"""
                return jsonify({"status": "healthy", "version": ENGINE_VERSION}), 200
            
            print(f"🌐 Webhook server starting on {self.host}:{self.port}")
            app.run(host=self.host, port=self.port, debug=False)
        
        except ImportError:
            print("✗ Flask not installed. Install with: pip install flask")
        except Exception as e:
            print(f"✗ Failed to start webhook server: {str(e)}")


# ==================== TELEGRAM BOT INTEGRATION ====================

class TelegramBot:
    """
    Telegram Bot สำหรับรับคำสั่งและส่งการแจ้งเตือน
    """
    def __init__(self, token: str):
        self.token = token
        self.engine = None
    
    def start(self) -> None:
        """เริ่ม Telegram Bot"""
        try:
            from telegram import Update
            from telegram.ext import Application, CommandHandler, ContextTypes
            
            async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
                """คำสั่ง /start"""
                await update.message.reply_text(
                    "🤖 HIM Trading Bot\n\n"
                    "Available commands:\n"
                    "/signal - Get current signal\n"
                    "/positions - View open positions\n"
                    "/stats - View statistics\n"
                    "/help - Show this message"
                )
            
            async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
                """คำสั่ง /signal"""
                try:
                    if self.engine is None:
                        self.engine = AdvancedTradingEngine()
                    
                    signal = self.engine.generate_signal_with_filters()
                    summary = format_signal_summary(signal)
                    
                    await update.message.reply_text(f"```\n{summary}\n```", parse_mode="Markdown")
                
                except Exception as e:
                    await update.message.reply_text(f"Error: {str(e)}")
            
            async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
                """คำสั่ง /positions"""
                try:
                    positions = get_open_positions()
                    
                    if not positions:
                        await update.message.reply_text("No open positions")
                        return
                    
                    message = "💼 Open Positions:\n\n"
                    for pos in positions:
                        message += f"Ticket: {pos['ticket']}\n"
                        message += f"{pos['type']} {pos['volume']} lots\n"
                        message += f"Profit: ${pos['profit']:.2f}\n\n"
                    
                    await update.message.reply_text(message)
                
                except Exception as e:
                    await update.message.reply_text(f"Error: {str(e)}")
            
            async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
                """คำสั่ง /stats"""
                try:
                    if self.engine is None:
                        self.engine = AdvancedTradingEngine()
                    
                    stats = self.engine.get_statistics()
                    
                    message = "📈 Statistics:\n\n"
                    message += f"Total Signals: {stats.get('total_signals', 0)}\n"
                    message += f"Buy Signals (24h): {stats.get('buy_signals_24h', 0)}\n"
                    message += f"Sell Signals (24h): {stats.get('sell_signals_24h', 0)}\n"
                    message += f"Win Rate: {stats.get('win_rate', 0.0):.2f}%\n"
                    
                    await update.message.reply_text(message)
                
                except Exception as e:
                    await update.message.reply_text(f"Error: {str(e)}")
            
            # สร้าง application
            app = Application.builder().token(self.token).build()
            
            # เพิ่ม handlers
            app.add_handler(CommandHandler("start", start_command))
            app.add_handler(CommandHandler("help", start_command))
            app.add_handler(CommandHandler("signal", signal_command))
            app.add_handler(CommandHandler("positions", positions_command))
            app.add_handler(CommandHandler("stats", stats_command))
            
            print("🤖 Telegram bot starting...")
            app.run_polling()
        
        except ImportError:
            print("✗ python-telegram-bot not installed. Install with: pip install python-telegram-bot")
        except Exception as e:
            print(f"✗ Failed to start Telegram bot: {str(e)}")
    
    def send_notification(self, chat_id: str, message: str) -> None:
        """
        ส่งการแจ้งเตือนผ่าน Telegram
        
        Args:
            chat_id: Telegram chat ID
            message: ข้อความที่ต้องการส่ง
        """
        try:
            import requests
            
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                print("✓ Notification sent via Telegram")
            else:
                print(f"✗ Failed to send Telegram notification: {response.text}")
        
        except Exception as e:
            print(f"✗ Failed to send Telegram notification: {str(e)}")


# ==================== DATA EXPORT & IMPORT ====================

def export_signals_to_csv(signals: list, filename: str = "signals.csv") -> None:
    """
    ส่งออกสัญญาณเป็นไฟล์ CSV
    
    Args:
        signals: รายการสัญญาณ
        filename: ชื่อไฟล์
    """
    try:
        import csv
        
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "Timestamp", "Symbol", "Direction", "Entry", "Stop Loss", 
                "Take Profit", "R:R", "Score", "Mode", "HTF Trend", 
                "MTF Trend", "LTF Trend", "Blocked By"
            ])
            
            # Data
            for sig_item in signals:
                sig = sig_item.get("signal", {})
                ctx = sig.get("context", {})
                
                writer.writerow([
                    sig_item.get("timestamp", ""),
                    sig.get("symbol", ""),
                    sig.get("direction", ""),
                    sig.get("entry_candidate", ""),
                    sig.get("stop_candidate", ""),
                    sig.get("tp_candidate", ""),
                    sig.get("rr", 0.0),
                    sig.get("score", 0.0),
                    ctx.get("mode", ""),
                    ctx.get("HTF_trend", ""),
                    ctx.get("MTF_trend", ""),
                    ctx.get("LTF_trend", ""),
                    ctx.get("blocked_by", ""),
                ])
        
        print(f"✓ Signals exported to {filename}")
    
    except Exception as e:
        print(f"✗ Failed to export signals: {str(e)}")


def import_signals_from_csv(filename: str = "signals.csv") -> list:
    """
    นำเข้าสัญญาณจากไฟล์ CSV
    
    Args:
        filename: ชื่อไฟล์
        
    Returns:
        รายการสัญญาณ
    """
    try:
        import csv
        
        signals = []
        
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                signal = {
                    "timestamp": row.get("Timestamp", ""),
                    "signal": {
                        "symbol": row.get("Symbol", ""),
                        "direction": row.get("Direction", ""),
                        "entry_candidate": float(row.get("Entry", 0)) if row.get("Entry") else None,
                        "stop_candidate": float(row.get("Stop Loss", 0)) if row.get("Stop Loss") else None,
                        "tp_candidate": float(row.get("Take Profit", 0)) if row.get("Take Profit") else None,
                        "rr": float(row.get("R:R", 0)),
                        "score": float(row.get("Score", 0)),
                        "context": {
                            "mode": row.get("Mode", ""),
                            "HTF_trend": row.get("HTF Trend", ""),
                            "MTF_trend": row.get("MTF Trend", ""),
                            "LTF_trend": row.get("LTF Trend", ""),
                            "blocked_by": row.get("Blocked By", ""),
                        }
                    }
                }
                signals.append(signal)
        
        print(f"✓ {len(signals)} signals imported from {filename}")
        return signals
    
    except Exception as e:
        print(f"✗ Failed to import signals: {str(e)}")
        return []


def export_trades_to_excel(trades: list, filename: str = "trades.xlsx") -> None:
    """
    ส่งออกข้อมูลการเทรดเป็นไฟล์ Excel
    
    Args:
        trades: รายการการเทรด
        filename: ชื่อไฟล์
    """
    try:
        import pandas as pd
        
        df = pd.DataFrame(trades)
        df.to_excel(filename, index=False, engine='openpyxl')
        
        print(f"✓ Trades exported to {filename}")
    
    except ImportError:
        print("✗ pandas or openpyxl not installed. Install with: pip install pandas openpyxl")
    except Exception as e:
        print(f"✗ Failed to export trades: {str(e)}")


# ==================== PERFORMANCE OPTIMIZATION ====================

def optimize_config_parameters(
    historical_data: Dict[str, np.ndarray],
    param_ranges: Dict[str, tuple],
    iterations: int = 100
) -> Dict[str, Any]:
    """
    หาค่า parameter ที่เหมาะสมที่สุด
    
    Args:
        historical_data: ข้อมูลราคาย้อนหลัง
        param_ranges: ช่วงค่าที่ต้องการทดสอบ {"param_name": (min, max, step)}
        iterations: จำนวนรอบที่ต้องการทดสอบ
        
    Returns:
        Configuration ที่ดีที่สุด
    """
    # TODO: Implement parameter optimization using grid search or genetic algorithm
    print("⚠️  Parameter optimization not yet implemented")
    return create_default_config()


def cache_indicator_results(func):
    """
    Decorator สำหรับ cache ผลลัพธ์ของ indicator
    """
    cache = {}
    
    def wrapper(*args, **kwargs):
        # สร้าง cache key
        key = str(args) + str(kwargs)
        
        if key in cache:
            return cache[key]
        
        result = func(*args, **kwargs)
        cache[key] = result
        
        return result
    
    return wrapper


# ==================== ERROR HANDLING & RECOVERY ====================

class ErrorHandler:
    """
    จัดการข้อผิดพลาดและการกู้คืน
    """
    def __init__(self, log_file: str = "errors.log"):
        self.log_file = log_file
        self.error_count = 0
        self.max_errors = 10
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """
        บันทึกข้อผิดพลาด
        
        Args:
            error: Exception object
            context: บริบทของข้อผิดพลาด
        """
        self.error_count += 1
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        error_msg = f"[{timestamp}] ERROR in {context}: {str(error)}\n"
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(error_msg)
                
                # เขียน traceback
                import traceback
                f.write(traceback.format_exc())
                f.write("\n" + "=" * 80 + "\n")
        
        except Exception as e:
            print(f"✗ Failed to log error: {str(e)}")
    
    def should_stop(self) -> bool:
        """
        ตรวจสอบว่าควรหยุดทำงานเนื่องจากข้อผิดพลาดมากเกินไป
        
        Returns:
            True ถ้าควรหยุด
        """
        return self.error_count >= self.max_errors
    
    def reset_error_count(self) -> None:
        """รีเซ็ตจำนวนข้อผิดพลาด"""
        self.error_count = 0


class ConnectionManager:
    """
    จัดการการเชื่อมต่อกับ MT5
    """
    @staticmethod
    def ensure_connection(max_retries: int = 3, retry_delay: int = 5) -> bool:
        """
        ตรวจสอบและกู้คืนการเชื่อมต่อ MT5
        
        Args:
            max_retries: จำนวนครั้งที่พยายามเชื่อมต่อใหม่
            retry_delay: ระยะเวลารอระหว่างการเชื่อมต่อใหม่ (วินาที)
            
        Returns:
            True ถ้าเชื่อมต่อสำเร็จ
        """
        for attempt in range(max_retries):
            try:
                if not mt5.initialize():
                    print(f"⚠️  MT5 connection failed. Attempt {attempt + 1}/{max_retries}")
                    time.sleep(retry_delay)
                    continue
                
                # ตรวจสอบการเชื่อมต่อ
                account_info = mt5.account_info()
                if account_info is None:
                    print(f"⚠️  MT5 account info unavailable. Attempt {attempt + 1}/{max_retries}")
                    mt5.shutdown()
                    time.sleep(retry_delay)
                    continue
                
                print("✓ MT5 connection established")
                return True
            
            except Exception as e:
                print(f"✗ Connection error: {str(e)}")
                time.sleep(retry_delay)
        
        print("✗ Failed to establish MT5 connection after all retries")
        return False
    
    @staticmethod
    def check_connection() -> bool:
        """
        ตรวจสอบสถานะการเชื่อมต่อ
        
        Returns:
            True ถ้าเชื่อมต่ออยู่
        """
        try:
            account_info = mt5.account_info()
            return account_info is not None
        except Exception:
            return False


# ==================== FINAL UTILITIES ====================

def get_system_info() -> Dict[str, Any]:
    """
    ดึงข้อมูลระบบ
    
    Returns:
        Dictionary ที่มีข้อมูลระบบ
    """
    import platform
    import psutil
    
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
        "memory_available": psutil.virtual_memory().available / (1024**3),  # GB
        "memory_percent": psutil.virtual_memory().percent,
    }
    
    return info


def print_system_info() -> None:
    """แสดงข้อมูลระบบ"""
    try:
        info = get_system_info()
        
        print("\n" + "=" * 60)
        print("SYSTEM INFORMATION")
        print("=" * 60)
        print(f"Platform: {info['platform']} {info['platform_version']}")
        print(f"Python Version: {info['python_version']}")
        print(f"CPU Cores: {info['cpu_count']}")
        print(f"CPU Usage: {info['cpu_percent']}%")
        print(f"Memory: {info['memory_available']:.2f}GB / {info['memory_total']:.2f}GB ({info['memory_percent']}%)")
        print(f"Engine Version: {ENGINE_VERSION}")
        print("=" * 60)
    
    except ImportError:
        print("⚠️  psutil not installed. Install with: pip install psutil")
    except Exception as e:
        print(f"✗ Failed to get system info: {str(e)}")


def check_dependencies() -> bool:
    """
    ตรวจสอบ dependencies ที่จำเป็น
    
    Returns:
        True ถ้า dependencies ครบถ้วน
    """
    required = {
        "MetaTrader5": "MetaTrader5",
        "numpy": "numpy",
        "pandas": "pandas",
    }
    
    optional = {
        "flask": "Flask (for webhook server)",
        "telegram": "python-telegram-bot (for Telegram bot)",
        "openpyxl": "openpyxl (for Excel export)",
        "psutil": "psutil (for system info)",
        "requests": "requests (for notifications)",
    }
    
    print("\n" + "=" * 60)
    print("DEPENDENCY CHECK")
    print("=" * 60)
    
    all_ok = True
    
    print("\nRequired Dependencies:")
    for module_name, package_name in required.items():
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT INSTALLED")
            all_ok = False
    
    print("\nOptional Dependencies:")
    for module_name, description in optional.items():
        try:
            __import__(module_name)
            print(f"  ✓ {description}")
        except ImportError:
            print(f"  ⚠ {description} - not installed")
    
    print("\n" + "=" * 60)
    
    if not all_ok:
        print("\n⚠️  Some required dependencies are missing!")
        print("Install with: pip install MetaTrader5 numpy pandas")
        return False
    
    print("\n✓ All required dependencies are installed")
    return True


def create_example_config_files() -> None:
    """
    สร้างไฟล์ตัวอย่าง
    """
    # สร้าง config.json
    config = create_default_config()
    save_config(config, "config_example.json")

### Generate Single Signal
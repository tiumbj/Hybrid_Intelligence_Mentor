"""
Hybrid Intelligence Mentor (HIM)
Trading Engine

Version: 2.4.0

Changelog:
2.4.0
- Add breakout confirmed states when BOS already crossed (distance < 0)
- Add breakout_state/breakout_side/breakout_overshoot_points to context
- Keep ATR-based threshold + points fallback
- Reload config per call (API config changes apply immediately)
"""

import json
import time
import numpy as np
import MetaTrader5 as mt5


class TradingEngine:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.cfg = self.load_config()

    # =========================
    # SAFE CONVERSION
    # =========================

    @staticmethod
    def safe_float(v, default=0.0):
        if v is None:
            return float(default)
        try:
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def safe_int(v, default=0):
        if v is None:
            return int(default)
        try:
            return int(v)
        except Exception:
            return int(default)

    # =========================
    # CONFIG
    # =========================

    def load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}

    def reload_config(self):
        self.cfg = self.load_config()
        return self.cfg

    # =========================
    # MT5 READY
    # =========================

    def ensure_mt5(self):
        try:
            if mt5.terminal_info() is not None:
                return True, "already_initialized"
            ok = mt5.initialize()
            if not ok:
                return False, f"initialize_failed: {mt5.last_error()}"
            return True, "initialized"
        except Exception as e:
            return False, f"initialize_exception: {e}"

    # =========================
    # TF MAP
    # =========================

    def tf(self, name):
        mapping = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        key = str(name).upper().strip()
        return mapping.get(key, mt5.TIMEFRAME_M5)

    # =========================
    # DATA
    # =========================

    def get_data(self, symbol, timeframe, bars):
        ok, msg = self.ensure_mt5()
        if not ok:
            raise RuntimeError(msg)

        sel = mt5.symbol_select(symbol, True)
        if not sel:
            raise RuntimeError(f"symbol_select_failed: {symbol} last_error={mt5.last_error()}")

        bars = int(bars)

        def fetch_once():
            return mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

        rates = fetch_once()
        if rates is None or len(rates) == 0:
            time.sleep(0.2)
            rates = fetch_once()

        if rates is None:
            raise RuntimeError(f"copy_rates_none: symbol={symbol} tf={timeframe} bars={bars} last_error={mt5.last_error()}")

        if len(rates) < 120:
            raise RuntimeError(f"not_enough_bars: got={len(rates)} need>=120 symbol={symbol} tf={timeframe}")

        return rates

    # =========================
    # ATR
    # =========================

    def atr(self, high, low, close, period):
        period = max(1, int(period))

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )

        atr = np.zeros(len(close), dtype=float)
        alpha = 1.0 / period

        for i in range(1, len(close)):
            atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha

        return atr

    # =========================
    # STRUCTURE
    # =========================

    def structure(self, data, sens):
        high = data["high"]
        low = data["low"]

        sens = max(1, int(sens))

        piv_hi = []
        piv_lo = []

        for i in range(sens, len(high) - sens):
            if high[i] == max(high[i - sens:i + sens + 1]):
                piv_hi.append(float(high[i]))
            if low[i] == min(low[i - sens:i + sens + 1]):
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

    # =========================
    # PROXIMITY / ETA
    # =========================

    def proximity_score(self, dist_buy, dist_sell, threshold_points):
        thr = max(self.safe_float(threshold_points, 0.0), 1e-9)

        best = None
        side = None

        # Only “pre-breakout” distances (>=0) contribute to proximity score
        if dist_buy is not None:
            try:
                db = float(dist_buy)
                if db >= 0:
                    best = db
                    side = "BUY"
            except Exception:
                pass

        if dist_sell is not None:
            try:
                ds = float(dist_sell)
                if ds >= 0:
                    if best is None or ds < best:
                        best = ds
                        side = "SELL"
            except Exception:
                pass

        if best is None:
            return 0.0, None, None

        score = 100.0 * max(0.0, min(1.0, (thr - best) / thr))
        return float(score), side, float(best)

    def eta_candles(self, dist_points, atr_points_per_candle):
        if dist_points is None:
            return None
        atrv = self.safe_float(atr_points_per_candle, 0.0)
        if atrv <= 1e-9:
            return None
        d = float(dist_points)
        if d < 0:
            return 0.0
        return float(d / atrv)

    # =========================
    # MAIN
    # =========================

    def generate_signal_package(self):
        cfg = self.reload_config()
        symbol = str(cfg.get("symbol", "GOLD"))

        tf_cfg = cfg.get("timeframes", {}) or {}
        sens_cfg = cfg.get("structure_sensitivity", {}) or {}
        risk_cfg = cfg.get("risk", {}) or {}
        prox_cfg = cfg.get("breakout_proximity", {}) or {}

        htf = str(tf_cfg.get("htf", "H1"))
        mtf = str(tf_cfg.get("mtf", "M30"))
        ltf = str(tf_cfg.get("ltf", "M5"))

        sens_htf = self.safe_int(sens_cfg.get("htf"), 4)
        sens_mtf = self.safe_int(sens_cfg.get("mtf"), 3)
        sens_ltf = self.safe_int(sens_cfg.get("ltf"), 1)

        atr_period = self.safe_int(risk_cfg.get("atr_period"), 14)

        prox_min_score = self.safe_float(prox_cfg.get("min_score"), 40.0)
        threshold_atr_raw = prox_cfg.get("threshold_atr", None)
        threshold_points_fallback = prox_cfg.get("threshold", None)

        blocked = []
        watch_state = "NONE"
        breakout_state = "NONE"
        breakout_side = None
        breakout_overshoot_points = None

        try:
            htf_data = self.get_data(symbol, self.tf(htf), 600)
            mtf_data = self.get_data(symbol, self.tf(mtf), 800)
            ltf_data = self.get_data(symbol, self.tf(ltf), 1200)
        except Exception as e:
            return {
                "symbol": symbol,
                "direction": "NONE",
                "context": {
                    "blocked_by": "no_data",
                    "error": str(e),
                    "watch_state": "NONE",
                    "breakout_state": "NONE",
                    "timeframes": {"htf": htf, "mtf": mtf, "ltf": ltf},
                },
            }

        htf_trend, _, _ = self.structure(htf_data, sens_htf)
        mtf_trend, _, _ = self.structure(mtf_data, sens_mtf)
        ltf_trend, bos_hi, bos_lo = self.structure(ltf_data, sens_ltf)

        direction = "NONE"
        bias_source = "NO_CLEAR_BIAS"
        if htf_trend == "bullish":
            direction = "BUY"
            bias_source = "HTF"
        elif htf_trend == "bearish":
            direction = "SELL"
            bias_source = "HTF"
        elif mtf_trend == "bullish":
            direction = "BUY"
            bias_source = "MTF_FALLBACK"
        elif mtf_trend == "bearish":
            direction = "SELL"
            bias_source = "MTF_FALLBACK"
        else:
            blocked.append("no_clear_bias")

        close = np.asarray(ltf_data["close"], dtype=float)
        high = np.asarray(ltf_data["high"], dtype=float)
        low = np.asarray(ltf_data["low"], dtype=float)
        price = float(close[-1])

        atr_arr = self.atr(high, low, close, atr_period)
        atr_val = float(atr_arr[-1]) if len(atr_arr) else 0.0

        thr_atr = self.safe_float(threshold_atr_raw, default=0.0)
        if thr_atr > 0 and atr_val > 0:
            threshold_points = float(atr_val * thr_atr)
            threshold_mode = "ATR"
            threshold_atr_out = thr_atr
        else:
            threshold_points = self.safe_float(threshold_points_fallback, default=2.5)
            threshold_mode = "POINTS"
            threshold_atr_out = None

        threshold_points = max(float(threshold_points), 1e-9)

        dist_buy = (float(bos_hi) - price) if bos_hi is not None else None
        dist_sell = (price - float(bos_lo)) if bos_lo is not None else None

        # Breakout confirmed detection (distance < 0)
        if dist_buy is not None and dist_buy < 0:
            breakout_state = "BREAKOUT_BUY_CONFIRMED"
            breakout_side = "BUY"
            breakout_overshoot_points = float(abs(dist_buy))
        elif dist_sell is not None and dist_sell < 0:
            breakout_state = "BREAKOUT_SELL_CONFIRMED"
            breakout_side = "SELL"
            breakout_overshoot_points = float(abs(dist_sell))

        prox_score, prox_side, prox_best_dist = self.proximity_score(dist_buy, dist_sell, threshold_points)

        eta_buy = self.eta_candles(dist_buy, atr_val) if dist_buy is not None else None
        eta_sell = self.eta_candles(dist_sell, atr_val) if dist_sell is not None else None
        eta_best = self.eta_candles(prox_best_dist, atr_val) if prox_best_dist is not None else None

        # Direction gates (only for BUY/SELL)
        if direction == "BUY":
            if dist_buy is None:
                blocked.append("no_bos_ref_high")
            elif dist_buy >= 0 and dist_buy > threshold_points:
                blocked.append("no_bos")
        elif direction == "SELL":
            if dist_sell is None:
                blocked.append("no_bos_ref_low")
            elif dist_sell >= 0 and dist_sell > threshold_points:
                blocked.append("no_bos")

        # WATCHLIST rule:
        # 1) If bias unclear but near BOS => WATCH_*_BREAKOUT
        # 2) If breakout already happened => WATCH_*_BREAKOUT_CONFIRMED
        if "no_clear_bias" in blocked:
            if breakout_state == "BREAKOUT_BUY_CONFIRMED":
                watch_state = "WATCH_BUY_BREAKOUT_CONFIRMED"
            elif breakout_state == "BREAKOUT_SELL_CONFIRMED":
                watch_state = "WATCH_SELL_BREAKOUT_CONFIRMED"
            elif prox_score >= prox_min_score and prox_side in ("BUY", "SELL"):
                watch_state = "WATCH_BUY_BREAKOUT" if prox_side == "BUY" else "WATCH_SELL_BREAKOUT"

        direction_out = "NONE" if blocked else direction

        return {
            "symbol": symbol,
            "direction": direction_out,
            "context": {
                "HTF_trend": htf_trend,
                "MTF_trend": mtf_trend,
                "LTF_trend": ltf_trend,
                "bias_source": bias_source,

                "bos_ref_high": bos_hi,
                "bos_ref_low": bos_lo,

                "distance_buy": dist_buy,
                "distance_sell": dist_sell,

                "atr": atr_val,

                "breakout_proximity_min_score": prox_min_score,
                "breakout_proximity_threshold_atr": threshold_atr_out,
                "breakout_proximity_threshold_mode": threshold_mode,
                "breakout_proximity_threshold_points": threshold_points,

                "proximity_score": prox_score,
                "proximity_side": prox_side,

                "eta_candles_to_bos_best": eta_best,
                "eta_candles_to_bos_buy": eta_buy,
                "eta_candles_to_bos_sell": eta_sell,

                "breakout_state": breakout_state,
                "breakout_side": breakout_side,
                "breakout_overshoot_points": breakout_overshoot_points,

                "watch_state": watch_state,
                "blocked_by": ",".join(blocked) if blocked else None,

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
    print(pkg)
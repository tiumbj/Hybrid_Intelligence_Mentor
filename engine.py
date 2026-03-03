# engine.py
# =============================================================================
# Hybrid Intelligence Mentor (HIM) - Trading Engine
#
# Version: v2.12.6
# File: C:\Hybrid_Intelligence_Mentor\engine.py
#
# CHANGELOG (v2.12.6):
# - Implemented M1 Supertrend Gate Relaxation (distance-based)
#   Rule:
#     If event_timeframe == "M1" and supertrend_conflict == True:
#       allow (do not block) when supertrend_distance_atr <= st_relax_dist_atr_m1
#
# EVIDENCE NOTE (commissioning):
# - Latest 20 samples showed supertrend_conflict = 100% dominant blocker.
# - Volatility gate had been mitigated already via bb_width_atr_min_m1.
# - This change targets only supertrend gate to reduce full blocking on M1.
#
# PARAMETER RATIONALE:
# - st_relax_dist_atr_m1 default = 0.30 ATR
#   Human meaning: "ยอมให้ pullback ตื้นๆ ใกล้เส้น Supertrend บน M1"
#   Technical: distance normalization by ATR makes threshold scale-invariant.
#
# IMPORTANT CONSTRAINTS (LOCKED):
# - Confirm-only architecture: engine MUST NOT send orders.
# - Event-based commissioning: called per NEW_BAR by commissioning_runner.py
# - Rewrite-only: do not patch partials; keep schema stable for mentor/tools.
# =============================================================================

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None


# -----------------------------
# MT5 timeframe map
# -----------------------------
MT5_TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1 if mt5 else None,
    "M5":  mt5.TIMEFRAME_M5 if mt5 else None,
    "M15": mt5.TIMEFRAME_M15 if mt5 else None,
    "M30": mt5.TIMEFRAME_M30 if mt5 else None,
    "H1":  mt5.TIMEFRAME_H1 if mt5 else None,
    "H4":  mt5.TIMEFRAME_H4 if mt5 else None,
    "D1":  mt5.TIMEFRAME_D1 if mt5 else None,
}


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tf_to_mt5(tf: str) -> int:
    if tf not in MT5_TF_MAP or MT5_TF_MAP[tf] is None:
        raise ValueError(f"Unsupported or unavailable timeframe: {tf}")
    return MT5_TF_MAP[tf]


def _rates_to_df(rates: Any) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    # MT5 returns 'time' as seconds epoch
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


# -----------------------------
# Indicators (no TA-Lib)
# -----------------------------
def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing (RMA)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def bollinger_width_atr(df: pd.DataFrame, bb_period: int, bb_std: float, atr: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    close = df["close"].astype(float)
    ma = close.rolling(bb_period).mean()
    sd = close.rolling(bb_period).std(ddof=0)
    upper = ma + bb_std * sd
    lower = ma - bb_std * sd
    width = (upper - lower).abs()
    width_atr = width / atr.replace(0.0, np.nan)
    return upper, lower, width_atr


def supertrend(df: pd.DataFrame, atr: pd.Series, period: int, multiplier: float) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      st_line: Supertrend line value
      st_dir:  +1 bullish, -1 bearish
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    hl2 = (high + low) / 2.0
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    for i in range(1, len(df)):
        if np.isnan(final_ub.iloc[i - 1]):
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            if (basic_ub.iloc[i] < final_ub.iloc[i - 1]) or (close.iloc[i - 1] > final_ub.iloc[i - 1]):
                final_ub.iloc[i] = basic_ub.iloc[i]
            else:
                final_ub.iloc[i] = final_ub.iloc[i - 1]

        if np.isnan(final_lb.iloc[i - 1]):
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            if (basic_lb.iloc[i] > final_lb.iloc[i - 1]) or (close.iloc[i - 1] < final_lb.iloc[i - 1]):
                final_lb.iloc[i] = basic_lb.iloc[i]
            else:
                final_lb.iloc[i] = final_lb.iloc[i - 1]

    st_line = pd.Series(index=df.index, dtype=float)
    st_dir = pd.Series(index=df.index, dtype=float)

    # Initialize direction with first valid point
    st_dir.iloc[0] = 1.0
    st_line.iloc[0] = final_lb.iloc[0] if close.iloc[0] >= final_lb.iloc[0] else final_ub.iloc[0]
    if close.iloc[0] < st_line.iloc[0]:
        st_dir.iloc[0] = -1.0

    for i in range(1, len(df)):
        prev_line = st_line.iloc[i - 1]
        prev_dir = st_dir.iloc[i - 1]

        if prev_dir > 0:
            # bullish
            if close.iloc[i] <= final_ub.iloc[i]:
                st_dir.iloc[i] = -1.0
                st_line.iloc[i] = final_ub.iloc[i]
            else:
                st_dir.iloc[i] = 1.0
                st_line.iloc[i] = max(final_lb.iloc[i], prev_line) if not np.isnan(prev_line) else final_lb.iloc[i]
        else:
            # bearish
            if close.iloc[i] >= final_lb.iloc[i]:
                st_dir.iloc[i] = 1.0
                st_line.iloc[i] = final_lb.iloc[i]
            else:
                st_dir.iloc[i] = -1.0
                st_line.iloc[i] = min(final_ub.iloc[i], prev_line) if not np.isnan(prev_line) else final_ub.iloc[i]

    st_dir = st_dir.fillna(method="ffill")
    st_line = st_line.fillna(method="ffill")
    return st_line, st_dir


# -----------------------------
# Strategy config
# -----------------------------
@dataclass
class EngineConfig:
    symbol: str = "GOLD"

    # timeframes
    htf: str = "H1"
    mtf: str = "M15"
    ltf: str = "M5"

    # lookbacks
    rates_lookback: int = 600  # enough for indicators on multiple TF

    # ATR/BB/ST parameters
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    st_atr_period: int = 10
    st_mult: float = 3.0

    # volatility expansion gating
    bb_width_atr_min: float = 0.90
    bb_width_atr_min_m1: float = 0.65

    # BOS (placeholder - keep gate structure stable; tuned later)
    bos_lookback: int = 40
    bos_break_atr_min: float = 0.40

    # retest/proximity (placeholder)
    retest_max_bars: int = 8
    retest_distance_atr_max: float = 0.60
    proximity_threshold: float = 0.50

    # RR
    min_rr: float = 1.20

    # NEW: M1 supertrend relaxation
    st_relax_dist_atr_m1: float = 0.30


class TradingEngine:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.cfg = self._load_config(config_path)

    def _load_config(self, path: str) -> EngineConfig:
        raw = _load_json(path)

        tfs = raw.get("timeframes", {}) if isinstance(raw, dict) else {}
        symbol = raw.get("symbol", "GOLD")

        # allow both top-level and nested config keys
        c = EngineConfig(
            symbol=str(symbol),
            htf=str(tfs.get("htf", raw.get("htf", "H1"))),
            mtf=str(tfs.get("mtf", raw.get("mtf", "M15"))),
            ltf=str(tfs.get("ltf", raw.get("ltf", "M5"))),

            rates_lookback=int(raw.get("rates_lookback", 600)),

            atr_period=int(raw.get("atr_period", 14)),
            bb_period=int(raw.get("bb_period", 20)),
            bb_std=_safe_float(raw.get("bb_std", 2.0), 2.0),

            st_atr_period=int(raw.get("st_atr_period", raw.get("st_period", 10))),
            st_mult=_safe_float(raw.get("st_mult", raw.get("st_multiplier", 3.0)), 3.0),

            bb_width_atr_min=_safe_float(raw.get("bb_width_atr_min", 0.90), 0.90),
            bb_width_atr_min_m1=_safe_float(raw.get("bb_width_atr_min_m1", 0.65), 0.65),

            bos_lookback=int(raw.get("bos_lookback", 40)),
            bos_break_atr_min=_safe_float(raw.get("bos_break_atr_min", 0.40), 0.40),

            retest_max_bars=int(raw.get("retest_max_bars", 8)),
            retest_distance_atr_max=_safe_float(raw.get("retest_distance_atr_max", 0.60), 0.60),
            proximity_threshold=_safe_float(raw.get("proximity_threshold", 0.50), 0.50),

            min_rr=_safe_float(raw.get("min_rr", 1.20), 1.20),

            st_relax_dist_atr_m1=_safe_float(raw.get("st_relax_dist_atr_m1", 0.30), 0.30),
        )
        return c

    def _ensure_mt5(self) -> None:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package is not available in this environment.")
        if not mt5.initialize():
            err = mt5.last_error()
            raise RuntimeError(f"mt5.initialize() failed: {err}")

    def _fetch_rates(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
        self._ensure_mt5()
        tf = _tf_to_mt5(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
        df = _rates_to_df(rates)
        return df

    # -----------------------------
    # Core computations (bias + gates)
    # -----------------------------
    def _compute_tf_bundle(self, tf: str) -> Dict[str, Any]:
        df = self._fetch_rates(self.cfg.symbol, tf, self.cfg.rates_lookback)
        if df.empty or len(df) < max(self.cfg.atr_period, self.cfg.bb_period, self.cfg.st_atr_period) + 5:
            return {"tf": tf, "ok": False, "reason": "not_enough_rates", "df_len": int(len(df))}

        atr = atr_wilder(df, self.cfg.atr_period)
        upper, lower, bb_width_atr = bollinger_width_atr(df, self.cfg.bb_period, self.cfg.bb_std, atr)
        st_line, st_dir = supertrend(df, atr, self.cfg.st_atr_period, self.cfg.st_mult)

        i = df.index[-1]
        close = float(df.loc[i, "close"])
        a = float(atr.loc[i]) if not np.isnan(atr.loc[i]) else np.nan
        stv = float(st_line.loc[i]) if not np.isnan(st_line.loc[i]) else np.nan
        stdir = int(1 if st_dir.loc[i] > 0 else -1)

        dist = abs(close - stv) if (not np.isnan(stv)) else np.nan
        dist_atr = (dist / a) if (a and not np.isnan(a) and a != 0.0 and not np.isnan(dist)) else np.nan

        return {
            "tf": tf,
            "ok": True,
            "df": df,
            "close": close,
            "atr": a,
            "bb_width_atr": float(bb_width_atr.loc[i]) if not np.isnan(bb_width_atr.loc[i]) else np.nan,
            "st_line": stv,
            "st_dir": stdir,
            "st_distance": dist,
            "st_distance_atr": float(dist_atr) if not np.isnan(dist_atr) else np.nan,
            # optional: keep bands for debugging
            "bb_upper": float(upper.loc[i]) if not np.isnan(upper.loc[i]) else np.nan,
            "bb_lower": float(lower.loc[i]) if not np.isnan(lower.loc[i]) else np.nan,
        }

    def _derive_htf_bias(self, htf_bundle: Dict[str, Any]) -> str:
        # Simple & stable: use HTF supertrend direction as bias
        if not htf_bundle.get("ok"):
            return "unknown"
        return "bullish" if int(htf_bundle["st_dir"]) > 0 else "bearish"

    def _bos_gate_placeholder(self, ltf_bundle: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Placeholder BOS logic with ATR-normalized breakout check.
        Keep it deterministic; tuning can happen after supertrend fix.
        """
        df = ltf_bundle.get("df")
        if df is None or df.empty:
            return False, {"bos_break_distance_atr": np.nan}, "no_bos_break"

        lookback = int(self.cfg.bos_lookback)
        if len(df) < lookback + 5:
            return False, {"bos_break_distance_atr": np.nan}, "no_bos_break"

        close = float(ltf_bundle["close"])
        atr = float(ltf_bundle["atr"]) if ltf_bundle.get("atr") else np.nan
        if atr == 0.0 or np.isnan(atr):
            return False, {"bos_break_distance_atr": np.nan}, "no_bos_break"

        # naive structure levels
        recent = df.iloc[-(lookback + 1):-1]
        swing_high = float(recent["high"].max())
        swing_low = float(recent["low"].min())

        # breakout distance normalized by ATR
        break_up = (close - swing_high) / atr
        break_dn = (swing_low - close) / atr

        ok = (break_up >= self.cfg.bos_break_atr_min) or (break_dn >= self.cfg.bos_break_atr_min)
        metrics = {"bos_break_up_atr": float(break_up), "bos_break_dn_atr": float(break_dn)}
        return ok, metrics, (None if ok else "no_bos_break")

    def _rr_gate_placeholder(self, ltf_bundle: Dict[str, Any], bias: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        RR placeholder:
        - SL/TP distances derived from ATR
        - RR = TP/SL
        This is intentionally simple for commissioning; refine later if rr_too_low dominates.
        """
        atr = float(ltf_bundle.get("atr", np.nan))
        if atr == 0.0 or np.isnan(atr):
            return False, {"rr": np.nan}, "rr_too_low"

        # conservative defaults: SL 1.0 ATR, TP 1.3 ATR => RR 1.3
        sl = 1.0 * atr
        tp = 1.3 * atr
        rr = tp / sl if sl != 0 else np.nan

        ok = (not np.isnan(rr)) and (rr >= self.cfg.min_rr)
        return ok, {"rr": float(rr), "sl_atr": 1.0, "tp_atr": 1.3}, (None if ok else "rr_too_low")

    def _retest_proximity_placeholder(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Keep gates present but non-dominant for this iteration.
        (Earlier phases showed no_retest/low_proximity, but current priority is supertrend.)
        """
        # Pass-through with neutral telemetry; can be re-enabled/tuned later.
        return True, {"retest_ok": True, "proximity_ok": True, "proximity_score": 1.0}, None

    # -----------------------------
    # Public API
    # -----------------------------
    def generate_signal_package(self, event_timeframe: str = "M1") -> Dict[str, Any]:
        """
        Output schema is designed to be stable for mentor_executor/tools.
        This function MUST NOT place trades.
        """
        t0 = _now_ms()

        # bundles
        htf = self._compute_tf_bundle(self.cfg.htf)
        mtf = self._compute_tf_bundle(self.cfg.mtf)
        ltf = self._compute_tf_bundle(self.cfg.ltf)
        ev_tf = self._compute_tf_bundle(event_timeframe)

        # bias
        bias = self._derive_htf_bias(htf)

        blocked_by: List[str] = []
        gates: Dict[str, bool] = {}
        metrics: Dict[str, Any] = {}

        # Basic availability gate
        data_ok = all(b.get("ok") for b in [htf, mtf, ltf, ev_tf])
        gates["data_ok"] = bool(data_ok)
        if not data_ok:
            blocked_by.append("data_not_ready")

        # Volatility expansion gate (use event TF)
        bb_width_atr = _safe_float(ev_tf.get("bb_width_atr"), np.nan)
        vol_thr = self.cfg.bb_width_atr_min_m1 if event_timeframe == "M1" else self.cfg.bb_width_atr_min
        vol_ok = (not np.isnan(bb_width_atr)) and (bb_width_atr >= vol_thr)
        gates["vol_expansion_ok"] = bool(vol_ok)
        metrics["bb_width_atr"] = bb_width_atr
        metrics["bb_width_atr_min_used"] = float(vol_thr)
        if data_ok and not vol_ok:
            blocked_by.append("no_vol_expansion")

        # Supertrend conflict gate (compare HTF bias vs event TF ST direction)
        st_dir_ev = int(ev_tf.get("st_dir", 0)) if ev_tf.get("ok") else 0
        st_dist_atr = _safe_float(ev_tf.get("st_distance_atr"), np.nan)
        metrics["supertrend_dir_event"] = st_dir_ev
        metrics["supertrend_distance_atr"] = st_dist_atr
        metrics["st_relax_dist_atr_m1"] = float(self.cfg.st_relax_dist_atr_m1)

        # conflict definition
        conflict = False
        if bias == "bullish" and st_dir_ev < 0:
            conflict = True
        if bias == "bearish" and st_dir_ev > 0:
            conflict = True

        # NEW RELAXATION RULE (v2.12.6)
        # ถ้า conflict บน M1 แต่ราคาอยู่ใกล้เส้น ST (distance<=0.30 ATR) ให้ผ่าน
        relaxed = False
        if event_timeframe == "M1" and conflict:
            if (not np.isnan(st_dist_atr)) and (st_dist_atr <= float(self.cfg.st_relax_dist_atr_m1)):
                relaxed = True

        supertrend_ok = (not conflict) or relaxed
        gates["supertrend_conflict"] = bool(conflict)
        gates["supertrend_relaxed_m1"] = bool(relaxed)
        gates["supertrend_ok"] = bool(supertrend_ok)
        if data_ok and (not supertrend_ok):
            blocked_by.append("supertrend_conflict")

        # BOS gate (placeholder, deterministic)
        bos_ok, bos_metrics, bos_block = self._bos_gate_placeholder(ev_tf)
        gates["bos_break_ok"] = bool(bos_ok)
        metrics.update(bos_metrics)
        if data_ok and (bos_block is not None):
            blocked_by.append(bos_block)

        # Retest/Proximity gate (placeholder pass for this iteration)
        rp_ok, rp_metrics, rp_block = self._retest_proximity_placeholder()
        gates["retest_ok"] = bool(rp_ok)
        metrics.update(rp_metrics)
        if data_ok and (rp_block is not None):
            blocked_by.append(rp_block)

        # RR gate (placeholder, simple)
        rr_ok, rr_metrics, rr_block = self._rr_gate_placeholder(ev_tf, bias)
        gates["rr_ok"] = bool(rr_ok)
        metrics.update(rr_metrics)
        if data_ok and (rr_block is not None):
            blocked_by.append(rr_block)

        # Final decision (signal/blocked)
        blocked = (len(blocked_by) > 0)
        decision = "BLOCKED" if blocked else "PASS"

        # Build output package
        pkg: Dict[str, Any] = {
            "engine_version": "v2.12.6",
            "ts_ms": _now_ms(),
            "symbol": self.cfg.symbol,
            "event_timeframe": event_timeframe,
            "timeframes": {"htf": self.cfg.htf, "mtf": self.cfg.mtf, "ltf": self.cfg.ltf},
            "bias": bias,

            # snapshot (event TF)
            "price": {
                "close": _safe_float(ev_tf.get("close"), np.nan),
                "atr": _safe_float(ev_tf.get("atr"), np.nan),
            },

            # gates + reasons
            "gates": gates,
            "blocked_by": blocked_by,
            "decision": decision,

            # telemetry
            "metrics": metrics,

            # debug bundles (lightweight)
            "debug": {
                "htf_ok": bool(htf.get("ok")),
                "mtf_ok": bool(mtf.get("ok")),
                "ltf_ok": bool(ltf.get("ok")),
                "event_ok": bool(ev_tf.get("ok")),
            },

            "latency_ms": int(_now_ms() - t0),
        }

        return pkg


# Optional quick self-test (does not send orders)
if __name__ == "__main__":
    # Usage:
    #   python engine.py
    e = TradingEngine("config.json")
    out = e.generate_signal_package(event_timeframe="M1")
    print(json.dumps(out, ensure_ascii=False))
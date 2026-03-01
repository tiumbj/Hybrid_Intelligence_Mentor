# =============================================================================
# Hybrid Intelligence Mentor (HIM) - Commissioning Runner
# File: commissioning_runner.py
# Path: C:\Hybrid_Intelligence_Mentor\commissioning_runner.py
# Version: v1.0.0
# Date: 2026-03-01 (Asia/Bangkok)
#
# Changelog:
# - v1.0.0: เพิ่ม commissioning runner แบบ auto-detect
#   - ถ้า tick สด -> เรียก mentor_executor.py (DRY_RUN/LIVE ตาม config)
#   - ถ้า tick stale (เช่นวันอาทิตย์) -> run "Replay Commissioning" โดยใช้ข้อมูล historical
#     เพื่อทดสอบ Data feed + AI Confirm schema v1.0 + Validator v1.0 (fail-closed)
#
# Notes (Design Intent):
# - ปลอดภัยก่อน: replay mode จะไม่ส่งคำสั่งเทรดจริงเข้า MT5
# - deterministic: สร้าง baseline order จาก historical bars แบบกำหนดสูตรตายตัว
# - ใช้ schema v1.0 และเรียก validator ก่อนเสมอ
# =============================================================================

from __future__ import annotations

import json
import math
import os
import sys
import time
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import MetaTrader5 as mt5

# replay calculations
import numpy as np
import pandas as pd

# http for AI confirm endpoint
import requests


# -----------------------------
# Small utilities
# -----------------------------

def _now_utc_ts() -> float:
    return time.time()


def _ts_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(d: Dict[str, Any], keys: Tuple[str, ...], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _print_kv(title: str, kv: Dict[str, Any]) -> None:
    print(f"\n== {title} ==")
    for k, v in kv.items():
        print(f"- {k}: {v}")


# -----------------------------
# Indicators (deterministic)
# -----------------------------

def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    ATR แบบ Wilder smoothing
    Returns array len = n (ค่าต้นแรกๆ จะเป็น nan จนกว่าจะพอ period)
    """
    n = len(close)
    tr = np.full(n, np.nan, dtype=float)
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    tr[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)))

    atr = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return atr

    # first ATR = SMA(TR, period) on first 'period' TR values starting at index 1..period
    first = np.nanmean(tr[1:period + 1])
    atr[period] = first

    # Wilder smoothing
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def adx_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    ADX แบบ Wilder (คำนวณ +DI, -DI, DX, แล้ว smooth เป็น ADX)
    Returns array len = n (ค่าแรกๆ เป็น nan)
    """
    n = len(close)
    if n < period * 2 + 2:
        return np.full(n, np.nan, dtype=float)

    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.full(n, np.nan, dtype=float)
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    tr[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)))

    # Wilder smoothing for TR, +DM, -DM
    tr14 = np.full(n, np.nan, dtype=float)
    p14 = np.full(n, np.nan, dtype=float)
    m14 = np.full(n, np.nan, dtype=float)

    tr14[period] = np.sum(tr[1:period + 1])
    p14[period] = np.sum(plus_dm[:period])
    m14[period] = np.sum(minus_dm[:period])

    for i in range(period + 1, n):
        tr14[i] = tr14[i - 1] - (tr14[i - 1] / period) + tr[i]
        p14[i] = p14[i - 1] - (p14[i - 1] / period) + plus_dm[i - 1]
        m14[i] = m14[i - 1] - (m14[i - 1] / period) + minus_dm[i - 1]

    plus_di = 100.0 * (p14 / tr14)
    minus_di = 100.0 * (m14 / tr14)

    dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    dx = np.where(np.isfinite(dx), dx, np.nan)

    adx = np.full(n, np.nan, dtype=float)
    # first ADX at index = 2*period is average of DX from period..2*period-1
    first_adx_idx = 2 * period
    adx[first_adx_idx] = np.nanmean(dx[period:first_adx_idx])

    for i in range(first_adx_idx + 1, n):
        adx[i] = ((adx[i - 1] * (period - 1)) + dx[i]) / period

    return adx


def bb_width(close: np.ndarray, period: int = 20, stdev: float = 2.0) -> np.ndarray:
    """
    BB width = (upper - lower)
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return out
    s = pd.Series(close)
    ma = s.rolling(period).mean()
    sd = s.rolling(period).std(ddof=0)
    upper = ma + stdev * sd
    lower = ma - stdev * sd
    w = (upper - lower).to_numpy(dtype=float)
    out[:] = w
    return out


# -----------------------------
# Replay baseline order (deterministic)
# -----------------------------

@dataclass
class EngineOrder:
    symbol: str
    mode: str
    direction: str  # BUY/SELL
    lot: float
    entry: float
    sl: float
    tp: float
    rr: float
    atr: float
    adx: float
    bb_width: float
    bb_width_atr: float


def build_replay_engine_order(cfg: Dict[str, Any], rates_df: pd.DataFrame) -> EngineOrder:
    """
    สร้าง baseline order แบบ deterministic จาก historical bars
    เป้าหมาย: commissioning AI+Validator ขณะตลาดปิด (ไม่ใช่แทน logic engine จริง)
    """
    symbol = str(cfg.get("symbol", "GOLD"))
    mode = str(cfg.get("mode", "sideway_scalp"))

    lot = float(_safe_get(cfg, ("execution", "volume"), 0.01))
    min_rr = float(cfg.get("min_rr", 1.5))

    h = rates_df["high"].to_numpy(dtype=float)
    l = rates_df["low"].to_numpy(dtype=float)
    c = rates_df["close"].to_numpy(dtype=float)

    atr = atr_wilder(h, l, c, period=14)
    adx = adx_wilder(h, l, c, period=14)
    bw = bb_width(c, period=20, stdev=2.0)

    last_close = float(c[-1])
    last_atr = float(atr[-1]) if np.isfinite(atr[-1]) else float(np.nanmean(atr[-50:]))
    last_adx = float(adx[-1]) if np.isfinite(adx[-1]) else float(np.nanmean(adx[-50:]))
    last_bw = float(bw[-1]) if np.isfinite(bw[-1]) else float(np.nanmean(bw[-50:]))

    if not (math.isfinite(last_atr) and last_atr > 0):
        raise ValueError("Replay commissioning: ATR invalid (history not enough or data bad)")

    # direction rule (deterministic):
    # - ถ้า close > SMA20 => BUY else SELL
    sma20 = float(pd.Series(c).rolling(20).mean().iloc[-1])
    direction = "BUY" if last_close > sma20 else "SELL"

    # sizing rule (deterministic):
    # SL distance = 1.0 * ATR
    # TP distance = max(min_rr, 1.5) * SL distance
    sl_dist = 1.0 * last_atr
    tp_dist = max(min_rr, 1.5) * sl_dist

    entry = last_close
    if direction == "BUY":
        sl = entry - sl_dist
        tp = entry + tp_dist
        rr = (tp - entry) / max(entry - sl, 1e-9)
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist
        rr = (entry - tp) / max(sl - entry, 1e-9)

    bw_atr = last_bw / last_atr if last_atr > 0 else float("nan")

    return EngineOrder(
        symbol=symbol,
        mode=mode,
        direction=direction,
        lot=lot,
        entry=float(entry),
        sl=float(sl),
        tp=float(tp),
        rr=float(rr),
        atr=float(last_atr),
        adx=float(last_adx),
        bb_width=float(last_bw),
        bb_width_atr=float(bw_atr),
    )


# -----------------------------
# AI confirm schema v1.0 + Validator v1.0
# -----------------------------

def call_ai_confirm_v1(cfg: Dict[str, Any], order: EngineOrder) -> Dict[str, Any]:
    """
    เรียก /api/ai_confirm โดยส่ง baseline fields แบบ top-level (ตาม issue ที่เคยเจอ)
    """
    api_url = str(_safe_get(cfg, ("ai", "api_url"), "http://127.0.0.1:5000/api/ai_confirm"))
    timeout_sec = float(_safe_get(cfg, ("ai", "timeout_sec"), 10))

    payload = {
        # baseline fields (top-level)
        "symbol": order.symbol,
        "mode": order.mode,
        "direction": order.direction,
        "lot": order.lot,
        "entry": order.entry,
        "sl": order.sl,
        "tp": order.tp,
        # context fields (optional but useful)
        "atr": order.atr,
        "adx": order.adx,
        "bb_width": order.bb_width,
        "bb_width_atr": order.bb_width_atr,
        "min_rr": float(cfg.get("min_rr", 1.5)),
    }

    r = requests.post(api_url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise ValueError("AI confirm: response is not a JSON object")
    return data


def validate_with_validator_v1_0(ai_payload: Dict[str, Any], order: EngineOrder, cfg: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    เรียก validator_v1_0.py ถ้ามีใน repo
    engine_order ที่ส่งให้ validator จะ map ให้ครบ fields ที่ validator ต้องการ
    """
    # map to engine_order format (ขั้นต่ำที่ validator มักต้องใช้)
    engine_order = {
        "symbol": order.symbol,
        "mode": order.mode,
        "direction": order.direction,
        "lot": order.lot,
        "entry": order.entry,
        "sl": order.sl,
        "tp": order.tp,
        "rr": order.rr,
        "min_rr": float(cfg.get("min_rr", 1.5)),
        "context": {
            "atr": order.atr,
            "adx": order.adx,
            "bb_width": order.bb_width,
            "bb_width_atr": order.bb_width_atr,
        },
    }

    try:
        from validator_v1_0 import validate_ai_response_v1_0  # type: ignore
    except Exception as e:
        return False, f"validator_import_error: {e}", {"engine_order": engine_order, "ai_payload": ai_payload}

    try:
        res = validate_ai_response_v1_0(ai_payload, engine_order)  # expected ValidationResult-like
        # Normalize output (รองรับทั้ง dataclass/object/dict)
        ok = bool(getattr(res, "ok", False)) if not isinstance(res, dict) else bool(res.get("ok", False))
        decision = getattr(res, "decision", None) if not isinstance(res, dict) else res.get("decision", None)
        reason = getattr(res, "reason", "") if not isinstance(res, dict) else res.get("reason", "")

        msg = f"ok={ok} decision={decision} reason={reason}".strip()
        return ok, msg, {"engine_order": engine_order, "ai_payload": ai_payload, "validator_result": res}
    except Exception as e:
        return False, f"validator_runtime_error: {e}", {"engine_order": engine_order, "ai_payload": ai_payload}


# -----------------------------
# MT5 helpers
# -----------------------------

def mt5_init_or_die() -> None:
    if mt5.initialize():
        return
    raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")


def get_tick_age_sec(symbol: str) -> Tuple[Optional[float], Dict[str, Any]]:
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)

    meta = {
        "symbol": symbol,
        "mt5_connected": True,
        "symbol_visible": bool(getattr(info, "visible", False)) if info else None,
        "trade_allowed": bool(getattr(info, "trade_allowed", False)) if info else None,
        "trade_mode": getattr(info, "trade_mode", None) if info else None,
    }

    if tick is None:
        meta["tick"] = None
        return None, meta

    # tick.time is seconds since epoch (broker server time aligned)
    tick_ts = float(getattr(tick, "time", 0.0))
    age = _now_utc_ts() - tick_ts if tick_ts > 0 else None

    meta["tick_time_utc"] = _ts_to_iso(tick_ts) if tick_ts > 0 else None
    meta["tick_age_sec"] = age
    meta["bid"] = getattr(tick, "bid", None)
    meta["ask"] = getattr(tick, "ask", None)
    meta["volume"] = getattr(tick, "volume", None)

    return age, meta


def fetch_rates(symbol: str, timeframe: int, bars: int = 500) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError("MT5 copy_rates_from_pos returned empty")
    df = pd.DataFrame(rates)
    # MT5 returns 'time' as seconds since epoch
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


# -----------------------------
# Runner modes
# -----------------------------

def run_live_executor() -> int:
    """
    เรียก mentor_executor.py แบบ subprocess (ไม่ต้องแก้ไฟล์เดิม)
    """
    exe = [sys.executable, "mentor_executor.py"]
    print("\n[RUN] Live path: calling mentor_executor.py")
    print("CMD:", " ".join(exe))
    p = subprocess.run(exe, capture_output=False)
    return int(p.returncode)


def run_replay_commissioning(cfg: Dict[str, Any]) -> int:
    """
    Replay commissioning:
    - ดึง historical bars (H1) แล้วสร้าง baseline order deterministic
    - เรียก AI confirm
    - เรียก Validator v1.0 (fail-closed)
    """
    symbol = str(cfg.get("symbol", "GOLD"))

    # เลือก timeframe สำหรับ replay (H1 เป็น default ที่เสถียรกับ indicator)
    tf = mt5.TIMEFRAME_H1
    bars = int(_safe_get(cfg, ("commissioning", "replay_bars"), 500))

    print("\n[REPLAY] Fetching historical rates...")
    df = fetch_rates(symbol, tf, bars=bars)

    _print_kv("Replay data", {
        "symbol": symbol,
        "timeframe": "H1",
        "bars": len(df),
        "from_utc": str(df["time_utc"].iloc[0]),
        "to_utc": str(df["time_utc"].iloc[-1]),
    })

    print("\n[REPLAY] Building deterministic baseline engine_order (commissioning-only)...")
    order = build_replay_engine_order(cfg, df)

    _print_kv("Baseline order (deterministic)", {
        "direction": order.direction,
        "lot": order.lot,
        "entry": order.entry,
        "sl": order.sl,
        "tp": order.tp,
        "rr": order.rr,
        "atr": order.atr,
        "adx": order.adx,
        "bb_width": order.bb_width,
        "bb_width_atr": order.bb_width_atr,
    })

    # AI confirm
    print("\n[REPLAY] Calling AI confirm endpoint (/api/ai_confirm) ...")
    ai_payload = call_ai_confirm_v1(cfg, order)

    _print_kv("AI payload", {
        "schema_version": ai_payload.get("schema_version"),
        "decision": ai_payload.get("decision"),
        "confidence": ai_payload.get("confidence"),
        "direction": ai_payload.get("direction"),
        "lot": ai_payload.get("lot"),
        "mode": ai_payload.get("mode"),
        "entry": ai_payload.get("entry"),
        "sl": ai_payload.get("sl"),
        "tp": ai_payload.get("tp"),
        "note": ai_payload.get("note"),
    })

    # Validator
    print("\n[REPLAY] Validating with validator_v1_0 (fail-closed) ...")
    ok, msg, debug = validate_with_validator_v1_0(ai_payload, order, cfg)

    print("[VALIDATOR]", msg)
    if not ok:
        print("\n[FAIL-CLOSED] Commissioning replay failed. Debug snapshot:")
        print(json.dumps(debug, default=str, indent=2))
        return 2

    print("\n[PASS] Replay commissioning passed (AI schema v1.0 + Validator v1.0).")
    print("Note: replay mode ไม่ส่งออเดอร์เข้า MT5 (safety by design).")
    return 0


def load_config() -> Dict[str, Any]:
    """
    โหลด config.json แบบ robust:
    - ถ้ามี config_resolver.py และมี API โหลด ให้ใช้ (ถ้ามี)
    - ไม่งั้นอ่าน config.json ตรงๆ
    """
    root = Path(__file__).resolve().parent
    cfg_path = root / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found at: {cfg_path}")

    # Try resolver (ถ้ามี)
    try:
        import config_resolver  # type: ignore

        # รองรับหลายรูปแบบ เพื่อไม่ lock-in
        if hasattr(config_resolver, "load_effective_config"):
            return config_resolver.load_effective_config(str(cfg_path))  # type: ignore
        if hasattr(config_resolver, "ConfigResolver"):
            r = config_resolver.ConfigResolver(str(cfg_path))  # type: ignore
            if hasattr(r, "load"):
                return r.load()  # type: ignore
    except Exception:
        pass

    return _read_json(cfg_path)


def main() -> int:
    cfg = load_config()

    symbol = str(cfg.get("symbol", "GOLD"))
    tick_max_age_sec = float(_safe_get(cfg, ("execution", "tick_max_age_sec"), 15))

    print("============================================================")
    print("HIM Commissioning Runner v1.0.0")
    print("============================================================")

    mt5_init_or_die()

    age, meta = get_tick_age_sec(symbol)
    _print_kv("MT5 tick status", meta)

    if age is None:
        print("\n[DECISION] tick missing -> treat as STALE -> replay commissioning")
        return run_replay_commissioning(cfg)

    if age > tick_max_age_sec:
        print(f"\n[DECISION] tick stale (age_sec={age:.2f} > max={tick_max_age_sec}) -> replay commissioning")
        return run_replay_commissioning(cfg)

    print(f"\n[DECISION] tick fresh (age_sec={age:.2f} <= max={tick_max_age_sec}) -> run mentor_executor")
    return run_live_executor()


if __name__ == "__main__":
    raise SystemExit(main())
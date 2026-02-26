"""
performance_tracker.py
Version: 2.0.1
Changelog:
- 2.0.1: Add schema migration/sanitize for existing performance_stats.json to avoid undefined rows in dashboard.
Rules:
- Must be deterministic and testable
- Never call mt5.shutdown() (to avoid killing main session)
- Idempotent: do not double-count closed deals
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

import MetaTrader5 as mt5


@dataclass
class PerfSummary:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    net_profit: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    best_trade: float
    worst_trade: float


class PerformanceTracker:
    def __init__(
        self,
        stats_file: str = "performance_stats.json",
        symbol: str = "GOLD",
        lookback_days: int = 365,
    ):
        self.stats_file = stats_file
        self.symbol = symbol
        self.lookback_days = int(lookback_days)
        self.stats = self._load_stats()

    def _default_stats(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "lookback_days": self.lookback_days,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_profit": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "processed_deal_tickets": [],
            "trades": [],
            "last_sync_iso": None,
        }

    def _sanitize_trade(self, t: Any) -> Dict[str, Any] | None:
        """
        Normalize old/partial trade records so dashboard never sees undefined keys.
        Returns normalized dict or None if unusable.
        """
        if not isinstance(t, dict):
            return None

        # Must have at least time or time_iso to be useful; must have deal_ticket to identify.
        deal_ticket = t.get("deal_ticket")
        if deal_ticket is None:
            return None

        try:
            deal_ticket = int(deal_ticket)
        except Exception:
            return None

        time_sec = t.get("time")
        time_iso = t.get("time_iso")

        if time_sec is None and not time_iso:
            return None

        # derive time/time_iso if one is missing
        if time_sec is None:
            try:
                # best-effort parse "YYYY-MM-DD HH:MM:SS"
                dt = datetime.strptime(str(time_iso), "%Y-%m-%d %H:%M:%S")
                time_sec = int(dt.timestamp())
            except Exception:
                time_sec = 0

        if not time_iso:
            try:
                time_iso = datetime.fromtimestamp(int(time_sec)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_iso = "1970-01-01 00:00:00"

        # Ensure keys exist
        side = str(t.get("side") or "UNKNOWN").upper()
        if side not in ("BUY", "SELL", "UNKNOWN"):
            side = "UNKNOWN"

        try:
            volume = float(t.get("volume", 0.0))
        except Exception:
            volume = 0.0

        def fnum(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return float(default)

        net = fnum(t.get("net"), 0.0)
        profit = fnum(t.get("profit"), 0.0)
        commission = fnum(t.get("commission"), 0.0)
        swap = fnum(t.get("swap"), 0.0)

        position_id = t.get("position_id", 0)
        order_id = t.get("order_id", t.get("order", 0))
        try:
            position_id = int(position_id)
        except Exception:
            position_id = 0
        try:
            order_id = int(order_id)
        except Exception:
            order_id = 0

        symbol = str(t.get("symbol") or self.symbol)

        return {
            "type": "trade_close",
            "symbol": symbol,
            "deal_ticket": deal_ticket,
            "position_id": position_id,
            "order_id": order_id,
            "time": int(time_sec),
            "time_iso": str(time_iso),
            "side": side,
            "volume": float(volume),
            "profit": profit,
            "commission": commission,
            "swap": swap,
            "net": net,
        }

    def _load_stats(self) -> Dict[str, Any]:
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if isinstance(data, dict):
                    data.setdefault("processed_deal_tickets", [])
                    data.setdefault("trades", [])

                    # sanitize trades (remove bad/old schema rows)
                    raw_trades = data.get("trades", [])
                    cleaned: List[Dict[str, Any]] = []
                    if isinstance(raw_trades, list):
                        for t in raw_trades:
                            nt = self._sanitize_trade(t)
                            if nt is not None and str(nt.get("symbol")) == self.symbol:
                                cleaned.append(nt)

                    data["trades"] = cleaned

                    # sanitize processed_deal_tickets
                    p = data.get("processed_deal_tickets", [])
                    if not isinstance(p, list):
                        p = []
                    pp = []
                    for x in p:
                        try:
                            pp.append(int(x))
                        except Exception:
                            continue
                    data["processed_deal_tickets"] = pp

                    # Ensure required aggregate keys exist
                    base = self._default_stats()
                    for k, v in base.items():
                        data.setdefault(k, v)

                    # Align symbol/lookback
                    data["symbol"] = self.symbol
                    data["lookback_days"] = self.lookback_days

                    # Save back if changed schema (safe)
                    self.stats = data
                    self._save_stats()
                    return data
            except Exception:
                pass

        return self._default_stats()

    def _save_stats(self) -> None:
        try:
            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _ensure_mt5(self) -> bool:
        if mt5.terminal_info() is not None:
            return True
        return bool(mt5.initialize())

    def sync_trade_closes(self) -> List[Dict[str, Any]]:
        """
        Returns list of new close-deals (normalized dicts).
        """
        if not self._ensure_mt5():
            return []

        from_date = datetime.now() - timedelta(days=self.lookback_days)
        deals = mt5.history_deals_get(from_date, datetime.now())
        if not deals:
            self.stats["last_sync_iso"] = datetime.now().isoformat()
            self._save_stats()
            return []

        processed = set(self.stats.get("processed_deal_tickets", []))
        new_closes: List[Dict[str, Any]] = []

        for d in deals:
            if d.entry != mt5.DEAL_ENTRY_OUT:
                continue
            if str(d.symbol) != self.symbol:
                continue
            if int(d.ticket) in processed:
                continue

            profit = float(d.profit)
            commission = float(getattr(d, "commission", 0.0))
            swap = float(getattr(d, "swap", 0.0))
            net = profit + commission + swap

            close_event = {
                "type": "trade_close",
                "symbol": self.symbol,
                "deal_ticket": int(d.ticket),
                "position_id": int(getattr(d, "position_id", 0)),
                "order_id": int(getattr(d, "order", 0)),
                "time": int(d.time),
                "time_iso": datetime.fromtimestamp(int(d.time)).strftime("%Y-%m-%d %H:%M:%S"),
                "side": "BUY" if d.type == mt5.DEAL_TYPE_BUY else "SELL",
                "volume": float(d.volume),
                "profit": profit,
                "commission": commission,
                "swap": swap,
                "net": net,
            }

            new_closes.append(close_event)
            processed.add(int(d.ticket))
            self.stats["processed_deal_tickets"].append(int(d.ticket))

            self.stats["total_trades"] += 1
            self.stats["net_profit"] = float(self.stats.get("net_profit", 0.0)) + net

            if net > 0:
                self.stats["winning_trades"] += 1
                self.stats["gross_profit"] = float(self.stats.get("gross_profit", 0.0)) + net
                self.stats["best_trade"] = max(float(self.stats.get("best_trade", 0.0)), net)
            else:
                self.stats["losing_trades"] += 1
                self.stats["gross_loss"] = float(self.stats.get("gross_loss", 0.0)) + abs(net)
                self.stats["worst_trade"] = min(float(self.stats.get("worst_trade", 0.0)), net)

            self.stats["trades"].append(close_event)

        self.stats["last_sync_iso"] = datetime.now().isoformat()
        self._save_stats()
        return new_closes

    def summary(self) -> PerfSummary:
        total = int(self.stats.get("total_trades", 0))
        wins = int(self.stats.get("winning_trades", 0))
        losses = int(self.stats.get("losing_trades", 0))
        win_rate = (wins / total * 100.0) if total > 0 else 0.0

        gross_profit = float(self.stats.get("gross_profit", 0.0))
        gross_loss = float(self.stats.get("gross_loss", 0.0))
        net_profit = float(self.stats.get("net_profit", 0.0))

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        return PerfSummary(
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            net_profit=net_profit,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            best_trade=float(self.stats.get("best_trade", 0.0)),
            worst_trade=float(self.stats.get("worst_trade", 0.0)),
        )
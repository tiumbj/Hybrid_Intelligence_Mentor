"""
ai_bridge_v1.py
Version: 1.0.0
Changelog:
- 1.0.0:
  - Add AIConfirmClient for concise payload -> AI -> strict response parsing
  - Return normalized decision: {final_confirm, side, entry, sl, tp, confidence, mentor_hint}
Rules:
- Fail closed: if AI response invalid => final_confirm=False
- Keep payload compact to reduce token cost
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger("HIM")


@dataclass(frozen=True)
class AIConfirmDecision:
    final_confirm: bool
    side: str  # "BUY" or "SELL"
    entry: float
    sl: float
    tp: float
    confidence: float  # 0..100
    mentor_hint: str = ""  # optional short text from AI


class AIConfirmClient:
    """
    Contract (expected):
    - POST ai.api_url with JSON payload
    - Response can be:
      A) JSON: {"final_confirm": true, "side":"BUY","entry":..,"sl":..,"tp":..,"confidence":..,"mentor_hint":"..."}
      B) Text line containing: "Entry xxx SL xxx TP xxx Confidence xx%" plus optional "FINAL_CONFIRM=true/false" and side
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = self._load_config(config_path)
        self.api_url = str((self.cfg.get("ai") or {}).get("api_url", "")).strip()
        self.timeout_sec = int((self.cfg.get("ai") or {}).get("timeout_sec", 15))

    def _load_config(self, path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception as e:
            logger.error(f"CRITICAL: AIConfirmClient config load failed: {e}")
            return {}

    def build_compact_payload(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep payload minimal to reduce token cost.
        Recommended keys only.
        """
        ctx = (signal.get("context") or {}) if isinstance(signal, dict) else {}
        payload = {
            "symbol": signal.get("symbol", "GOLD"),
            "side_hint": (ctx.get("proximity_side") or ctx.get("bias_side") or signal.get("direction") or "NONE"),
            "watch_state": ctx.get("watch_state", "NONE"),
            "breakout_state": ctx.get("breakout_state", "NONE"),
            "price_bid": ctx.get("price_bid"),
            "price_ask": ctx.get("price_ask"),
            "atr": ctx.get("atr"),
            "bos_high": ctx.get("bos_ref_high") or ctx.get("bos_high"),
            "bos_low": ctx.get("bos_ref_low") or ctx.get("bos_low"),
            "distance_buy": ctx.get("distance_buy"),
            "distance_sell": ctx.get("distance_sell"),
            "proximity_score": ctx.get("proximity_score"),
            "htf_trend": ctx.get("htf_trend"),
            "mtf_trend": ctx.get("mtf_trend"),
            "ltf_trend": ctx.get("ltf_trend"),
            "spread_points": ctx.get("spread_points"),
        }
        # Remove None to keep compact
        return {k: v for k, v in payload.items() if v is not None}

    def request_confirm(self, signal: Dict[str, Any]) -> AIConfirmDecision:
        if not self.api_url:
            logger.error("AI api_url missing in config.json under ai.api_url")
            return AIConfirmDecision(False, "NONE", 0.0, 0.0, 0.0, 0.0, "ai.api_url missing")

        payload = self.build_compact_payload(signal)

        try:
            resp = requests.post(self.api_url, json=payload, timeout=self.timeout_sec)
            text = (resp.text or "").strip()

            # Try JSON first
            try:
                data = resp.json()
                dec = self._parse_json_response(data)
                if dec:
                    return dec
            except Exception:
                pass

            # Fallback to text parse
            dec2 = self._parse_text_response(text)
            if dec2:
                return dec2

            logger.error(f"AI response unparseable: status={resp.status_code} body={text[:500]}")
            return AIConfirmDecision(False, "NONE", 0.0, 0.0, 0.0, 0.0, "ai response unparseable")

        except Exception as e:
            logger.error(f"AI request failed: {e}")
            return AIConfirmDecision(False, "NONE", 0.0, 0.0, 0.0, 0.0, f"ai exception: {e}")

    def _parse_json_response(self, data: Any) -> Optional[AIConfirmDecision]:
        if not isinstance(data, dict):
            return None
        fc = bool(data.get("final_confirm", False))
        side = str(data.get("side", "NONE")).upper()
        try:
            entry = float(data.get("entry"))
            sl = float(data.get("sl"))
            tp = float(data.get("tp"))
            conf = float(data.get("confidence"))
        except Exception:
            return None
        mentor_hint = str(data.get("mentor_hint", "") or "")
        if side not in ("BUY", "SELL"):
            return None
        conf = max(0.0, min(100.0, conf))
        return AIConfirmDecision(fc, side, entry, sl, tp, conf, mentor_hint)

    def _parse_text_response(self, text: str) -> Optional[AIConfirmDecision]:
        if not text:
            return None

        # SIDE
        side = "NONE"
        m_side = re.search(r"\b(BUY|SELL)\b", text.upper())
        if m_side:
            side = m_side.group(1)

        # FINAL_CONFIRM
        fc = False
        m_fc = re.search(r"FINAL[_ ]?CONFIRM\s*=\s*(TRUE|FALSE)", text.upper())
        if m_fc:
            fc = (m_fc.group(1) == "TRUE")
        else:
            # If not present, infer: confidence >= 50 and side exists
            fc = (side in ("BUY", "SELL"))

        # Numbers
        def _find_float(label: str) -> Optional[float]:
            m = re.search(label + r"\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
            return float(m.group(1)) if m else None

        entry = _find_float("Entry")
        sl = _find_float("SL")
        tp = _find_float("TP")
        conf = _find_float("Confidence")

        if entry is None or sl is None or tp is None or conf is None:
            return None
        if side not in ("BUY", "SELL"):
            return None

        conf = max(0.0, min(100.0, float(conf)))
        mentor_hint = ""
        return AIConfirmDecision(bool(fc), side, float(entry), float(sl), float(tp), conf, mentor_hint)


def _sample_signal() -> Dict[str, Any]:
    return {
        "symbol": "GOLD",
        "context": {
            "watch_state": "WATCH_BUY_BREAKOUT",
            "breakout_state": "BREAKOUT_BUY_CONFIRMED",
            "price_bid": 5190.10,
            "price_ask": 5190.20,
            "atr": 4.55,
            "bos_ref_high": 5189.67,
            "bos_ref_low": 5160.12,
            "distance_buy": -0.53,
            "distance_sell": 30.0,
            "proximity_score": 76.7,
            "htf_trend": "ranging",
            "mtf_trend": "ranging",
            "ltf_trend": "bullish",
            "spread_points": 5,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    client = AIConfirmClient(args.config)
    signal = _sample_signal() if args.sample else {}
    decision = client.request_confirm(signal)

    print("AIConfirmDecision:", decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
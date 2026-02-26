"""
AI Mentor - Mock Version (STRICT JSON ONLY)
Version: 2.0.0
Changelog:
- 2.0.0: Enforce Final Spec response format strictly:
  {
    "approved": true,
    "confidence": 87,
    "entry": 1234.5,
    "sl": 1220.0,
    "tp": 1270.0,
    "reasoning": "...",
    "warnings": "..."
  }
Notes:
- This is MOCK AI (Python logic) but returns the exact JSON fields.
- Confidence is 0..100 int.
"""

from __future__ import annotations

import json
from typing import Any, Dict


class AIMentor:
    def __init__(self):
        self.tokens_used = 0
        self.approved_count = 0
        self.rejected_count = 0
        print("✅ AI Mentor (Mock Mode) - STRICT JSON ONLY")

    def approve_trade(self, signal_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: compact signal package from engine (dict).
        Output: STRICT JSON fields only (dict ready to json.dumps).
        """
        # Defensive reads
        direction = str(signal_package.get("direction", "NONE"))
        ctx = signal_package.get("context", {}) or {}
        bos = bool(ctx.get("bos", False))
        st_ok = bool(ctx.get("supertrend_ok", False))
        score = float(signal_package.get("score", 0.0))
        rr = float(signal_package.get("rr", 0.0))

        entry = float(signal_package.get("entry_candidate", 0.0))
        sl = float(signal_package.get("stop_candidate", 0.0))
        tp = float(signal_package.get("tp_candidate", 0.0))

        reasons = []
        warns = []

        # Required rules (final spec)
        if direction not in ("BUY", "SELL"):
            self.rejected_count += 1
            return {
                "approved": False,
                "confidence": 0,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "reasoning": "Rejected: direction is NONE/invalid.",
                "warnings": "No trade.",
            }

        if not bos:
            warns.append("❌ BOS not confirmed (BOS is REQUIRED).")
        else:
            reasons.append("✅ BOS confirmed.")

        if not st_ok:
            warns.append("❌ SuperTrend contradicts signal direction.")
        else:
            reasons.append("✅ SuperTrend confirms direction.")

        # Score / RR evaluation
        if score >= 7.0:
            reasons.append(f"✅ Score OK ({score:.1f}).")
        else:
            warns.append(f"⚠️ Score low ({score:.1f}).")

        if rr >= 2.0:
            reasons.append(f"✅ RR OK ({rr:.2f}).")
        else:
            warns.append(f"⚠️ RR low ({rr:.2f}).")

        # Confidence model (simple deterministic mapping)
        # Base from score (0..10 => 0..70), bonus for BOS/ST/RR
        conf = int(max(0, min(100, round(score * 7))))
        if bos:
            conf += 10
        if st_ok:
            conf += 10
        if rr >= 2.0:
            conf += 5
        conf = int(max(0, min(100, conf)))

        approved = (bos and st_ok and score >= 7.0 and rr >= 2.0 and conf >= 75)

        if approved:
            self.approved_count += 1
        else:
            self.rejected_count += 1

        reasoning = "\n".join(reasons) if reasons else "No strong confluence."
        warnings = "\n".join(warns) if warns else ""

        return {
            "approved": bool(approved),
            "confidence": int(conf),
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "reasoning": reasoning,
            "warnings": warnings,
        }

    @staticmethod
    def validate_response(ai_resp: Dict[str, Any]) -> bool:
        """
        Strict validation for executor gate.
        """
        required = ["approved", "confidence", "entry", "sl", "tp", "reasoning", "warnings"]
        if not all(k in ai_resp for k in required):
            return False
        if not isinstance(ai_resp["approved"], bool):
            return False
        if not isinstance(ai_resp["confidence"], int):
            return False
        if ai_resp["confidence"] < 0 or ai_resp["confidence"] > 100:
            return False
        # floats
        for k in ["entry", "sl", "tp"]:
            if not isinstance(ai_resp[k], (int, float)):
                return False
        if not isinstance(ai_resp["reasoning"], str) or not isinstance(ai_resp["warnings"], str):
            return False
        return True
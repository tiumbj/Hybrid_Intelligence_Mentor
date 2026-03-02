"""
HIM
File: tests/test_event_gating.py
Version: v2.12.4
Purpose: Basic unit tests for event gating state transitions (no MT5 dependency).
"""

import unittest
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple


@dataclass
class EventState:
    last_bar_time: Dict[str, int]
    last_signature: Optional[str]


def default_event_state() -> EventState:
    return EventState(last_bar_time={}, last_signature=None)


def simulate_should_fire_event(
    tfs: List[str],
    state: EventState,
    bar_times: Dict[str, int],
    signature: Optional[str],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    # NEW_BAR
    new_bar = False
    for tf in tfs:
        bt = bar_times.get(tf)
        if bt is None:
            continue
        prev = state.last_bar_time.get(tf)
        if prev is None:
            state.last_bar_time[tf] = bt
        elif bt != prev:
            new_bar = True
            state.last_bar_time[tf] = bt
    if new_bar:
        reasons.append("NEW_BAR")

    # SIGNAL_CHANGE
    if signature is not None:
        if state.last_signature is None:
            state.last_signature = signature
        elif signature != state.last_signature:
            reasons.append("SIGNAL_CHANGE")
            state.last_signature = signature

    return (len(reasons) > 0), reasons


class TestEventGating(unittest.TestCase):
    def test_initialization_no_event(self):
        st = default_event_state()
        fire, reasons = simulate_should_fire_event(
            ["M5"], st, {"M5": 1000}, "aaa"
        )
        self.assertFalse(fire)
        self.assertEqual(reasons, [])

    def test_new_bar_triggers(self):
        st = default_event_state()
        simulate_should_fire_event(["M5"], st, {"M5": 1000}, "aaa")  # init
        fire, reasons = simulate_should_fire_event(["M5"], st, {"M5": 2000}, "aaa")
        self.assertTrue(fire)
        self.assertIn("NEW_BAR", reasons)
        self.assertNotIn("SIGNAL_CHANGE", reasons)

    def test_signal_change_triggers(self):
        st = default_event_state()
        simulate_should_fire_event(["M5"], st, {"M5": 1000}, "aaa")  # init
        fire, reasons = simulate_should_fire_event(["M5"], st, {"M5": 1000}, "bbb")
        self.assertTrue(fire)
        self.assertIn("SIGNAL_CHANGE", reasons)

    def test_both_triggers(self):
        st = default_event_state()
        simulate_should_fire_event(["M5"], st, {"M5": 1000}, "aaa")  # init
        fire, reasons = simulate_should_fire_event(["M5"], st, {"M5": 2000}, "bbb")
        self.assertTrue(fire)
        self.assertIn("NEW_BAR", reasons)
        self.assertIn("SIGNAL_CHANGE", reasons)


if __name__ == "__main__":
    unittest.main()
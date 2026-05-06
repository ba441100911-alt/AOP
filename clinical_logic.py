"""
Core Clinical Logic Module
===========================
Implements:
    Step 3 — AOP & PB detection (sliding window)
    Step 4 — Risk classification (Low / Moderate / High)
    Step 5 — Rule-based recommendation engine
    Step 6 — Event tracking (1h / 24h counts, last event)
"""

from collections import deque
from datetime import datetime, timedelta


# =================================================================
# STEP 3 — DETECTION LOGIC
# =================================================================
#
# Uses a sliding window of the most recent vitals (1 sample per second).
# The window must be at least 120s long to detect Periodic Breathing.
#
#  AOP rule:
#      RR < 5 for >= 20 consecutive seconds
#      AND (SpO2 < 90 OR HR < 100) at some point in that window
#
#  PB rule:
#      Short pauses (RR < 5 for 5-10 seconds), >= 3 times in 120s
#      AND no SpO2 < 90, AND no HR < 100 (otherwise it would be AOP)
# =================================================================

# Detection thresholds — mirror the clinical definitions exactly
RR_PAUSE_THRESHOLD = 5
SPO2_LOW_THRESHOLD = 90
HR_LOW_THRESHOLD = 100

AOP_MIN_DURATION = 20      # seconds
PB_PAUSE_MIN = 5           # seconds
PB_PAUSE_MAX = 10          # seconds
PB_WINDOW = 120            # seconds
PB_MIN_PAUSES = 3


def _find_low_rr_runs(rr_window):
    """Return a list of (start_idx, length) for every consecutive run of RR < 5."""
    runs = []
    start = None
    for i, rr in enumerate(rr_window):
        if rr < RR_PAUSE_THRESHOLD:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - start))
                start = None
    if start is not None:
        runs.append((start, len(rr_window) - start))
    return runs


def detect_aop(hr_window, spo2_window, rr_window):
    """
    Return True if AOP criteria are met inside the supplied window.

    AOP = RR < 5 for >= 20 consecutive seconds
          AND (SpO2 < 90 OR HR < 100) somewhere during that pause.
    """
    runs = _find_low_rr_runs(rr_window)
    for start, length in runs:
        if length >= AOP_MIN_DURATION:
            end = start + length
            spo2_low = any(s < SPO2_LOW_THRESHOLD for s in spo2_window[start:end])
            hr_low = any(h < HR_LOW_THRESHOLD for h in hr_window[start:end])
            if spo2_low or hr_low:
                return True
    return False


def detect_pb(hr_window, spo2_window, rr_window):
    """
    Return True if Periodic Breathing criteria are met within last 120s.

    PB = >= 3 short pauses (RR < 5 for 5-10 seconds) within 120s,
         WITHOUT any SpO2 < 90 and WITHOUT any HR < 100.
    """
    if len(rr_window) < PB_WINDOW:
        return False

    # Use only the last 120 seconds of data
    hr_w = hr_window[-PB_WINDOW:]
    spo2_w = spo2_window[-PB_WINDOW:]
    rr_w = rr_window[-PB_WINDOW:]

    # Disqualify if any desat or bradycardia present (those make it AOP, not PB)
    if any(s < SPO2_LOW_THRESHOLD for s in spo2_w):
        return False
    if any(h < HR_LOW_THRESHOLD for h in hr_w):
        return False

    runs = _find_low_rr_runs(rr_w)
    short_pauses = [r for r in runs if PB_PAUSE_MIN <= r[1] <= PB_PAUSE_MAX]
    return len(short_pauses) >= PB_MIN_PAUSES


def detect_event(hr_window, spo2_window, rr_window):
    """Return 'AOP', 'PB', or 'None' — AOP takes priority."""
    if detect_aop(hr_window, spo2_window, rr_window):
        return "AOP"
    if detect_pb(hr_window, spo2_window, rr_window):
        return "PB"
    return "None"


# =================================================================
# STEP 4 — RISK CLASSIFICATION
# =================================================================
def classify_risk(probability):
    """
    Convert model probability into a 3-level clinical risk category.
        Low      : < 30%
        Moderate : 30% – 70%
        High     : > 70%
    """
    if probability < 0.30:
        return "Low"
    if probability <= 0.70:
        return "Moderate"
    return "High"


def risk_color(risk_level):
    """Color codes matching standard clinical alarm conventions."""
    return {"Low": "#28a745",       # green
            "Moderate": "#ffc107",  # amber
            "High": "#dc3545"}.get(risk_level, "#6c757d")


def overall_risk(probabilities_by_horizon):
    """
    Combine multiple horizons into a single 'most concerning' risk level.

    Rule of thumb:
        - A High risk on the SHORTEST horizon (30s) is most clinically urgent.
        - We pick the worst risk level across horizons, but prefer shorter
          horizons when the risk level is tied.
    """
    severity = {"Low": 0, "Moderate": 1, "High": 2}
    worst_level = "Low"
    worst_horizon = None
    for h in sorted(probabilities_by_horizon.keys()):  # short -> long
        level = classify_risk(probabilities_by_horizon[h])
        if severity[level] > severity[worst_level]:
            worst_level = level
            worst_horizon = h
        elif severity[level] == severity[worst_level] and worst_horizon is None:
            worst_horizon = h
    return worst_level, worst_horizon


# =================================================================
# STEP 5 — RECOMMENDATION ENGINE
# =================================================================
def get_recommendation(risk_level, current_event):
    """Return clinical action recommendation based on risk + current event."""

    # Active AOP overrides everything — emergency response
    if current_event == "AOP":
        return {
            "title": "🚨 ACTIVE AOP — Immediate intervention required",
            "actions": [
                "Provide tactile stimulation immediately",
                "Check airway and reposition infant",
                "Ensure supplemental oxygen is available",
                "Notify attending physician/nurse",
                "Prepare for bag-mask ventilation if no recovery in 30 seconds",
            ],
            "urgency": "EMERGENCY",
        }

    if risk_level == "High":
        return {
            "title": "⚠️ HIGH RISK — Apnea likely within 60 seconds",
            "actions": [
                "Begin gentle tactile stimulation",
                "Verify airway patency and infant position",
                "Confirm SpO2 probe and ECG leads are properly placed",
                "Alert bedside nurse and prepare oxygen",
                "Re-check vitals every 15 seconds",
            ],
            "urgency": "URGENT",
        }

    if risk_level == "Moderate":
        return {
            "title": "⚡ MODERATE RISK — Increased monitoring",
            "actions": [
                "Increase monitoring frequency to every 30 seconds",
                "Verify infant position and feeding status",
                "Check ambient temperature (cold stress can trigger apnea)",
                "Document trend and notify nurse if it worsens",
            ],
            "urgency": "CAUTION",
        }

    # Low risk
    if current_event == "PB":
        return {
            "title": "ℹ️ Periodic Breathing detected — Low immediate risk",
            "actions": [
                "Continue routine monitoring",
                "PB is common and typically self-resolving in preterm infants",
                "Document event in patient record",
                "Monitor for progression to AOP",
            ],
            "urgency": "INFORMATIONAL",
        }

    return {
        "title": "✅ LOW RISK — Routine monitoring",
        "actions": [
            "Continue standard NICU monitoring",
            "Maintain vitals trending every 5 minutes",
            "Reassess in 15 minutes",
        ],
        "urgency": "ROUTINE",
    }


# =================================================================
# STEP 6 — EVENT TRACKING
# =================================================================
class EventTracker:
    """
    Tracks AOP events over time, providing:
        • count of events in the last 60 minutes
        • count of events in the last 24 hours
        • timestamp of the most recent event
    """

    def __init__(self):
        self.events = deque()       # list of (timestamp, event_type)
        self.last_event = None      # (timestamp, event_type) or None
        self.last_event_type = None # remember last event-type to avoid double-counting

    def log_event(self, event_type, timestamp=None):
        """
        Log a detection. We only record an AOP/PB transition once —
        as long as the event keeps being detected, we don't add new entries.
        """
        if timestamp is None:
            timestamp = datetime.now()

        if event_type == "None":
            self.last_event_type = None  # reset so next AOP/PB is counted
            return

        # Only log when the event TYPE changes (start of a new episode)
        if event_type != self.last_event_type:
            self.events.append((timestamp, event_type))
            self.last_event = (timestamp, event_type)
            self.last_event_type = event_type

    def _count_since(self, since_time, event_type="AOP"):
        # Drop expired records (older than 24h) to keep memory small
        cutoff_24h = datetime.now() - timedelta(hours=24)
        while self.events and self.events[0][0] < cutoff_24h:
            self.events.popleft()
        return sum(1 for ts, et in self.events
                   if ts >= since_time and et == event_type)

    def aop_count_last_hour(self):
        return self._count_since(datetime.now() - timedelta(hours=1), "AOP")

    def aop_count_last_24h(self):
        return self._count_since(datetime.now() - timedelta(hours=24), "AOP")

    def get_last_event(self):
        if self.last_event is None:
            return None, None
        return self.last_event[0], self.last_event[1]

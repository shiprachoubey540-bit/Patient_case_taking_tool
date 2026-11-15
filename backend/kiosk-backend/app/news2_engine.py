"""
news2_engine.py — Deterministic National Early Warning Score (NEWS2) Calculator.
Conforms strictly to the Royal College of Physicians (RCP London) 2017 standard.
Used for acute clinical risk stratification in outpatient and emergency triage.
"""
from __future__ import annotations
from typing import Dict, Any, Optional


def score_respiration_rate(rate: Optional[int]) -> int:
    """Scores respiration rate in breaths per minute."""
    if rate is None:
        return 0
    if rate <= 8:
        return 3
    if 9 <= rate <= 11:
        return 1
    if 12 <= rate <= 20:
        return 0
    if 21 <= rate <= 24:
        return 2
    return 3  # >= 25


def score_spo2(spo2: Optional[int], on_oxygen: bool = False) -> int:
    """Scores oxygen saturation (Scale 1 - standard)."""
    if spo2 is None:
        return 0
    score = 0
    if spo2 <= 91:
        score = 3
    elif 92 <= spo2 <= 93:
        score = 2
    elif 94 <= spo2 <= 95:
        score = 1
    else:
        score = 0  # >= 96

    if on_oxygen:
        score += 2  # Supplemental O2 adds 2 points on NEWS2
    return score


def score_systolic_bp(sbp: Optional[int]) -> int:
    """Scores systolic blood pressure in mmHg."""
    if sbp is None:
        return 0
    if sbp <= 90:
        return 3
    if 91 <= sbp <= 100:
        return 2
    if 101 <= sbp <= 110:
        return 1
    if 111 <= sbp <= 219:
        return 0
    return 3  # >= 220


def score_pulse(pulse: Optional[int]) -> int:
    """Scores heart rate / pulse in beats per minute."""
    if pulse is None:
        return 0
    if pulse <= 40:
        return 3
    if 41 <= pulse <= 50:
        return 1
    if 51 <= pulse <= 90:
        return 0
    if 91 <= pulse <= 110:
        return 1
    if 111 <= pulse <= 130:
        return 2
    return 3  # >= 131


def score_consciousness(consciousness: Optional[str]) -> int:
    """Scores AVPU neurological state: Alert (A), Voice (V), Pain (P), Unresponsive (U)."""
    if not consciousness or str(consciousness).lower() in ["alert", "a"]:
        return 0
    return 3


def score_temperature(temp_c: Optional[float]) -> int:
    """Scores body temperature in degrees Celsius."""
    if temp_c is None:
        return 0
    if temp_c <= 35.0:
        return 3
    if 35.1 <= temp_c <= 36.0:
        return 1
    if 36.1 <= temp_c <= 38.0:
        return 0
    if 38.1 <= temp_c <= 39.0:
        return 1
    return 2  # >= 39.1


def calculate_news2(vitals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes complete NEWS2 composite score from vital signs dictionary.
    """
    temp_c = vitals.get("temperature_c")
    if temp_c is None and vitals.get("temperature_f") is not None:
        try:
            temp_c = round((float(vitals["temperature_f"]) - 32) / 1.8, 1)
        except (ValueError, TypeError):
            temp_c = None

    on_oxygen = bool(vitals.get("on_oxygen", 0))

    breakdown = {
        "respiration_score": score_respiration_rate(vitals.get("respiratory_rate")),
        "spo2_score": score_spo2(vitals.get("spo2"), on_oxygen),
        "systolic_bp_score": score_systolic_bp(vitals.get("systolic_bp")),
        "pulse_score": score_pulse(vitals.get("pulse")),
        "consciousness_score": score_consciousness(vitals.get("consciousness")),
        "temperature_score": score_temperature(temp_c),
    }

    total_score = sum(breakdown.values())
    has_extreme_vital = any(v == 3 for v in breakdown.values())

    if total_score >= 7 or has_extreme_vital:
        risk_level = "High"
        badge_color = "red"
        clinical_action = "CRITICAL ALERT: Immediate emergency physician assessment required."
    elif total_score in [5, 6]:
        risk_level = "Medium"
        badge_color = "amber"
        clinical_action = "URGENT: Prioritize OPD review. Nurse observation within 30 minutes."
    else:
        risk_level = "Low"
        badge_color = "green"
        clinical_action = "ROUTINE: Continue standard OPD wait queue."

    return {
        "total_score": total_score,
        "risk_level": risk_level,
        "badge_color": badge_color,
        "clinical_action": clinical_action,
        "has_extreme_vital": has_extreme_vital,
        "breakdown": breakdown
    }

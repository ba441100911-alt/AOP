"""
Step 1: Synthetic NICU Vital Signs Data Generator (Multi-Horizon)
==================================================================
Generates realistic time-series of HR, SpO2, RR for premature infants
with THREE prediction labels:
    apnea_in_30s   — apnea occurs within next 30 seconds
    apnea_in_60s   — apnea occurs within next 60 seconds
    apnea_in_120s  — apnea occurs within next 120 seconds

Output: nicu_data.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Clinical baselines for a stable preterm infant
BASELINE_HR = 150
BASELINE_SPO2 = 97
BASELINE_RR = 45

HR_NOISE = 5
SPO2_NOISE = 1
RR_NOISE = 4

# Prediction horizons (in seconds) — defined once, used everywhere
HORIZONS = [30, 60, 120]


def generate_patient_timeseries(patient_id, duration_seconds=3600, n_apnea_events=4):
    """
    Generate one patient's vital signs at 1-second resolution.

    Each apnea episode follows this physiological pattern:
        1. Normal baseline
        2. ~30-90s deterioration (HR↓, SpO2↓, RR↓)
              ↑ widened so the 120s label has meaningful early signal
        3. Apnea event (RR<5, often with bradycardia + desaturation)
        4. Recovery phase (~30s)
    """
    t = np.arange(duration_seconds)

    HR = np.random.normal(BASELINE_HR, HR_NOISE, duration_seconds)
    SpO2 = np.random.normal(BASELINE_SPO2, SPO2_NOISE, duration_seconds)
    RR = np.random.normal(BASELINE_RR, RR_NOISE, duration_seconds)
    apnea_now = np.zeros(duration_seconds, dtype=int)

    # Place apnea event centers — leave more buffer (150s) so 120s window fits cleanly
    apnea_event_centers = sorted(np.random.choice(
        np.arange(150, duration_seconds - 150),
        size=n_apnea_events,
        replace=False
    ))

    for center in apnea_event_centers:
        # ---- 1) Deterioration window: 30-90s before event ----
        # Wider than before so the 120s-horizon model has signal to learn from
        deterioration_len = np.random.randint(30, 91)
        det_start = max(center - deterioration_len, 0)
        det_end = center

        for i in range(det_start, det_end):
            progress = (i - det_start) / deterioration_len  # 0 -> 1
            HR[i] -= 30 * progress
            SpO2[i] -= 8 * progress
            RR[i] -= 25 * progress

        # ---- 2) Apnea: RR collapses for 20-30s ----
        apnea_duration = np.random.randint(20, 31)
        ap_start = center
        ap_end = min(center + apnea_duration, duration_seconds)

        RR[ap_start:ap_end] = np.random.uniform(0, 4, ap_end - ap_start)
        SpO2[ap_start:ap_end] = np.random.uniform(80, 89, ap_end - ap_start)
        HR[ap_start:ap_end] = np.random.uniform(80, 99, ap_end - ap_start)
        apnea_now[ap_start:ap_end] = 1

        # ---- 3) Recovery: 30s back to baseline ----
        rec_start = ap_end
        rec_end = min(ap_end + 30, duration_seconds)
        for i in range(rec_start, rec_end):
            progress = (i - rec_start) / 30
            HR[i] = HR[i] * progress + 100 * (1 - progress)
            SpO2[i] = SpO2[i] * progress + 88 * (1 - progress)
            RR[i] = RR[i] * progress + 5 * (1 - progress)

    # Clip to physiologically possible ranges
    HR = np.clip(HR, 40, 220)
    SpO2 = np.clip(SpO2, 60, 100)
    RR = np.clip(RR, 0, 90)

    df = pd.DataFrame({
        'timestamp': t,
        'patient_id': patient_id,
        'HR': np.round(HR, 1),
        'SpO2': np.round(SpO2, 1),
        'RR': np.round(RR, 1),
        'apnea_now': apnea_now,
    })

    # ---- Build labels for ALL horizons in one pass ----
    # apnea_in_Xs[i] = 1 if any apnea second exists in (i, i+X]
    for h in HORIZONS:
        labels = np.zeros(duration_seconds, dtype=int)
        for i in range(duration_seconds):
            future = apnea_now[i+1 : min(i+1+h, duration_seconds)]
            if future.sum() > 0:
                labels[i] = 1
        df[f'apnea_in_{h}s'] = labels

    return df


def generate_full_dataset(n_patients=20, duration_seconds=3600):
    all_patients = []
    for pid in range(n_patients):
        n_events = np.random.randint(2, 6)
        df = generate_patient_timeseries(pid, duration_seconds, n_events)
        all_patients.append(df)
    return pd.concat(all_patients, ignore_index=True)


if __name__ == "__main__":
    print("Generating synthetic NICU dataset (multi-horizon)...")
    dataset = generate_full_dataset(n_patients=20, duration_seconds=3600)
    dataset.to_csv("nicu_data.csv", index=False)

    print(f"\n✅ Dataset saved: nicu_data.csv")
    print(f"   Total rows : {len(dataset):,}")
    print(f"   Patients   : {dataset['patient_id'].nunique()}")
    print(f"   Apnea seconds : {dataset['apnea_now'].sum():,}")
    print("\n   Class balance per horizon:")
    for h in HORIZONS:
        col = f'apnea_in_{h}s'
        print(f"     {col:<16} positive: {dataset[col].mean():.2%} ({dataset[col].sum():,} rows)")

    print("\nFirst 5 rows:")
    print(dataset.head())

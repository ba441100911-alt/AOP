# AOP Early Warning System — MVP (Multi-Horizon)

AI-powered early-warning and decision-support system that predicts **Apnea of Prematurity (AOP)** in NICU patients **at three look-ahead horizons (30 s, 60 s, 120 s)**, using only HR, SpO₂, and RR.

## 📁 Project Structure
```
aop_mvp/
├── generate_data.py     Step 1 — synthetic NICU dataset (3 horizon labels)
├── train_model.py       Step 2 — trains one Random Forest per horizon
├── clinical_logic.py    Steps 3-6 — detection, risk, recommendations, tracking
├── app.py               Step 7 — Streamlit dashboard (multi-horizon)
├── requirements.txt
├── nicu_data.csv        (generated)
└── aop_models.pkl       (generated — bundle of 3 models keyed by horizon)
```

## 🚀 How to Run

```bash
pip install -r requirements.txt
python generate_data.py    # creates nicu_data.csv
python train_model.py      # creates aop_models.pkl (3 models)
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`.

## ⏱️ Multi-Horizon Prediction

| Horizon | Use case | Typical recall (synthetic) |
|---------|----------|----------------------------|
| **30 s**  | Imminent — react NOW       | ~96 % |
| **60 s**  | Short-term warning (primary clinical action) | ~86 % |
| **120 s** | Early heads-up — start watching | ~55 % |

The closer to the event, the stronger the deterioration signal. The 60 s horizon drives the recommendation panel because it strikes the best balance between reliability and time to react.

## 🩺 Clinical Definitions

| Event | Criteria |
|-------|----------|
| **AOP** | RR < 5 for ≥ 20 seconds **AND** (SpO₂ < 90 **OR** HR < 100) |
| **PB**  | RR < 5 for 5–10 seconds, ≥ 3 times within 120 s, **NO** SpO₂ < 90 and **NO** HR < 100 |

## 🎯 Risk Levels

| Level | Probability | Action |
|-------|-------------|--------|
| 🟢 Low      | < 30 %    | Routine monitoring |
| 🟡 Moderate | 30 – 70 % | Increase monitoring frequency |
| 🔴 High     | > 70 %    | Stimulation + airway check |

## 📊 Trend Display (120 seconds)

The dashboard shows three side-by-side charts for HR, SpO₂, and RR over the last 120 seconds. Each has its own clinical-threshold reference line:
- HR — bradycardia threshold at 100 bpm
- SpO₂ — desaturation threshold at 90 %
- RR — apnea threshold at 5 bpm

A combined probability-trend chart shows all three horizons (30 s, 60 s, 120 s) together so you can watch them rise as apnea approaches.

## 🧪 Try It

Sidebar **Quick Scenarios**:
- ✅ Normal vitals
- 📉 Gradual deterioration *(new — best for showing 120 s horizon firing first)*
- 🚨 AOP episode
- 💨 Periodic breathing

## ⚠️ Disclaimer
Research/educational MVP. **Not validated for clinical use.**

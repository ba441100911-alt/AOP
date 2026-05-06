"""
Step 2: Train Three Random Forest Models (Multi-Horizon)
=========================================================
Trains a separate model for each prediction horizon (30s, 60s, 120s).

Why one model per horizon (not multi-output)?
  • Each horizon has a different positive-class rate, so class_weight balancing
    is tuned per-horizon.
  • Per-horizon evaluation (recall, AUC) is what clinicians actually care about
    — they want to know "how good is the 30-second alarm vs the 120-second one".
  • Inference cost is still negligible (3 small forests, milliseconds).

Saves: aop_models.pkl  (a dict { 30: model, 60: model, 120: model })
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split

HORIZONS = [30, 60, 120]
FEATURES = ['HR', 'SpO2', 'RR']

print("Loading dataset...")
df = pd.read_csv("nicu_data.csv")
X = df[FEATURES]

models = {}

for h in HORIZONS:
    target = f'apnea_in_{h}s'
    y = df[target]

    print("\n" + "=" * 60)
    print(f"TRAINING MODEL: horizon = {h} seconds   (target = {target})")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")
    print(f"  Positive class: {y_train.mean():.2%}")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        class_weight='balanced',  # important — each horizon is imbalanced
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                  Pred No   Pred Yes")
    print(f"      Actual No   {cm[0,0]:>7,}    {cm[0,1]:>7,}")
    print(f"      Actual Yes  {cm[1,0]:>7,}    {cm[1,1]:>7,}")

    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['No Apnea', f'Apnea in {h}s'],
        digits=3,
    ))

    models[h] = model

# Save all three models in one bundle
joblib.dump(models, "aop_models.pkl")
print("\n✅ All models saved as aop_models.pkl")
print(f"   Bundle keys: {list(models.keys())}")

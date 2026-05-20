import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score

print("Starting Production Traffic Simulation...")

# ── 1. Load Artifacts ──────────────────────────────────────────────────────────
print("Loading model and reference data...")

model        = joblib.load("model/model.pkl")
encoders     = joblib.load("model/encoders.pkl")
feature_cols = joblib.load("model/feature_cols.pkl")
reference_df = pd.read_csv("data/reference.csv")

print(f"Reference data loaded: {reference_df.shape[0]} rows")

# ── 2. Setup SQLite Database ───────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
conn   = sqlite3.connect("logs/predictions.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS predictions")
cursor.execute("DROP TABLE IF EXISTS weekly_metrics")
conn.commit()
print("Cleared old predictions and metrics tables.")

# ── 3. Safe Encoder ────────────────────────────────────────────────────────────
def safe_encode(le, values):
    known    = set(le.classes_)
    fallback = le.classes_[0]
    cleaned  = [v if v in known else fallback for v in values]
    return le.transform(cleaned)

# ── 4. Simulate 4 Weeks of Traffic ────────────────────────────────────────────
for week in range(1, 5):
    print(f"\n{'─'*45}")
    print(f"  Simulating Week {week}...")

    batch = reference_df.sample(n=500, replace=True, random_state=week).copy()

    # ── Inject Drift ──────────────────────────────────────────────────────────
    if week == 1:
        print("  No drift — clean production data.")

    elif week == 2:
        # Slight drift — shift 2 numeric features gently
        print("  Injecting SLIGHT drift...")
        batch["age"]            = batch["age"] + np.random.normal(loc=6, scale=2, size=len(batch))
        batch["hours-per-week"] = batch["hours-per-week"] + np.random.normal(loc=5, scale=2, size=len(batch))

    elif week == 3:
        # Moderate drift — shift 4 numeric features noticeably
        print("  Injecting MODERATE drift...")
        batch["age"]            = batch["age"] + np.random.normal(loc=12, scale=3, size=len(batch))
        batch["hours-per-week"] = batch["hours-per-week"] + np.random.normal(loc=12, scale=4, size=len(batch))
        batch["capital-gain"]   = batch["capital-gain"] * 4.0
        batch["education-num"]  = batch["education-num"] + np.random.normal(loc=2, scale=1, size=len(batch))

    elif week == 4:
        # Severe drift — shift 6+ features hard to guarantee is_drifted = True
        print("  Injecting SEVERE drift...")
        batch["age"]            = batch["age"] + np.random.normal(loc=22, scale=5, size=len(batch))
        batch["hours-per-week"] = batch["hours-per-week"] + np.random.normal(loc=22, scale=5, size=len(batch))
        batch["capital-gain"]   = batch["capital-gain"] * 9.0
        batch["capital-loss"]   = batch["capital-loss"] * 7.0
        batch["education-num"]  = batch["education-num"] + np.random.normal(loc=4, scale=1, size=len(batch))
        batch["fnlwgt"]         = batch["fnlwgt"] * 2.5

    # Clip impossible values
    batch["age"]            = batch["age"].clip(lower=17, upper=90).astype(int)
    batch["hours-per-week"] = batch["hours-per-week"].clip(lower=1, upper=99).astype(int)
    batch["education-num"]  = batch["education-num"].clip(lower=1, upper=16).astype(int)

    # ── Prepare Features for Model ────────────────────────────────────────────
    X_batch = batch.drop("target", axis=1, errors="ignore").copy()

    for col, le in encoders.items():
        if col in X_batch.columns:
            X_batch[col] = safe_encode(le, X_batch[col])

    X_batch = X_batch[feature_cols]

    # ── Generate Predictions ──────────────────────────────────────────────────
    preds = model.predict(X_batch)
    probs = model.predict_proba(X_batch).max(axis=1)

    # ── Log Performance Metrics ───────────────────────────────────────────────
    if "target" in batch.columns:
        acc = accuracy_score(batch["target"], preds)
        f1  = f1_score(batch["target"], preds)
        print(f"  Accuracy : {acc:.4f}  |  F1-Score : {f1:.4f}")

        metrics_df = pd.DataFrame([{
            "week":      week,
            "accuracy":  acc,
            "f1_score":  f1,
            "timestamp": datetime.now() + timedelta(days=week * 7)
        }])
        metrics_df.to_sql("weekly_metrics", conn, if_exists="append", index=False)
    else:
        print("  (No ground truth available — skipping accuracy calculation)")

    # ── Log Raw Predictions to SQLite ─────────────────────────────────────────
    log_df = batch.copy()
    log_df["prediction"] = preds
    log_df["confidence"] = probs
    log_df["timestamp"]  = datetime.now() + timedelta(days=week * 7)
    log_df["week"]       = week

    log_df.to_sql("predictions", conn, if_exists="append", index=False)
    print(f"  Logged {len(log_df)} predictions to SQLite.")

conn.close()

print(f"\n{'='*45}")
print("✅ Phase 3 Complete! Artifacts saved:")
print("   logs/predictions.db  ← 4 weeks of raw predictions")
print("   Table: predictions   ← input features + prediction + confidence + week")
print("   Table: weekly_metrics← accuracy + f1 per week (for dashboard)")
print("\nNext → Run Phase 4: python monitoring/detect_drift.py")
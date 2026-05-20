import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import joblib
import os

# ── 1. Load Data ───────────────────────────────────────────────────────────────
print("Loading data...")

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "target"
]

df = pd.read_csv("data/adult.csv", names=columns, na_values=" ?")
df = df.dropna()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Encode Target ───────────────────────────────────────────────────────────
print("Encoding target column...")

df["target"] = df["target"].apply(lambda x: 1 if x == ">50K" else 0)
print(f"Class distribution:\n{df['target'].value_counts()}")

# ── 3. Split BEFORE Encoding → Save Raw Reference for Evidently ───────────────
# FIX: Reference data must be saved with ORIGINAL categorical string values.
# Evidently AI needs strings like "Private", "Male" to detect categorical drift.
# If we save encoded integers, Evidently treats categoricals as numerics → wrong tests.
print("\nSplitting data (saving raw reference before encoding)...")

df_train_val_raw, df_reference_raw = train_test_split(
    df, test_size=0.15, random_state=42
)

os.makedirs("data", exist_ok=True)
df_reference_raw.to_csv("data/reference.csv", index=False)
print(f"Saved reference data (original values) → data/reference.csv  [{len(df_reference_raw)} rows]")

# ── 4. Encode Categorical Columns for Model Training ──────────────────────────
# FIX: Encode AFTER saving reference, and persist encoders so the
# simulator uses identical encoding — otherwise model gets garbage input.
print("\nEncoding categorical features...")

categorical_cols = df_train_val_raw.select_dtypes(include=["object"]).columns.tolist()
print(f"Categorical columns to encode: {categorical_cols}")

# Fit encoders on full train+val set (not just train) to avoid unseen-label errors
encoders = {}
df_train_val = df_train_val_raw.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_train_val[col] = le.fit_transform(df_train_val[col])
    encoders[col] = le

os.makedirs("model", exist_ok=True)
joblib.dump(encoders, "model/encoders.pkl")
print("Saved encoders → model/encoders.pkl")

# ── 5. Final Train / Val Split ─────────────────────────────────────────────────
# 70 / 15 / 15 total split:
#   df_train_val is 85% of data → split into 70% train + 15% val
#   0.15 / 0.85 = 0.1765
df_train, df_val = train_test_split(
    df_train_val, test_size=0.1765, random_state=42
)

X_train = df_train.drop("target", axis=1)
y_train = df_train["target"]
X_val   = df_val.drop("target", axis=1)
y_val   = df_val["target"]

print(f"\nSplit sizes → Train: {len(X_train)} | Val: {len(X_val)} | Reference: {len(df_reference_raw)}")

# Save feature column order so the simulator always sends columns in the same order
feature_cols = X_train.columns.tolist()
joblib.dump(feature_cols, "model/feature_cols.pkl")
print(f"Saved feature column order → model/feature_cols.pkl")

# ── 6. Train Model + Log with MLflow ──────────────────────────────────────────
print("\nTraining model and logging to MLflow...")

mlflow.set_experiment("ML_Monitoring_Baseline")

with mlflow.start_run() as run:

    # Hyperparameters
    n_estimators = 100
    max_depth     = 10

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth",     max_depth)
    mlflow.log_param("train_size",    len(X_train))
    mlflow.log_param("val_size",      len(X_val))

    # Train
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    val_preds    = model.predict(X_val)
    val_proba    = model.predict_proba(X_val)[:, 1]
    acc          = accuracy_score(y_val, val_preds)
    f1           = f1_score(y_val, val_preds)
    auc          = roc_auc_score(y_val, val_proba)

    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_metric("val_f1_score", f1)
    mlflow.log_metric("val_auc_roc",  auc)

    # Log model artifact to MLflow
    mlflow.sklearn.log_model(model, "random_forest_model")

    # FIX: Capture and save run ID for later comparison runs
    run_id = run.info.run_id
    with open("model/run_id.txt", "w") as f:
        f.write(run_id)

    print(f"\n{'='*45}")
    print(f"  Validation Accuracy : {acc:.4f}")
    print(f"  Validation F1-Score : {f1:.4f}")
    print(f"  Validation AUC-ROC  : {auc:.4f}")
    print(f"  MLflow Run ID       : {run_id}")
    print(f"{'='*45}")

# ── 7. Persist Model for Simulator ────────────────────────────────────────────
joblib.dump(model, "model/model.pkl")
print("\nSaved model → model/model.pkl")

# ── 8. Summary ─────────────────────────────────────────────────────────────────
print("\n✅ Phase 2 Complete! Artifacts saved:")
print("   model/model.pkl        ← trained RandomForest")
print("   model/encoders.pkl     ← LabelEncoders for all categorical columns")
print("   model/feature_cols.pkl ← column order the model expects")
print("   model/run_id.txt       ← MLflow run ID for comparison")
print("   data/reference.csv     ← raw (pre-encoded) baseline for Evidently AI")
print("\nNext → Run Phase 3: python simulator/simulate_traffic.py")
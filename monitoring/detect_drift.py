import os
import sqlite3
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from evidently import Report
from evidently.presets import DataDriftPreset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR       = "data"
LOGS_DIR       = "logs"
REFERENCE_PATH = os.path.join(DATA_DIR, "reference.csv")
DB_PATH        = os.path.join(LOGS_DIR, "predictions.db")


# ── 1. Load Reference ──────────────────────────────────────────────────────────
def load_reference_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Reference data not found at {filepath}")
    ref_df = pd.read_csv(filepath)
    if ref_df.empty:
        raise ValueError("Reference dataframe is empty.")
    return ref_df.drop("target", axis=1, errors="ignore")


# ── 2. Align Columns & Dtypes ──────────────────────────────────────────────────
def preprocess_current_data(current_df: pd.DataFrame,
                             ref_features: pd.DataFrame) -> pd.DataFrame:
    aligned_df = pd.DataFrame(index=current_df.index)
    for col in ref_features.columns:
        if col not in current_df.columns:
            aligned_df[col] = pd.NA
        else:
            try:
                aligned_df[col] = current_df[col].astype(ref_features[col].dtype)
            except (ValueError, TypeError):
                aligned_df[col] = pd.to_numeric(current_df[col], errors="coerce")
    return aligned_df


# ── 3. Performance Metrics ─────────────────────────────────────────────────────
def calculate_performance_metrics(current_df: pd.DataFrame):
    if "target" not in current_df.columns or "prediction" not in current_df.columns:
        return None, None
    valid_df = current_df.dropna(subset=["target", "prediction"])
    if valid_df.empty:
        return None, None
    y_true = valid_df["target"].astype(str)
    y_pred = valid_df["prediction"].astype(str)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return round(acc, 4), round(f1, 4)


# ── 4. Extract Drift Metrics (Evidently 0.7.x Structure) ──────────────────────
#
# Confirmed structure from debug output:
#
#   metrics[0] → DriftedColumnsCount
#       config.drift_share  = 0.5    ← THRESHOLD (deep_search was grabbing this!)
#       value.share         = 0.64   ← REAL drift share (what we actually want)
#       value.count         = 9.0    ← number of drifted columns
#
#   metrics[1..N] → ValueDrift (one per feature)
#       config.column    = "age"
#       config.threshold = 0.05      ← p-value threshold
#       value            = 2.45e-15  ← raw p-value float (lower = more drift)
#
def extract_drift_metrics(report_dict: dict):
    dataset_drift = False
    drift_share   = 0.0
    n_drifted     = 0
    n_total       = 0
    per_feature   = {}

    for metric in report_dict.get("metrics", []):
        config      = metric.get("config", {})
        value       = metric.get("value", {})
        metric_type = config.get("type", "")

        # Overall dataset drift
        if metric_type == "evidently:metric_v2:DriftedColumnsCount":
            drift_share   = float(value.get("share", 0.0))
            n_drifted     = int(value.get("count", 0))
            threshold     = float(config.get("drift_share", 0.5))
            dataset_drift = drift_share >= threshold

        # Per-feature drift — value is a raw float p-value, NOT a nested dict
        elif metric_type == "evidently:metric_v2:ValueDrift":
            col            = config.get("column", "unknown")
            threshold      = float(config.get("threshold", 0.05))
            method         = config.get("method", "unknown")
            p_value        = float(value) if isinstance(value, (int, float)) else 1.0
            drift_detected = p_value < threshold

            per_feature[col] = {
                "p_value":        p_value,
                "drift_detected": drift_detected,
                "stattest_name":  method,
            }

    n_total = len(per_feature)
    return dataset_drift, drift_share, n_drifted, n_total, per_feature


# ── 5. Main ────────────────────────────────────────────────────────────────────
def main():
    logger.info("Starting Evidently AI Drift Detection (v0.7.x)...")

    try:
        ref_features = load_reference_data(REFERENCE_PATH)
        logger.info(f"Loaded reference data: {ref_features.shape}")
    except Exception as e:
        logger.error(f"Failed to load reference data: {e}")
        return

    if not os.path.exists(DB_PATH):
        logger.error(f"Database not found at {DB_PATH}. Run simulator first.")
        return

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Clear old drift tables so reruns stay clean
    cursor.execute("DROP TABLE IF EXISTS drift_metrics")
    cursor.execute("DROP TABLE IF EXISTS feature_drift")
    conn.commit()
    logger.info("Cleared old drift tables.")

    try:
        weeks = pd.read_sql(
            "SELECT DISTINCT week FROM predictions ORDER BY week", conn
        )["week"].tolist()

        if not weeks:
            logger.error("No prediction data found. Run simulator first.")
            return

        logger.info(f"Found weeks: {weeks}")

        for week in weeks:
            logger.info(f"\n{'─'*45}")
            logger.info(f"Analyzing Week {week}...")

            current_df = pd.read_sql(
                "SELECT * FROM predictions WHERE week = ?", conn, params=(week,)
            )
            if current_df.empty:
                logger.warning(f"No data for week {week}, skipping.")
                continue

            curr_features = preprocess_current_data(
                current_df.drop(
                    ["prediction", "confidence", "timestamp", "week", "target"],
                    axis=1, errors="ignore"
                ),
                ref_features
            )

            # Run Evidently
            report = Report(metrics=[DataDriftPreset(drift_share=0.4)])
            snapshot = report.run(
                reference_data=ref_features,
                current_data=curr_features
            )

            # Save HTML
            os.makedirs(LOGS_DIR, exist_ok=True)
            report_path = os.path.join(LOGS_DIR, f"drift_report_week_{week}.html")
            snapshot.save_html(report_path)
            logger.info(f"Saved HTML → {report_path}")

            try:
                report_dict = snapshot.as_dict()
            except AttributeError:
                report_dict = snapshot.dict()

            dataset_drift, drift_share, n_drifted, n_total, per_feature = \
                extract_drift_metrics(report_dict)

            logger.info(f"Drift Share      : {drift_share:.4f} ({drift_share*100:.1f}%)")
            logger.info(f"Dataset Drifted  : {dataset_drift}")
            logger.info(f"Drifted Features : {n_drifted} / {n_total}")

            acc, f1 = calculate_performance_metrics(current_df)
            if acc is not None:
                logger.info(f"Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

            # Save drift_metrics
            pd.DataFrame([{
                "week":               week,
                "drift_score":        round(drift_share, 4),
                "is_drifted":         int(dataset_drift),
                "n_drifted_features": n_drifted,
                "n_total_features":   n_total,
                "report_path":        report_path
            }]).to_sql("drift_metrics", conn, if_exists="append", index=False)

            # Save feature_drift
            if per_feature:
                feature_rows = [{
                    "week":        week,
                    "feature":     col,
                    "drift_score": round(data["p_value"], 6),
                    "is_drifted":  int(data["drift_detected"]),
                    "stat_test":   data["stattest_name"]
                } for col, data in per_feature.items()]

                pd.DataFrame(feature_rows).to_sql(
                    "feature_drift", conn, if_exists="append", index=False
                )

                # Print top 3 most drifted (lowest p-value = most drifted)
                top3 = sorted(feature_rows, key=lambda x: x["drift_score"])[:3]
                logger.info("Top drifted features:")
                for row in top3:
                    flag = "DRIFTED" if row["is_drifted"] else "OK"
                    logger.info(
                        f"  [{flag}] {row['feature']:<20} "
                        f"p={row['drift_score']:.6f}  ({row['stat_test']})"
                    )
            else:
                logger.warning("No feature-level drift data found.")

            # Save weekly_metrics
            if acc is not None:
                pd.DataFrame([{
                    "week": week, "accuracy": acc, "f1_score": f1
                }]).to_sql("weekly_metrics", conn, if_exists="append", index=False)

        # Final summary table
        logger.info(f"\n{'='*45}")
        logger.info("DRIFT SUMMARY:")
        summary = pd.read_sql(
            "SELECT week, drift_score, is_drifted, n_drifted_features "
            "FROM drift_metrics ORDER BY week", conn
        )
        print(summary.to_string(index=False))

        logger.info("\n✅ Phase 4 Complete!")
        logger.info("   drift_metrics  → overall drift score per week")
        logger.info("   feature_drift  → p-value per feature per week")
        logger.info("   weekly_metrics → accuracy + F1 per week")
        logger.info("\nNext → streamlit run dashboard/app.py")

    finally:
        conn.close()
        logger.info("Database connection closed.")


if __name__ == "__main__":
    main()
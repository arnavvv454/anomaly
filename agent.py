import os
import sys
import smtplib

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

import config as cfg

try:
    from google import genai
except ImportError:
    genai = None

try:
    import anthropic
except ImportError:
    anthropic = None


def excel_padho(file_path):
    sheet_name = getattr(cfg, "SHEET_NAME", 0)
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    date_col = getattr(cfg, "DATE_COLUMN", "Date")
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def anomaly_dhundo(df):
    anomalies = []
    val_col = getattr(cfg, "VALUE_COLUMN", "Revenue")
    date_col = getattr(cfg, "DATE_COLUMN", "Date")
    window = getattr(cfg, "ROLLING_WINDOW", 7)
    threshold = getattr(cfg, "Z_SCORE_THRESHOLD", 2)
    method = getattr(cfg, "ANOMALY_METHOD", "HYBRID").upper()
    contamination = getattr(cfg, "CONTAMINATION", 0.05)

    df["rolling_avg"] = df[val_col].rolling(window=window, min_periods=1).mean()
    df["rolling_std"] = df[val_col].rolling(window=window, min_periods=1).std().fillna(0)

    other_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["rolling_avg", "rolling_std"]]
    for col in other_cols:
        if col != val_col:
            df[f"{col}_rolling_avg"] = df[col].rolling(window=window, min_periods=1).mean()

    feature_cols = [c for c in other_cols if c in df.columns]
    if len(feature_cols) > 0 and len(df) >= 10:
        X = df[feature_cols].fillna(df[feature_cols].median())
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df["ml_is_outlier"] = iso_forest.fit_predict(X) == -1
        df["ml_score"] = -iso_forest.decision_function(X)
    else:
        df["ml_is_outlier"] = False
        df["ml_score"] = 0.0

    for i in range(len(df)):
        avg = df["rolling_avg"].iloc[i]
        std = df["rolling_std"].iloc[i]
        value = df[val_col].iloc[i]
        ml_flag = bool(df["ml_is_outlier"].iloc[i])
        ml_score = round(float(df["ml_score"].iloc[i]), 3)

        z_score = 0.0
        if std > 0:
            z_score = (value - avg) / std

        is_z_anomaly = abs(z_score) >= threshold
        is_ml_anomaly = ml_flag

        is_anomaly = False
        detection_type = ""
        if method == "Z_SCORE":
            is_anomaly = is_z_anomaly
            detection_type = "Z-Score"
        elif method == "ISOLATION_FOREST":
            is_anomaly = is_ml_anomaly
            detection_type = "Isolation Forest (ML)"
        else:
            if is_z_anomaly and is_ml_anomaly:
                is_anomaly = True
                detection_type = "Hybrid (Z-Score + ML)"
            elif is_z_anomaly:
                is_anomaly = True
                detection_type = "Z-Score"
            elif is_ml_anomaly and abs(z_score) > 1.5:
                is_anomaly = True
                detection_type = "Isolation Forest (ML)"

        if is_anomaly:
            if abs(z_score) >= 3 or (abs(z_score) >= 2 and ml_flag):
                severity = "CRITICAL"
            else:
                severity = "WARNING"

            correlations = {}
            for col in other_cols:
                if col != val_col:
                    actual_other = df[col].iloc[i]
                    exp_other = df[f"{col}_rolling_avg"].iloc[i] if f"{col}_rolling_avg" in df.columns else actual_other
                    diff_pct = 0.0
                    if exp_other and exp_other != 0:
                        diff_pct = round(((actual_other - exp_other) / exp_other) * 100, 1)
                    correlations[col] = {
                        "actual": actual_other,
                        "expected": round(exp_other, 2),
                        "change_pct": diff_pct
                    }

            date_val = df[date_col].iloc[i] if date_col in df.columns else i
            pct_change = round(((value - avg) / avg) * 100, 1) if avg != 0 else 0.0

            anomalies.append({
                "date": date_val,
                "value": value,
                "expected": round(avg, 2),
                "direction": "UP" if value > avg else "DOWN",
                "change_pct": pct_change,
                "z_score": round(z_score, 2),
                "ml_score": ml_score,
                "severity": severity,
                "detection_type": detection_type,
                "correlations": correlations
            })

    return anomalies



def agent():
    print(">> Running Hybrid Anomaly Detection with Cross-Metric Correlations")
    data = excel_padho(cfg.INPUT_FILE)
    anomalies = anomaly_dhundo(data)
    print(f"Detected {len(anomalies)} anomalies with correlation data.")
    for a in anomalies:
        print(f" - {a['date']}: {a['direction']} ({a['change_pct']}%) | Correlations: {list(a.get('correlations', {}).keys())}")

if __name__ == "__main__":
    agent()

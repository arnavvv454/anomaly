import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

import config as cfg

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
    if len(other_cols) > 0 and len(df) >= 10:
        X = df[other_cols].fillna(df[other_cols].median())
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

        z_score = (value - avg) / std if std > 0 else 0.0
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
            severity = "CRITICAL" if (abs(z_score) >= 3 or (abs(z_score) >= 2 and ml_flag)) else "WARNING"
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
                "detection_type": detection_type
            })
    return anomalies

def agent():
    print(">> Running Hybrid Anomaly Detection Agent (Z-Score + ML)")
    data = excel_padho(cfg.INPUT_FILE)
    print(f"Loaded {len(data)} rows.")
    anomalies = anomaly_dhundo(data)
    print(f"Detected {len(anomalies)} anomalies using Hybrid model.")
    for a in anomalies:
        print(f" - {a['date']}: {a['direction']} ({a['change_pct']}%) [{a['severity']} - {a['detection_type']}]")

if __name__ == "__main__":
    agent()

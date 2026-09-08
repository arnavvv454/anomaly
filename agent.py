import os
import sys
import pandas as pd
import numpy as np

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

    df["rolling_avg"] = df[val_col].rolling(window=window, min_periods=1).mean()
    df["rolling_std"] = df[val_col].rolling(window=window, min_periods=1).std().fillna(0)

    for i in range(len(df)):
        avg = df["rolling_avg"].iloc[i]
        std = df["rolling_std"].iloc[i]
        value = df[val_col].iloc[i]

        z_score = (value - avg) / std if std > 0 else 0.0

        if abs(z_score) >= threshold:
            date_val = df[date_col].iloc[i] if date_col in df.columns else i
            pct_change = round(((value - avg) / avg) * 100, 1) if avg != 0 else 0.0
            severity = "CRITICAL" if abs(z_score) >= 3 else "WARNING"
            anomalies.append({
                "date": date_val,
                "value": value,
                "expected": round(avg, 2),
                "direction": "UP" if value > avg else "DOWN",
                "change_pct": pct_change,
                "z_score": round(z_score, 2),
                "severity": severity
            })
    return anomalies

def agent():
    print(">> Baseline Anomaly Detection with Severity Tagging")
    data = excel_padho(cfg.INPUT_FILE)
    print(f"Loaded {len(data)} rows.")
    anomalies = anomaly_dhundo(data)
    print(f"Detected {len(anomalies)} anomalies.")
    for a in anomalies:
        print(f" - {a['date']}: {a['direction']} ({a['change_pct']}%) [{a['severity']}] | Z-Score: {a['z_score']}")

if __name__ == "__main__":
    agent()

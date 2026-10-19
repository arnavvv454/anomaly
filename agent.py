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


def summary_banao(anomalies):
    if not anomalies:
        return "Sab metrics normal hain, koi anomaly detect nahi hui."

    val_col = getattr(cfg, "VALUE_COLUMN", "Revenue")

    anomaly_text = ""
    for a in anomalies:
        date_str = a['date'].strftime('%Y-%m-%d') if hasattr(a['date'], 'strftime') else str(a['date'])
        corr_details = []
        for metric, data in a.get("correlations", {}).items():
            corr_details.append(f"{metric}: actual {data['actual']} vs expected ~{data['expected']} ({data['change_pct']}%)")
        corr_str = " | ".join(corr_details) if corr_details else "No additional metrics"

        anomaly_text += (
            f"- Date: {date_str} [{a['severity']} - {a['detection_type']}]\n"
            f"  {val_col}: {a['value']} (Expected: ~{a['expected']}, Change: {a['change_pct']}%, Z-Score: {a['z_score']})\n"
            f"  Correlated Indicators: {corr_str}\n\n"
        )

    prompt = f"""You are a Principal Business Intelligence AI Agent. Analyze the following detected anomalies in business metrics:

{anomaly_text}

Provide an executive, structured diagnostic report in crisp Hinglish (or English) covering:
1. **Executive Overview**: High-level impact and severity of the anomaly.
2. **Root Cause Analysis (Kyun hua?)**: Correlate {val_col} with the other metrics (e.g. Website Visits, Signups, Refunds). Explain the most likely underlying driver.
3. **Business Impact**: Quantify estimated deviation or impact.
4. **Actionable Recommendations**: 3 concrete, prioritized next steps for the engineering or marketing team.

Keep it concise, professional, and actionable."""

    gemini_key = getattr(cfg, "GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY")
    ai_provider = getattr(cfg, "AI_PROVIDER", "gemini").lower()

    if ai_provider == "gemini" and gemini_key and genai is not None:
        try:
            print("[AI] Generating insights via Google Gemini...")
            client = genai.Client(api_key=gemini_key)
            model_name = getattr(cfg, "GEMINI_MODEL", "gemini-2.5-flash")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"[Warning] Gemini API call fail hui: {e}. Fallback use kar rahe hain.")

    claude_key = getattr(cfg, "ANTHROPIC_API_KEY", None) or os.environ.get("ANTHROPIC_API_KEY")
    if (ai_provider == "anthropic" or not gemini_key) and claude_key and anthropic is not None:
        try:
            print("[AI] Generating insights via Anthropic Claude...")
            client = anthropic.Anthropic(api_key=claude_key)
            response = client.messages.create(
                model=getattr(cfg, "CLAUDE_MODEL", "claude-sonnet-4-5"),
                max_tokens=getattr(cfg, "MAX_TOKENS", 600),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"[Warning] Claude API call fail hui: {e}. Fallback use kar rahe hain.")

    print("[Notice] LLM API key set nahi hai. Smart Rule-Based Diagnostic Analysis generate ki ja rahi hai.")
    fallback = "### [AI Diagnostic Summary] Rule-Based Root-Cause Analysis\n\n"
    for a in anomalies:
        date_str = a['date'].strftime('%Y-%m-%d') if hasattr(a['date'], 'strftime') else str(a['date'])
        fallback += f"**{date_str} [{a['severity']} ALERT]**: {val_col} {a['direction']} by {abs(a['change_pct'])}% (Actual: {a['value']:,.2f} vs Expected: ~{a['expected']:,.2f})\n\n"

        drivers = []
        for metric, data in a.get("correlations", {}).items():
            if metric == "Website_Visits" and data["change_pct"] < -15:
                drivers.append(f"- Website Traffic me {abs(data['change_pct'])}% ka drop aaya ({data['actual']} visits). Server outage ya acquisition ad pause hone ki possibility hai.")
            elif metric == "Refunds" and data["change_pct"] > 25:
                drivers.append(f"- Refunds me {data['change_pct']}% ka spike hua ({data['actual']} refunds). Product quality ya payment gateway chargeback issue check karein.")
            elif metric == "Signups" and data["change_pct"] < -15:
                drivers.append(f"- Signups me {abs(data['change_pct'])}% ka drop aaya ({data['actual']}). Funnel drop-off ya registration page issue ho sakta hai.")

        if drivers:
            fallback += "**Probable Root Cause(s):**\n" + "\n".join(drivers) + "\n\n"
        else:
            fallback += "**Probable Root Cause:** Sudden statistical volatility detected by Isolation Forest ML model.\n\n"

        fallback += "**Recommended Action Items:**\n"
        fallback += "1. Immediate audit of payment gateway success rates & checkout logs.\n"
        fallback += "2. Check web analytics and server uptime for traffic anomalies.\n"
        fallback += "3. Verify data pipeline consistency for any missing batch transactions.\n"

    return fallback


def generate_html_email(anomalies, summary_text):
    val_col = getattr(cfg, "VALUE_COLUMN", "Revenue")
    total_anomalies = len(anomalies)
    has_critical = any(a.get("severity") == "CRITICAL" for a in anomalies)
    badge_color = "#dc2626" if has_critical else "#d97706"
    badge_text = "CRITICAL ALERT" if has_critical else "WARNING"

    formatted_summary = summary_text.replace("\n", "<br>")
    formatted_summary = formatted_summary.replace("**", "<strong>")
    formatted_summary = formatted_summary.replace("### ", "<h3 style='color:#1e293b; margin:10px 0 5px;'>")

    table_rows = ""
    for a in anomalies:
        date_str = a['date'].strftime('%Y-%m-%d') if hasattr(a['date'], 'strftime') else str(a['date'])
        direction_badge = (
            "<span style='background:#fee2e2; color:#b91c1c; padding:3px 8px; border-radius:12px; font-weight:600;'>🔻 DOWN</span>"
            if a["direction"] == "DOWN"
            else "<span style='background:#dcfce7; color:#15803d; padding:3px 8px; border-radius:12px; font-weight:600;'>🔺 UP</span>"
        )
        sev_color = "#dc2626" if a["severity"] == "CRITICAL" else "#d97706"

        table_rows += f"""
        <tr style="border-bottom: 1px solid #e2e8f0; font-size: 14px;">
            <td style="padding: 12px 10px; font-weight: 500;">{date_str}</td>
            <td style="padding: 12px 10px;">{direction_badge}</td>
            <td style="padding: 12px 10px; font-weight: 600;">{a['value']:,.2f}</td>
            <td style="padding: 12px 10px; color: #64748b;">~{a['expected']:,.2f}</td>
            <td style="padding: 12px 10px; font-weight: 600; color:{sev_color};">{a['change_pct']:+.1f}%</td>
            <td style="padding: 12px 10px;"><span style="color:{sev_color}; font-weight:700;">{a['severity']}</span></td>
            <td style="padding: 12px 10px; font-size:12px; color:#475569;">{a['detection_type']}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Anomaly Alert</title>
    </head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; margin: 0; padding: 24px;">
        <div style="max-width: 680px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);">
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 24px 30px; color: #ffffff;">
                <div style="display: inline-block; background: {badge_color}; color: #ffffff; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;">
                    {badge_text}
                </div>
                <h1 style="margin: 0; font-size: 22px; font-weight: 700; color: #ffffff;">Business Metric Anomaly Alert</h1>
                <p style="margin: 6px 0 0; color: #94a3b8; font-size: 13px;">AI & Machine Learning Automated Detection Engine</p>
            </div>

            <div style="display: table; width: 100%; border-bottom: 1px solid #e2e8f0; background: #f1f5f9;">
                <div style="display: table-cell; width: 33.33%; padding: 16px 20px; text-align: center; border-right: 1px solid #e2e8f0;">
                    <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Anomalies Detected</div>
                    <div style="font-size: 22px; font-weight: 800; color: #0f172a; margin-top: 4px;">{total_anomalies}</div>
                </div>
                <div style="display: table-cell; width: 33.33%; padding: 16px 20px; text-align: center; border-right: 1px solid #e2e8f0;">
                    <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Monitored Metric</div>
                    <div style="font-size: 18px; font-weight: 700; color: #2563eb; margin-top: 4px;">{val_col}</div>
                </div>
                <div style="display: table-cell; width: 33.33%; padding: 16px 20px; text-align: center;">
                    <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Detection Model</div>
                    <div style="font-size: 13px; font-weight: 700; color: #0f172a; margin-top: 6px;">Z-Score + ML Isolation Forest</div>
                </div>
            </div>

            <div style="padding: 24px 30px;">
                <div style="background: #f8fafc; border-left: 4px solid #6366f1; border-radius: 8px; padding: 18px 20px; margin-bottom: 24px;">
                    <div style="font-size: 12px; font-weight: 700; color: #4f46e5; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">
                        AI Root-Cause & Executive Insights
                    </div>
                    <div style="font-size: 14px; line-height: 1.6; color: #334155;">
                        {formatted_summary}
                    </div>
                </div>

                <h3 style="margin: 0 0 12px; font-size: 16px; color: #0f172a;">Detailed Anomaly Breakdown</h3>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; text-align: left;">
                        <thead>
                            <tr style="background: #f8fafc; color: #64748b; font-size: 12px; text-transform: uppercase; border-bottom: 2px solid #e2e8f0;">
                                <th style="padding: 10px;">Date</th>
                                <th style="padding: 10px;">Direction</th>
                                <th style="padding: 10px;">Actual</th>
                                <th style="padding: 10px;">Expected</th>
                                <th style="padding: 10px;">Change</th>
                                <th style="padding: 10px;">Severity</th>
                                <th style="padding: 10px;">Engine</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                </div>
            </div>

            <div style="background: #f8fafc; border-top: 1px solid #e2e8f0; padding: 16px 30px; text-align: center; color: #94a3b8; font-size: 12px;">
                This automated report was generated by <strong>Anomaly Project Agent</strong> with Machine Learning & AI Insights.
            </div>
        </div>
    </body>
    </html>
    """
    return html


def email_bhejo(anomalies, summary):
    send_if_no_anomaly = getattr(cfg, "SEND_EMAIL_IF_NO_ANOMALY", False)
    if not anomalies and not send_if_no_anomaly:
        print("Koi anomaly nahi, email skip kar rahe hain.")
        return

    sender = cfg.SENDER_EMAIL or os.environ.get("SENDER_EMAIL")
    password = cfg.SENDER_APP_PASSWORD or os.environ.get("SENDER_APP_PASSWORD")
    if password:
        password = password.replace(" ", "").strip()
    receiver = cfg.RECEIVER_EMAIL or os.environ.get("RECEIVER_EMAIL")
    smtp_host = getattr(cfg, "SMTP_HOST", "smtp.gmail.com")
    smtp_port = getattr(cfg, "SMTP_PORT", 587)
    use_html = getattr(cfg, "HTML_EMAIL", True)

    if not sender or not password or not receiver:
        print("[Notice] Email credentials set nahi hain. Email send skip kar rahe hain.")
        return

    val_col = getattr(cfg, "VALUE_COLUMN", "Revenue")

    plain_body = f"Anomaly Alert: {len(anomalies)} anomalies detect hui hain.\n\n"
    plain_body += summary + f"\n\n--- Details ({val_col}) ---\n"
    for a in anomalies:
        date_str = a['date'].strftime('%Y-%m-%d') if hasattr(a['date'], 'strftime') else str(a['date'])
        plain_body += f"{date_str}: {a['direction']} ({a['severity']}) — Actual={a['value']}, Expected~{a['expected']} (Change: {a['change_pct']}%)\n"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 Anomaly Alert: {len(anomalies)} Anomalies Detected in {val_col}"
    msg["From"] = sender
    msg["To"] = receiver

    part1 = MIMEText(plain_body, "plain", "utf-8")
    msg.attach(part1)

    if use_html:
        html_content = generate_html_email(anomalies, summary)
        part2 = MIMEText(html_content, "html", "utf-8")
        msg.attach(part2)

    try:
        print(f"Connecting to SMTP server ({smtp_host}:{smtp_port})...")
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        print(f"[OK] Rich HTML Email bhej diya {receiver} ko!")
    except Exception as e:
        print(f"[Warning] Email send fail hua: {e}")


def agent():
    cfg.validate(check_credentials=False)

    print("\n==========================================")
    print(">> Running Business Anomaly Detection Agent")
    print("==========================================")
    print(f"File: {cfg.INPUT_FILE}")
    print("Model: Z-Score + Isolation Forest ML (Hybrid)")

    data = excel_padho(cfg.INPUT_FILE)
    print(f"Loaded {len(data)} rows.")

    print("\n[*] Detecting anomalies with Machine Learning...")
    anomalies = anomaly_dhundo(data)
    print(f"Found {len(anomalies)} anomalies.")

    print("\n[*] Generating AI Root-Cause Diagnostic Summary...")
    summary = summary_banao(anomalies)
    print("\n" + summary + "\n")

    print("[*] Sending Rich HTML Email Alert...")
    email_bhejo(anomalies, summary)

    print("\nAgent ka kaam successfully complete ho gaya! [OK]\n")


if __name__ == "__main__":
    agent()

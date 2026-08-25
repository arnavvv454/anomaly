import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

INPUT_FILE = os.path.join(BASE_DIR, "business_metrics_sample.xlsx")
SHEET_NAME = 0
DATE_COLUMN = "Date"
VALUE_COLUMN = "Revenue"

ROLLING_WINDOW = 7
Z_SCORE_THRESHOLD = 2
ANOMALY_METHOD = os.environ.get("ANOMALY_METHOD", "HYBRID")
CONTAMINATION = 0.05

AI_PROVIDER = os.environ.get("AI_PROVIDER", "gemini")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 600
SUMMARY_LINES = "5-6"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_APP_PASSWORD = os.environ.get("SENDER_APP_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

SEND_EMAIL_IF_NO_ANOMALY = False
HTML_EMAIL = True


def validate(check_credentials=True):
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    if check_credentials:
        missing = [
            name for name, value in {
                "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY,
                "SENDER_EMAIL": os.environ.get("SENDER_EMAIL") or SENDER_EMAIL,
                "SENDER_APP_PASSWORD": os.environ.get("SENDER_APP_PASSWORD") or SENDER_APP_PASSWORD,
                "RECEIVER_EMAIL": os.environ.get("RECEIVER_EMAIL") or RECEIVER_EMAIL,
            }.items() if not value
        ]
        if missing:
            raise EnvironmentError(
                "Missing environment variables: " + ", ".join(missing) +
                "\nPlease set them in a .env file (see .env.example) or system environment."
            )

from pathlib import Path
import os
from datetime import datetime
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).parent
# Small .env reader: process environment always takes precedence.
if (BASE_DIR / ".env").exists():
    for line in (BASE_DIR / ".env").read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "class_monitor.db"
APP_TITLE = "Nexus Vision | Smart Class Monitor"
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
APP_TIMEZONE = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Kolkata"))
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", "0.18"))
YAW_THRESHOLD = float(os.getenv("YAW_THRESHOLD", "20"))
PITCH_THRESHOLD = float(os.getenv("PITCH_THRESHOLD", "18"))
SIGNAL_DWELL_SECONDS = float(os.getenv("SIGNAL_DWELL_SECONDS", "2"))
MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.7"))
PHONE_CONFIDENCE = float(os.getenv('PHONE_CONFIDENCE', '0.10'))
PHONE_DWELL_SECONDS = float(os.getenv('PHONE_DWELL_SECONDS', '0.25'))
MODEL_DIR = BASE_DIR / 'models'
FACE_MATCH_THRESHOLD = float(os.getenv('FACE_MATCH_THRESHOLD', '0.50'))
FACE_MATCH_MARGIN = float(os.getenv('FACE_MATCH_MARGIN', '0.08'))
RECOGNITION_SECONDS = float(os.getenv('RECOGNITION_SECONDS', '2'))
INFERENCE_INTERVAL = float(os.getenv('INFERENCE_INTERVAL', '0.40'))
PHONE_INFERENCE_INTERVAL = float(os.getenv('PHONE_INFERENCE_INTERVAL', '0.35'))
LIVE_MAX_DIMENSION = int(os.getenv('LIVE_MAX_DIMENSION', '640'))
EVENT_COOLDOWN_SECONDS = float(os.getenv('EVENT_COOLDOWN_SECONDS', '60'))
CONVERSATION_DWELL_SECONDS = float(os.getenv('CONVERSATION_DWELL_SECONDS', '2'))
CONVERSATION_DISTANCE_RATIO = float(os.getenv('CONVERSATION_DISTANCE_RATIO', '2.2'))

def local_now():
    return datetime.now(APP_TIMEZONE)

def ensure_directories():
    (DATA_DIR / "faces").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "static").mkdir(parents=True, exist_ok=True)

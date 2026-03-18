
import os

"""
==============================================================================
MOLTY ROYALE BOT - CONFIGURATION
==============================================================================
Environment-variable friendly settings for Railway / Docker.
"""

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)

API_KEY = _env_str("API_KEY", "mr_live_xxxxxxxxxxxxxxxxxxxx")
BASE_URL = _env_str("BASE_URL", "https://cdn.moltyroyale.com/api")
WALLET_ADDRESS = _env_str("WALLET_ADDRESS", "0xxxxxxxxxxxxxxxxxxxx")

PREFERRED_GAME_TYPE = _env_str("PREFERRED_GAME_TYPE", "free")
AUTO_CREATE_GAME = _env_bool("AUTO_CREATE_GAME", False)
GAME_MAP_SIZE = _env_str("GAME_MAP_SIZE", "medium")

HP_CRITICAL = _env_int("HP_CRITICAL", 65)
HP_LOW = _env_int("HP_LOW", 45)
EP_MIN_ATTACK = _env_int("EP_MIN_ATTACK", 2)
EP_REST_THRESHOLD = _env_int("EP_REST_THRESHOLD", 3)

WIN_PROBABILITY_ATTACK = _env_float("WIN_PROBABILITY_ATTACK", 0.65)
WIN_PROBABILITY_AGGRESSIVE = _env_float("WIN_PROBABILITY_AGGRESSIVE", 0.80)

PHASE_EARLY_PVP_BIAS = _env_float("PHASE_EARLY_PVP_BIAS", -0.10)
PHASE_MID_PVP_BIAS = _env_float("PHASE_MID_PVP_BIAS", 0.00)
PHASE_LATE_PVP_BIAS = _env_float("PHASE_LATE_PVP_BIAS", 0.08)
ZONE_ESCAPE_PRIORITY = _env_float("ZONE_ESCAPE_PRIORITY", 1.20)
MEDICAL_FACILITY_HP_THRESHOLD = _env_int("MEDICAL_FACILITY_HP_THRESHOLD", 60)
MIN_FREE_INVENTORY_SLOTS = _env_int("MIN_FREE_INVENTORY_SLOTS", 1)
MONSTER_FARM_SCORE_MIN = _env_float("MONSTER_FARM_SCORE_MIN", 0.58)

LEARNING_ENABLED = _env_bool("LEARNING_ENABLED", True)
DATA_DIR = _env_str("DATA_DIR", "data")
MIN_GAMES_FOR_ML = _env_int("MIN_GAMES_FOR_ML", 5)
LEARNING_RATE = _env_float("LEARNING_RATE", 0.1)

REDIS_ENABLED = _env_bool("REDIS_ENABLED", False)
REDIS_HOST = _env_str("REDIS_HOST", "localhost")
REDIS_PORT = _env_int("REDIS_PORT", 6379)
REDIS_DB = _env_int("REDIS_DB", 0)

LOG_LEVEL = _env_str("LOG_LEVEL", "DEBUG")
LOG_TO_FILE = _env_bool("LOG_TO_FILE", True)
LOG_FILE = _env_str("LOG_FILE", "logs/bot.log")

TURN_INTERVAL = _env_int("TURN_INTERVAL", 60)
POLL_INTERVAL_WAITING = _env_int("POLL_INTERVAL_WAITING", 5)
POLL_INTERVAL_DEAD = _env_int("POLL_INTERVAL_DEAD", 60)
ROOM_HUNT_INTERVAL = _env_int("ROOM_HUNT_INTERVAL", 5)
HEARTBEAT_INTERVAL = _env_int("HEARTBEAT_INTERVAL", 300)

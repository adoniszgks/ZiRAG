# Standard libs
from os import getenv
from pathlib import Path

# 3rdparty libs
from dotenv import load_dotenv

load_dotenv()

DRIVE_DIR = Path(getenv("DRIVE_DIR", "/default/path"))
CACHE_DIR = DRIVE_DIR / "cache"
DATA_DIR = DRIVE_DIR / "data"

LLM_API_KEY = getenv("LLM_API_KEY", "")
LLM_MODEL = getenv("LLM_MODEL", "")

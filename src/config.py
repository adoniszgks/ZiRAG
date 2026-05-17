# Standard libs
from os import getenv
from pathlib import Path

# 3rdparty libs
from dotenv import load_dotenv

load_dotenv()

DRIVE_DIR: Path = Path(getenv("DRIVE_DIR", "/default/path"))
CACHE_DIR: Path = DRIVE_DIR / "cache"
DATA_DIR: Path = DRIVE_DIR / "data"

LLM_API_KEY: str = getenv("LLM_API_KEY", "")
LLM_MODEL: str = getenv("LLM_MODEL", "")

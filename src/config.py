# Standard libs
import os
from pathlib import Path

# 3rdparty libs
from dotenv import load_dotenv

load_dotenv()

DRIVE_DIR = Path(os.getenv("DRIVE_DIR", "/default/path"))
CACHE_DIR = DRIVE_DIR / "cache"
DATA_DIR = DRIVE_DIR / "data"
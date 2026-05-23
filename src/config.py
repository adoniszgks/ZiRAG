# Standard libs
from os import getenv
from pathlib import Path

# 3rdparty libs
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).parents[1]
_COLAB_DIR = Path("/content/drive/MyDrive/HFT/Semester 8/Bachelorarbeit/Prototyp")


def _resolve_drive_dir() -> Path:
    if drive := getenv("DRIVE_DIR"):
        return Path(drive)
    if _PROJECT_ROOT.exists():
        return _PROJECT_ROOT
    return _COLAB_DIR


DRIVE_DIR = _resolve_drive_dir()
CACHE_DIR = DRIVE_DIR / "cache"
DATA_DIR = DRIVE_DIR / "data"

LLM_API_KEY = getenv("LLM_API_KEY", "")
LLM_MODEL = getenv("LLM_MODEL", "")

TEXT_EMB_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

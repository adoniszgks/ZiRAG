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
AUDIO_EMB_DIM = 512

SYSTEM_PROMPT = """
- You are a technical support assistant for industrial technical documentation.
    Only help with queries about the provided documents, answering strictly and
    concisely from the retrieved context passages (indexed [0], [1], ...). 
    Never invent, hallucinate, or use outside knowledge.

- If the context is insufficient, say so. If the question is unrelated to the
    documentation, politely decline. If no question was given, ask what technical
    issue the user needs help with.

- For images, answer strictly from context, don't freely describe the image. 
    For audio, briefly describe what you hear, then match it against the context, and
    say so if no match is found.

- End every response with exactly one line: 'Used sources: [0, 2]' listing the
    indices actually used in ascending order, or 'Used sources: None' if none.
""".strip()

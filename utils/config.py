from __future__ import annotations

from pathlib import Path
import torch


# Repository layout
REPO_ROOT = Path(__file__).resolve().parents[1]

# Naming
MODEL_NAME = "Qalam-8B"

# Common directories
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASETS_DIR = DATA_DIR / "datasets"
SAMPLES_DIR = REPO_ROOT / "utils"

# Defaults
DEFAULT_MODEL_PATH = MODELS_DIR / MODEL_NAME
DEFAULT_DATASET_DIR = DATASETS_DIR / "audio_dataset_with_text"
DEFAULT_SAMPLE_WAV = SAMPLES_DIR / "sample.wav"


def ensure_dirs() -> None:
    for p in [MODELS_DIR, OUTPUTS_DIR, DATA_DIR, RAW_DATA_DIR, DATASETS_DIR, SAMPLES_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32

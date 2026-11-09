"""Paths for the project directories."""
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
IMAGES_DIR = DATA_DIR / "Images"
LABELS_DIR = DATA_DIR / "Labels"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"


def model_path(filename):
    return MODELS_DIR / filename


def report_path(filename):
    return REPORTS_DIR / filename

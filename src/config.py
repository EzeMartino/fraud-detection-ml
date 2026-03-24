from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LATEST_FILE = MODELS_DIR / "latest.txt"

DATA_FILE = RAW_DATA_DIR / "creditcard.csv"
TARGET_COLUMN = "Class"

PIPELINE_FILENAME = "model.joblib"
METADATA_FILENAME ="metadata.json"
CONFIG_FILENAME ="config.json"

def get_active_model_name() -> str:
    if not LATEST_FILE.exists():
        raise FileNotFoundError("latest.txt not found. Export a model first.")

    name = LATEST_FILE.read_text(encoding="utf-8").strip()
    if not name:
        raise ValueError("latest.txt is empty.")

    model_dir = MODELS_DIR / name
    if not model_dir.exists():
        raise FileNotFoundError(f"Active model directory does not exist: {model_dir}")

    return name

def get_active_model_dir() -> Path:
    return MODELS_DIR / get_active_model_name()

def get_active_pipeline_path() -> Path:
    return get_active_model_dir() / PIPELINE_FILENAME

def get_active_metadata_path() -> Path:
    return get_active_model_dir() / METADATA_FILENAME

def get_best_config_path() -> Path:
    return ARTIFACTS_DIR / CONFIG_FILENAME

THRESHOLD = 0.5
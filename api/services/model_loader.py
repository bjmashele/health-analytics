from datetime import datetime, timezone
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]

RISK_MODEL_PATH = BASE_DIR / "models" / "risk_model.joblib"
CLAIM_MODEL_PATH = BASE_DIR / "models" / "claim_model.joblib"


def _load(path: Path, model_name: str):
    if not path.exists():
        raise FileNotFoundError(f"{model_name} not found at: {path}")

    artifact = joblib.load(path)

    # The notebook dumps a bare sklearn Pipeline, but allow artifacts that
    # already carry their registry metadata as (model, name, version).
    if isinstance(artifact, tuple) and len(artifact) == 3:
        return artifact

    mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return artifact, model_name, mtime.strftime("%Y%m%d%H%M%S")


def load_risk_model():
    return _load(RISK_MODEL_PATH, "risk_model")


def load_claim_model():
    return _load(CLAIM_MODEL_PATH, "claim_model")

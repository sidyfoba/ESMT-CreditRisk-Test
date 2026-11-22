"""Load versioned test artifact and predict a probability."""
import json
from pathlib import Path
import joblib
import numpy as np
from app.preprocessing import FEATURES, to_matrix

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / 'model' / 'credit_model.joblib'
META_PATH = ROOT / 'model' / 'metadata.json'


class ModelUnavailable(Exception):
    pass


class PredictionFailed(Exception):
    pass


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except (FileNotFoundError, OSError, ValueError, EOFError) as exc:
        raise ModelUnavailable('modele indisponible') from exc


def model_info():
    try:
        return json.loads(META_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ModelUnavailable('metadonnees indisponibles') from exc


def predict(applicant):
    model = load_model()
    try:
        values = to_matrix(applicant)
        if getattr(model, 'n_features_in_', len(FEATURES)) != len(FEATURES):
            raise ValueError('dimensions du modele incorrectes')
        probability = float(model.predict_proba(values)[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('probabilite invalide')
        return probability
    except Exception as exc:
        raise PredictionFailed('prediction impossible') from exc

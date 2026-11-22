import json
from pathlib import Path
import joblib
import pytest
from app.prediction import MODEL_PATH, predict
from app.validation import Applicant

ROOT = Path(__file__).resolve().parent.parent
BASE = json.loads((ROOT / 'model' / 'baseline.json').read_text())


def test_golden_prediction(applicant):
    actual = predict(Applicant(**applicant))
    assert abs(actual - BASE['probability']) <= BASE['tolerance']


def test_metrics_not_regressed():
    metadata = json.loads((ROOT / 'model' / 'metadata.json').read_text())
    assert metadata['test_metrics']['recall'] >= BASE['recall_min']
    assert metadata['test_metrics']['f1'] >= BASE['f1_min']


def test_save_reload_consistency(applicant, tmp_path):
    model = joblib.load(MODEL_PATH)
    path = tmp_path / 'reloaded.joblib'
    joblib.dump(model, path)
    other = joblib.load(path)
    from app.preprocessing import to_matrix
    values = to_matrix(Applicant(**applicant))
    assert other.predict_proba(values)[0, 1] == pytest.approx(model.predict_proba(values)[0, 1])

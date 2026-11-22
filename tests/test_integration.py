from app import prediction
from app.preprocessing import to_matrix
from app.validation import Applicant


def test_full_pipeline(client, applicant):
    x = to_matrix(Applicant(**applicant))
    assert x.shape == (1, 5)
    response = client.post('/predict', json=applicant)
    assert response.status_code == 200
    assert abs(response.json()['probabilite_defaut'] - prediction.predict(Applicant(**applicant))) < 1e-12
    assert client.get('/model-info').json()['features'] == list(applicant)


def test_preprocessing_error_is_masked(client, applicant, monkeypatch):
    def fail(_):
        raise ValueError('sensitive internal details')
    monkeypatch.setattr(prediction, 'to_matrix', fail)
    r = client.post('/predict', json=applicant)
    assert r.status_code == 500
    assert 'sensitive' not in r.text


def test_prediction_error_is_masked(client, applicant, monkeypatch):
    class Broken:
        n_features_in_ = 5
        def predict_proba(self, _):
            raise RuntimeError('private traceback')
    monkeypatch.setattr(prediction, 'load_model', lambda: Broken())
    r = client.post('/predict', json=applicant)
    assert r.status_code == 500 and 'private' not in r.text


def test_wrong_model_dimensions(client, applicant, monkeypatch):
    class Wrong:
        n_features_in_ = 4
    monkeypatch.setattr(prediction, 'load_model', lambda: Wrong())
    assert client.post('/predict', json=applicant).status_code == 500


def test_missing_variable(client, applicant):
    del applicant['anciennete_activite']
    assert client.post('/predict', json=applicant).status_code == 422

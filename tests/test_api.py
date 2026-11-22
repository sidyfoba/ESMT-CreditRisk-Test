import math
import pytest


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200 and r.json()['status'] == 'ok'


def test_model_info(client):
    r = client.get('/model-info')
    assert r.status_code == 200 and r.json()['version'] == 'synthetic-v1.0'


def test_valid_request(client, applicant):
    r = client.post('/predict', json=applicant)
    assert r.status_code == 200
    p = r.json()['probabilite_defaut']
    assert math.isfinite(p) and 0 <= p <= 1
    assert 'avertissement' in r.json()


@pytest.mark.parametrize('field,value', [('age', -1), ('revenu_mensuel', 'test'),
    ('montant_credit', 0), ('montant_credit', -50), ('duree_mois', 12.5),
    ('revenu_mensuel', 1e110), ('montant_credit', 1e110)])
def test_invalid_request(client, applicant, field, value):
    applicant[field] = value
    r = client.post('/predict', json=applicant)
    assert r.status_code == 422 and r.json()['detail']


def test_missing_field(client, applicant):
    del applicant['age']
    assert client.post('/predict', json=applicant).status_code == 422


def test_extra_field(client, applicant):
    applicant['unknown_field'] = 'untrusted'
    assert client.post('/predict', json=applicant).status_code == 422


def test_malformed_json(client):
    r = client.post('/predict', content='{invalid', headers={'content-type': 'application/json'})
    assert r.status_code == 422


def test_unavailable_model(client, applicant, monkeypatch):
    from app import prediction
    def unavailable():
        raise prediction.ModelUnavailable()
    monkeypatch.setattr(prediction, 'load_model', unavailable)
    assert client.post('/predict', json=applicant).status_code == 503
    assert client.get('/health').status_code == 503

import math
import pytest


@pytest.mark.parametrize('field,value', [('age', None), ('age', '40'),
    ('revenu_mensuel', None), ('revenu_mensuel', 'NaN'),
    ('revenu_mensuel', 'Infinity'), ('duree_mois', '12'),
    ('duree_mois', 10.5), ('anciennete_activite', None)])
def test_bad_types(client, applicant, field, value):
    applicant[field] = value
    assert client.post('/predict', json=applicant).status_code == 422


def test_wrong_schema(client):
    assert client.post('/predict', json=[1, 2]).status_code == 422


def test_missing_all(client):
    assert client.post('/predict', json={}).status_code == 422


def test_valid_unusual_values(client, applicant):
    applicant.update(age=100, revenu_mensuel=100_000_000, montant_credit=1_000_000_000,
                     duree_mois=360, anciennete_activite=82)
    r = client.post('/predict', json=applicant)
    assert r.status_code == 200
    assert math.isfinite(r.json()['probabilite_defaut'])


def test_nonfinite_internal_validation(applicant):
    from pydantic import ValidationError
    from app.validation import Applicant
    applicant['revenu_mensuel'] = float('inf')
    with pytest.raises(ValidationError):
        Applicant(**applicant)

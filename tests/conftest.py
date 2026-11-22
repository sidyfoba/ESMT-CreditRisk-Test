import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def applicant():
    return {'age': 35, 'revenu_mensuel': 250000, 'montant_credit': 1000000,
            'duree_mois': 12, 'anciennete_activite': 4}

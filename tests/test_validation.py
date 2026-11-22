import pytest
from pydantic import ValidationError
from app.validation import Applicant


def test_valid(applicant):
    assert Applicant(**applicant).age == 35


@pytest.mark.parametrize('field,value', [('age', 17), ('age', 101), ('age', -1),
    ('revenu_mensuel', 0), ('revenu_mensuel', -10), ('revenu_mensuel', 'texte'),
    ('montant_credit', 0), ('montant_credit', -1), ('duree_mois', 1.5),
    ('duree_mois', 0), ('duree_mois', 361), ('anciennete_activite', -1),
    ('anciennete_activite', 50)])
def test_invalid_field(applicant, field, value):
    applicant[field] = value
    with pytest.raises(ValidationError):
        Applicant(**applicant)


def test_boundaries(applicant):
    applicant.update(age=18, anciennete_activite=0, duree_mois=1)
    Applicant(**applicant)
    applicant.update(age=100, anciennete_activite=82, duree_mois=360)
    Applicant(**applicant)

"""Shared input ordering for training and inference."""
import numpy as np
from app.validation import Applicant

FEATURES = ('age', 'revenu_mensuel', 'montant_credit', 'duree_mois', 'anciennete_activite')


def to_matrix(applicant: Applicant) -> np.ndarray:
    values = [getattr(applicant, name) for name in FEATURES]
    array = np.asarray([values], dtype=np.float64)
    if array.shape != (1, len(FEATURES)) or not np.isfinite(array).all():
        raise ValueError('dimensions ou valeurs invalides')
    return array

"""Microfinance risk API. Educational prototype; no automatic credit decision."""
from fastapi import FastAPI, HTTPException
from app.prediction import ModelUnavailable, PredictionFailed, model_info, predict
from app.validation import Applicant

app = FastAPI(title='CreditRisk-Test', version='1.0.0')


@app.get('/health')
def health():
    try:
        info = model_info()
        from app.prediction import load_model
        load_model()
        return {'status': 'ok', 'model_version': info['version']}
    except ModelUnavailable:
        raise HTTPException(status_code=503, detail='Service de prediction indisponible') from None


@app.get('/model-info')
def model_metadata():
    try:
        return model_info()
    except ModelUnavailable:
        raise HTTPException(status_code=503, detail='Metadonnees indisponibles') from None


@app.post('/predict')
def predict_credit(applicant: Applicant):
    try:
        p = predict(applicant)
        return {'probabilite_defaut': p,
                'niveau_risque': 'eleve' if p >= 0.5 else 'faible',
                'seuil_indicatif': 0.5,
                'avertissement': 'Prototype pedagogique : examen humain obligatoire'}
    except ModelUnavailable:
        raise HTTPException(status_code=503, detail='Modele indisponible') from None
    except PredictionFailed:
        raise HTTPException(status_code=500, detail='Erreur interne de prediction') from None

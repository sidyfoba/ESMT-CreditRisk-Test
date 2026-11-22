"""Deterministic synthetic training and reference snapshot; not real-world validation."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'model'
DEST.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)
n = 2400
age = rng.integers(18, 75, n)
income = rng.uniform(80000, 1400000, n)
credit = rng.uniform(150000, 10_000_000, n)
months = rng.integers(3, 84, n)
years = rng.random(n) * np.maximum(age - 17, 0)
x = np.column_stack((age, income, credit, months, years))
# Artificial label-generating rule, expressly NOT a real credit-risk mechanism.
logit = -1.2 + 1.3 * (credit / (income * months) - 0.8) - 0.028 * years
logit = np.clip(logit, -8, 8)
p = 1 / (1 + np.exp(-logit))
y = rng.binomial(1, p)
xtr, xte, ytr, yte = train_test_split(x, y, stratify=y, test_size=0.25, random_state=2026)
model = make_pipeline(StandardScaler(), LogisticRegression(random_state=2026, max_iter=800))
model.fit(xtr, ytr)
pred = model.predict(xte)
metrics = {'recall': round(float(recall_score(yte, pred, zero_division=0)), 6),
           'f1': round(float(f1_score(yte, pred, zero_division=0)), 6)}
joblib.dump(model, DEST / 'credit_model.joblib')
meta = {'version': 'synthetic-v1.0', 'features': ['age', 'revenu_mensuel', 'montant_credit', 'duree_mois', 'anciennete_activite'],
        'data_type': 'synthetic', 'samples': n, 'seed': 2026, 'decision_threshold': 0.5,
        'test_metrics': metrics, 'limitations': 'Donnees synthetiques ; non valide pour octroi reel'}
(DEST / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
ref_input = np.asarray([[35, 250000, 1000000, 12, 4]], dtype=float)
ref = {'probability': float(model.predict_proba(ref_input)[0, 1]),
       'recall_min': max(0, metrics['recall'] - 0.03),
       'f1_min': max(0, metrics['f1'] - 0.03), 'tolerance': 0.000001,
       'metrics': metrics}
(DEST / 'baseline.json').write_text(json.dumps(ref, indent=2), encoding='utf-8')
print(json.dumps(meta, indent=2))

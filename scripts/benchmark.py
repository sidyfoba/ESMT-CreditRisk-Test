"""Small local benchmark; results depend on machine and are not a load-test claim."""
import json
import platform
import statistics
import time
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.prediction import predict
from app.validation import Applicant

sample = {'age': 35, 'revenu_mensuel': 250000, 'montant_credit': 1000000,
          'duree_mois': 12, 'anciennete_activite': 4}
client = TestClient(app)
client.post('/predict', json=sample)
a = Applicant(**sample)
api_ms, model_ms = [], []
for _ in range(100):
    start = time.perf_counter()
    res = client.post('/predict', json=sample)
    api_ms.append((time.perf_counter() - start) * 1000)
    if res.status_code != 200:
        raise RuntimeError(res.text)
    start = time.perf_counter()
    predict(a)
    model_ms.append((time.perf_counter() - start) * 1000)
sorted_api = sorted(api_ms)
report = {'environment': platform.platform(), 'python': platform.python_version(),
          'method': '100 sequential in-process TestClient calls (not network/concurrent)',
          'count': len(api_ms), 'mean_api_ms': round(statistics.mean(api_ms), 3),
          'p95_api_ms': round(sorted_api[94], 3),
          'mean_model_pipeline_ms': round(statistics.mean(model_ms), 3),
          'approx_throughput_requests_sec': round(100 / (sum(api_ms) / 1000), 2),
          'proposed_p95_target_ms': 500,
          'p95_target_met_in_local_test': sorted_api[94] <= 500}
path = Path(__file__).resolve().parent.parent / 'reports' / 'performance.json'
path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))

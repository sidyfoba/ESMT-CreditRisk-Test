# CreditRisk-Test — assurance qualité d’une IA de risque de crédit

**Projet pédagogique individuel — auteur : CHEIKH SIDY MOCTAR FOBA** 
**Important :** les observations sont synthétiques ; **ne pas utiliser pour refuser ou accorder un crédit réel**.

## Objet
Prototype FastAPI de prédiction du défaut de remboursement et stratégie de tests de validation, intégration, robustesse, non-régression, performance et CI. Les 5 variables sont `age`, `revenu_mensuel`, `montant_credit`, `duree_mois`, `anciennete_activite`.

## Démarrage (depuis le dossier racine)
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/train_model.py     # uniquement si le modèle fourni est absent / régénération voulue
uvicorn app.main:app --reload
```
Ouvrir `http://127.0.0.1:8000/docs` pour tester les endpoints `GET /health`, `GET /model-info` et `POST /predict`.

Exemple POST `/predict` :
```json
{"age":35,"revenu_mensuel":250000,"montant_credit":1000000,"duree_mois":12,"anciennete_activite":4}
```
Statuts : 200 succès ; 422 données invalides ; 503 modèle indisponible ; 500 erreur de pipeline masquée.

## Tests et rapports
```bash
python -m pytest tests -q --cov=app --cov-report=term-missing --junitxml=reports/junit.xml
python -m scripts.benchmark
ruff check app tests scripts
```
Fichiers : `reports/matrice_tracabilite.csv`, `reports/performance.json`, `reports/test_run.txt`, `reports/rapport_qualification.docx`. Baseline : `model/baseline.json` et `model/metadata.json`.

## CI GitHub Actions
Le workflow `.github/workflows/tests.yml` se déclenche aux `push` et `pull_request` : installation, Ruff, Pytest, couverture, conservation des rapports. Pour démontrer un échec de CI, introduire **temporairement sur une branche de démonstration** `assert False` dans un test, constater le rouge dans GitHub Actions, puis revenir à la version saine (ne pas conserver la faute).

**Le dépôt GitHub et les exécutions sur GitHub ne sont pas créés depuis cette archive.** Créez un dépôt `CreditRisk-Test`, poussez le contenu et ajoutez une capture de la CI à la remise.

## Contrat et décisions
Age 18–100 ; revenu (0 ; 100 000 000] ; crédit (0 ; 1 000 000 000] ; durée entière 1–360 ; ancienneté 0–82 cohérente avec âge. Les bornes sont des **hypothèses de test du prototype**, non des politiques de crédit officielles. Pour les montants extrêmes mais dans les bornes, le prototype renvoie une prédiction ; une détection d'hors-distribution et une revue humaine seraient nécessaires avant exploitation.

Le seuil 0,5 n'est qu'illustratif, pas calibré métier. Le rappel faible impose **une décision NO-GO pour toute mise en production ou pilote opérationnel fondé sur des décisions de crédit**. Une démonstration technique en environnement fermé est possible. Un test local in-process sur 100 requêtes séquentielles n'est pas une preuve de débit réel ni de résistance sous charge concurrente.

## Synthèse de cours (pour soutenance)
1. **Validation** : refuser les entrées impossibles avant le modèle.
2. **Prétraitement** : transformer les 5 valeurs dans un ordre strict.
3. **Inférence** : produire un score numérique, non une décision bancaire.
4. **Tests fonctionnels** : confirmer les contrats HTTP 200/422/503.
5. **Tests d'intégration** : prouver le fonctionnement entre modules et masquer les erreurs.
6. **Robustesse** : vérifier bornes, types, valeurs manquantes et scénarios atypiques.
7. **Non-régression** : comparer score de référence et métriques avec des tolérances prédéfinies.
8. **Performance** : mesurer latence moyenne et P95, documenter les limites du banc.
9. **CI** : exécuter automatiquement tests et analyse du code à chaque PR.
10. **Qualification** : distinguer « logiciel testé » et « modèle validé métier ».

## Limites
Données simulées, pas de validation sur population réelle, pas de test d'équité par groupes, pas d'analyse de dérive ni de conformité réglementaire, pas de véritables tests concurrents. Les métriques sont calculées sur un seul découpage train/test artificiel. Comparaison de non-régression sur métadonnées versionnées ; ne remplace pas une évaluation indépendante d'une nouvelle version réentraînée.

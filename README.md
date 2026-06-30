# 🏭 Maintenance Prédictive Industrielle — RUL
## Prédiction de la Durée de Vie Restante des Équipements

> **EFREI — Data Engineering & AI | Projet Data Science 2025-26**  
> Epreuve certifiante RNCP40875 — Bloc 2 | Enseignante : Sarah Malaeb

---
 
## Structure du projet

```
.
├── maintenance_predictive_RUL.ipynb   ← Notebook complet (EDA + FE + modèles + SHAP + SMOGN)
├── api.py                             ← API REST FastAPI v2.0
├── dashboard_technique.py             ← Dashboard EDA & Preprocessing (équipe Data Science)
├── dashboard_client.py                ← Dashboard décisionnel (responsable maintenance)
├── models/
│   └── best_model_pipeline.pkl        ← Pipeline sklearn sérialisé (Random Forest)
├── predictive_maintenance_v3.csv      ← Dataset (24 042 lignes × 15 variables)
└── README.md
```

---

## Installation des dépendances

```bash
pip install scikit-learn xgboost shap pandas numpy matplotlib seaborn \
            fastapi uvicorn streamlit plotly joblib smogn statsmodels
```

---

## 1. Notebook — Analyse & Modélisation

```bash
jupyter notebook maintenance_predictive_RUL.ipynb
```

**Sections couvertes :**
- EDA : valeurs manquantes, distributions, corrélations, analyse par machine
- Feature Engineering temporel : `cumcount`, `hours_sq`, interactions capteurs
- Pipeline preprocessing sans data leakage (sklearn Pipeline + ColumnTransformer)
- 4 modèles : Ridge · Random Forest · XGBoost · MLP (Deep Learning)
- Validation croisée 5-fold (KFold, shuffle=True)
- Feature Importance native + Permutation Importance + SHAP (summary + force plot)
- Gestion du déséquilibre : SMOGN (appliqué uniquement sur le train set)
- Analyse des résidus par plage de RUL
- Sauvegarde du modèle final (joblib)

---

## 2. API REST — FastAPI v2.0

```bash
uvicorn api:app --reload --port 8000
```

**Documentation Swagger interactive :** http://localhost:8000/docs

**Endpoints :**

| Méthode | Endpoint         | Description                                  |
|---------|------------------|----------------------------------------------|
| GET     | `/`              | Infos générales + liste des endpoints        |
| GET     | `/health`        | Statut du service, modèle chargé, métriques  |
| GET     | `/model-info`    | Détails modèle, features, performance, FE    |
| POST    | `/predict`       | Prédiction RUL + alerte + features calculées |
| POST    | `/predict/batch` | Prédiction batch jusqu'à 100 machines        |

> **Note :** Le champ `cumcount` est optionnel (défaut = 500).
> En production, fournir le vrai nombre de mesures depuis le début du cycle machine.

**Exemple de requête :**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "machine_type": "CNC",
    "vibration_rms": 4.2,
    "temperature_motor": 85.0,
    "current_phase_avg": 8.5,
    "pressure_level": 35.0,
    "rpm": 1200.0,
    "operating_mode": "peak",
    "hours_since_maintenance": 850.0,
    "ambient_temp": 28.0,
    "cumcount": 900
  }'
```

**Exemple de réponse :**

```json
{
  "rul_hours": 5.2,
  "alert_level": "critical",
  "alert_message": "ALERTE CRITIQUE — 5.2h restantes. Intervention immédiate.",
  "model_name": "RandomForestRegressor",
  "inference_time_ms": 11.4,
  "features_used": {
    "vibration_rms": 4.2,
    "temperature_motor": 85.0,
    "cumcount": 900,
    "hours_sq": 722500.0,
    "vib_x_temp": 357.0,
    "vib_x_hours": 3570.0,
    "temp_x_hours": 72250.0
  }
}
```

---

## 3. Dashboards Streamlit

### Dashboard Technique — EDA & Preprocessing

Destiné à l'équipe Data Science.

```bash
streamlit run dashboard_technique.py --server.port 8501
```

| Section | Contenu |
|---------|---------|
| Analyse du Dataset | Types de variables, statistiques, rôle de chaque colonne |
| Valeurs Manquantes | Visualisation % NaN, justification imputation médiane |
| Distributions | Histogrammes, boxplots par machine et mode opératoire |
| Corrélations | Heatmap interactive, corrélation individuelle avec rul_hours |
| Feature Engineering | Explication des 5 nouvelles features, visualisation cumcount vs RUL |
| Pipeline Preprocessing | Code annoté anti-leakage, résultats CV 5-fold |

### Dashboard Client — Interface Décisionnelle

Destiné aux responsables maintenance.

```bash
streamlit run dashboard_client.py --server.port 8502
```

| Module | Contenu |
|--------|---------|
| Prédiction RUL | Sliders capteurs + cumcount, jauge RUL colorée, alerte 3 niveaux, radar, 4 scénarios |
| Performances Modèles | KPIs, comparaison R²/MAE des 4 modèles, feature importance, CV stabilité |
| Données Capteurs | Évolution temporelle par machine, zones d'alerte, comparaison multi-machines |

---

## Résultats

| Modèle | MAE (h) | RMSE (h) | R² | CV MAE |
|--------|---------|----------|----|--------|
| Ridge Regression | 16.288 | 20.468 | 0.394 | 16.603 ± 0.114 |
| MLP (Deep Learning) | 4.911 | 7.604 | 0.916 | 4.943 ± 0.200 |
| XGBoost | 4.402 | 6.542 | 0.938 | 4.430 ± 0.114 |
| **Random Forest** | **3.077** | **5.575** | **0.955** | **3.097 ± 0.073** |

> Ces résultats incluent le Feature Engineering temporel.
> Baseline sans FE : R² = 0.671 — **gain de +42%** grâce au feature engineering.

**Modèle final : Random Forest** — meilleur compromis performance / interprétabilité / stabilité CV.

---

## Feature Engineering — Clé de la Performance

| Feature | Formule | Importance RF | Description |
|---------|---------|--------------|-------------|
| `cumcount` | `groupby('machine_id').cumcount()` | 40.0% | Nb mesures depuis début du cycle |
| `hours_sq` | `hours_since_maintenance²` | 23.5% | Relation non-linéaire avec le temps |
| `vib_x_temp` | `vibration_rms × temperature_motor` | 1.3% | Stress thermomécanique combiné |
| `vib_x_hours` | `vibration_rms × hours_since_maintenance` | 1.1% | Fatigue vibratoire cumulée |
| `temp_x_hours` | `temperature_motor × hours_since_maintenance` | 0.9% | Fatigue thermique cumulée |

---

## Système d'alerte

| RUL estimé | Niveau | Action recommandée |
|------------|--------|--------------------|
| < 10 h | CRITIQUE | Arrêt et intervention immédiate |
| 10–30 h | MODERE | Planifier intervention sous 48h |
| > 30 h | NORMAL | Surveillance standard |

---

## Garanties Anti Data Leakage

- Split train/test réalisé **avant** tout preprocessing
- `SimpleImputer` et `StandardScaler` ajustés **uniquement sur X_train**
- Variables exclues pour fuite causale : `failure_within_24h`, `failure_type`, `estimated_repair_cost`
- SMOGN appliqué **uniquement sur le train set** après le split
- Validation croisée : preprocessor refit à chaque fold indépendamment

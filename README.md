# Student Dropout Prediction AI

An early-warning machine learning system that identifies students at dropout risk and presents predictions through an interactive Streamlit dashboard.

---

## Why This Project Stands Out

This project is designed for early intervention, not just score prediction.

- It predicts risk before final outcomes by intentionally excluding late-stage signals.
- It compares multiple model families instead of relying on a single baseline.
- It provides both single-student and bulk CSV workflows for real-world use.
- It converts raw probabilities into actionable risk levels for faster decision-making.

---

## At a Glance

| Category | Details |
|---|---|
| Goal | Predict student dropout risk for proactive support |
| Dataset | UCI Student Performance (Mathematics) |
| App Mode | Single prediction + batch CSV prediction |
| Core Model for App | Logistic Regression |
| Alternative Models | Random Forest, Gradient Boosting |
| Output | Risk score, binary prediction, risk band |

---

## Risk Logic

### Target Label Rule

- dropout = 1 if G3 < 10
- dropout = 0 if G3 >= 10

### Risk Bands

- High Risk: score >= 0.70
- Medium Risk: 0.40 <= score < 0.70
- Low Risk: score < 0.40

---

## Project Layout

```text
.
|-- streamlit_app.py
|-- train_model.py
|-- requirements.txt
|-- data/
|   `-- student-mat.csv
|-- Models/
|-- Notebooks/
|   |-- dropout_risk_classification.py
|   `-- train_model.py
`-- demo-video/
```

---

## Quick Start

### 1) Install Dependencies

```bash
pip install -r requirements.txt
```

### 2) Train Models and Save Artifacts

```bash
python train_model.py
```

Training generates these files inside Models:

- logistic_regression_model.pkl
- scaler.pkl
- feature_columns.pkl
- random_forest_model.pkl
- gradient_boosting_model.pkl
- performance_metrics.pkl
- model_comparison.png

### 3) Launch the Web App

```bash
streamlit run streamlit_app.py
```

---

## App Experience

### Single Student Mode

- Minimal form with 5 high-impact fields.
- Remaining features are auto-filled using baseline defaults.
- Instant risk score with interpreted risk band.

### Batch Prediction Mode

- Upload a CSV with raw student features.
- App validates required columns.
- Download predictions as a ready-to-use CSV.

---

## Batch Input Contract

Uploaded CSV should include training-time raw features (except target-only or excluded columns).

- If present, dropout, G2, and G3 are ignored automatically.
- Include fields covering:
  - student profile and demographics
  - family background and social context
  - study behavior and academic indicators (including G1)

---

## Modeling Pipeline

1. Load data from data/student-mat.csv.
2. Create binary target from G3.
3. Drop G3 to prevent leakage.
4. One-hot encode categorical variables.
5. Drop G2 to preserve early intervention objective.
6. Split train/test with stratification.
7. Scale numeric features for Logistic Regression.
8. Train and evaluate Logistic Regression, Random Forest, and Gradient Boosting.
9. Save model artifacts for deployment.

---

## Technology Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- streamlit
- joblib

---

## Recommended Add-ons (Optional)

To make this repository even more presentable for judges, recruiters, or GitHub visitors, add:

1. Screenshot of the Streamlit interface.
2. Sample input CSV and sample output predictions CSV.
3. Latest metrics snapshot from your final training run.
4. Demo video link from demo-video.
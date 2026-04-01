"""
Dropout Risk Classification - Local Training Script
This script trains the dropout prediction model locally without Colab dependencies
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, recall_score, 
                             classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set style for plots
sns.set_style("whitegrid")

print("=" * 60)
print("DROPOUT RISK CLASSIFICATION - MODEL TRAINING")
print("=" * 60)

# ==================== LOAD AND EXPLORE DATA ====================
print("\n[1/8] Loading dataset...")
df = pd.read_csv("../data/student-mat.csv", sep=";")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nDataset info:\n{df.info()}")
print(f"\nDataset statistics:\n{df.describe()}")

# ==================== CHECK FOR NULL VALUES ====================
print("\n[2/8] Checking for null values...")
null_values = df.isnull().sum()
if null_values.sum() > 0:
    print(f"Null values found:\n{null_values[null_values > 0]}")
else:
    print("No null values found!")

# ==================== FEATURE ENGINEERING ====================
print("\n[3/8] Feature engineering - Converting G3 to dropout labels...")
print("Rule: G3 < 10 → At Risk (1), else Not At Risk (0)")
df['dropout'] = df['G3'].apply(lambda x: 1 if x < 10 else 0)

print(f"\nDropout value counts:\n{df['dropout'].value_counts()}")
print(f"\nDropout distribution:\n{df['dropout'].value_counts(normalize=True)}")

# Drop G3 to prevent data leakage
df = df.drop('G3', axis=1)
print(f"\nColumns after dropping G3: {list(df.columns)}")

# ==================== PREPARE FEATURES AND TARGET ====================
print("\n[4/8] Preparing features and target...")
X = df.drop('dropout', axis=1)
y = df['dropout']
print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)
print(f"Features shape after one-hot encoding: {X.shape}")

# Drop G2 for early intervention
X_no_g2 = X.drop('G2', axis=1)
print(f"Features shape after dropping G2: {X_no_g2.shape}")

# ==================== TRAIN-TEST SPLIT ====================
print("\n[5/8] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_no_g2, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# ==================== NORMALIZATION ====================
print("\n[6/8] Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features normalized successfully!")

# ==================== TRAIN MODELS ====================
print("\n[7/8] Training models...")

# --- LOGISTIC REGRESSION ---
print("\n  → Training Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_prob_lr = model.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = model.predict(X_test_scaled)

print(f"    Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"    ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")
print(f"    Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"\n    Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")
print(f"\n    Classification Report:\n{classification_report(y_test, y_pred_lr)}")

# --- RANDOM FOREST ---
print("\n  → Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = rf.predict(X_test)

print(f"    Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"    ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"    Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"\n    Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")
print(f"\n    Classification Report:\n{classification_report(y_test, y_pred_rf)}")

# --- GRADIENT BOOSTING ---
print("\n  → Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)

y_prob_gb = gb.predict_proba(X_test)[:, 1]
y_pred_gb = gb.predict(X_test)

print(f"    Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"    ROC-AUC: {roc_auc_score(y_test, y_prob_gb):.4f}")
print(f"    Recall: {recall_score(y_test, y_pred_gb):.4f}")
print(f"\n    Confusion Matrix:\n{confusion_matrix(y_test, y_pred_gb)}")
print(f"\n    Classification Report:\n{classification_report(y_test, y_pred_gb)}")

# ==================== FEATURE IMPORTANCE ====================
print("\n[8/8] Analyzing feature importance...")

feature_importance = pd.DataFrame({
    'Feature': X_no_g2.columns,
    'Coefficient': model.coef_[0]
})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

print("\nTop 5 features for AT RISK prediction:")
print(feature_importance.head(5))

print("\nTop 5 features for NOT AT RISK prediction:")
print(feature_importance.tail(5))

# ==================== MODEL COMPARISON & VISUALIZATION ====================
print("\nGenerating model comparison visualizations...")

performance_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_gb)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_prob_lr),
        roc_auc_score(y_test, y_prob_rf),
        roc_auc_score(y_test, y_prob_gb)
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_gb)
    ]
})

print("\nModel Performance Summary:")
print(performance_df)

# Create comparison plots
fig = plt.figure(figsize=(18, 6))

# Accuracy
plt.subplot(1, 3, 1)
sns.barplot(x='Model', y='Accuracy', data=performance_df, palette='viridis', hue='Model', legend=False)
plt.title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy Score')
plt.ylim(0.7, 1.0)
plt.xticks(rotation=45, ha='right')

# ROC-AUC
plt.subplot(1, 3, 2)
sns.barplot(x='Model', y='ROC-AUC', data=performance_df, palette='magma', hue='Model', legend=False)
plt.title('Model ROC-AUC Comparison', fontsize=12, fontweight='bold')
plt.ylabel('ROC-AUC Score')
plt.ylim(0.7, 1.0)
plt.xticks(rotation=45, ha='right')

# Recall
plt.subplot(1, 3, 3)
sns.barplot(x='Model', y='Recall', data=performance_df, palette='plasma', hue='Model', legend=False)
plt.title('Model Recall Comparison', fontsize=12, fontweight='bold')
plt.ylabel('Recall Score')
plt.ylim(0.0, 1.0)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../Models/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Model comparison plot saved to Models/model_comparison.png")
plt.show()

# ==================== SAVE MODELS ====================
print("\nSaving models to disk...")

# Create Models directory if it doesn't exist
os.makedirs('../Models', exist_ok=True)

joblib.dump(model, '../Models/logistic_regression_model.pkl')
print("✓ Logistic Regression model saved")

joblib.dump(scaler, '../Models/scaler.pkl')
print("✓ Scaler saved")

joblib.dump(X_no_g2.columns.tolist(), '../Models/feature_columns.pkl')
print("✓ Feature columns saved")

joblib.dump(rf, '../Models/random_forest_model.pkl')
print("✓ Random Forest model saved")

joblib.dump(gb, '../Models/gradient_boosting_model.pkl')
print("✓ Gradient Boosting model saved")

joblib.dump(performance_df, '../Models/performance_metrics.pkl')
print("✓ Performance metrics saved")

# ==================== VERIFY MODELS ====================
print("\nVerifying saved models...")
loaded_model = joblib.load('../Models/logistic_regression_model.pkl')
loaded_scaler = joblib.load('../Models/scaler.pkl')
loaded_columns = joblib.load('../Models/feature_columns.pkl')

print("✓ All models loaded successfully!")
print(f"✓ Feature count: {len(loaded_columns)}")

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nSaved artifacts:")
print("  • logistic_regression_model.pkl")
print("  • random_forest_model.pkl")
print("  • gradient_boosting_model.pkl")
print("  • scaler.pkl")
print("  • feature_columns.pkl")
print("  • performance_metrics.pkl")
print("  • model_comparison.png")

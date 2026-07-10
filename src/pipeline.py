"""
Micro-loan Payback Prediction — End-to-end pipeline
Rebuilds the original 2022 project with proper handling of class imbalance,
full-data training, and metrics appropriate for imbalanced classification.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                              classification_report, confusion_matrix, f1_score)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 110
sns.set_style("whitegrid")
PALETTE = {"pay": "#2E86AB", "default": "#E63946"}

IMG_DIR = "/home/claude/loan-payback-v2/images"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
import zipfile, os
DATA_DIR = "/home/claude/loan-payback-v2/data"
csv_path = os.path.join(DATA_DIR, "sample_data_intw.csv")
if not os.path.exists(csv_path):
    with zipfile.ZipFile(os.path.join(DATA_DIR, "sample_data_intw.zip")) as z:
        z.extractall(DATA_DIR)
df = pd.read_csv(csv_path, index_col=0)
print("Shape:", df.shape)

# ---------------------------------------------------------------
# 2. EDA — class balance
# ---------------------------------------------------------------
class_counts = df['label'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(['Defaulted (0)', 'Paid back (1)'], class_counts.values,
              color=[PALETTE['default'], PALETTE['pay']])
for b, v in zip(bars, class_counts.values):
    ax.text(b.get_x() + b.get_width()/2, v + 2000, f"{v:,}\n({v/len(df)*100:.1f}%)",
            ha='center', fontsize=10)
ax.set_title("Class Balance: Loan Repayment Outcome", fontsize=13, fontweight='bold')
ax.set_ylabel("Number of loans")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/01_class_balance.png")
plt.close()

# ---------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------
# last_rech_date_da, cnt_da_rech30/90, fr_da_rech30/90 are almost entirely
# missing/zero in the original notebook's own exploration — drop non-informative /
# identifier columns.
drop_cols = ['msisdn', 'pcircle', 'pdate', 'last_rech_date_da',
             'cnt_da_rech30', 'fr_da_rech30', 'cnt_da_rech90', 'fr_da_rech90']
df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
df_clean = df_clean.fillna(0)

y = df_clean['label']
X = df_clean.drop(columns=['label'])

# ---------------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 11))
corr = df_clean.corr()
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/02_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------------
# 5. Train/test split — stratified (full data, not truncated to 1000 rows)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# ---------------------------------------------------------------
# 6. Models — all trained on FULL training data, class_weight balanced
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                             max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        eval_metric='logloss', random_state=42, n_jobs=-1
    ),
}

results = []
roc_data = {}
fitted_models = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_std, y_train)
        y_proba = model.predict_proba(X_test_std)[:, 1]
        y_pred = model.predict(X_test_std)
    else:
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)
    fitted_models[name] = model

    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        "Model": name,
        "ROC-AUC": round(auc, 4),
        "F1 (default class)": round(f1_score(y_test, y_pred, pos_label=0), 4),
        "F1 (paid class)": round(f1, 4),
        "Recall (default class)": round(report['0']['recall'], 4),
        "Precision (default class)": round(report['0']['precision'], 4),
    })
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=['Defaulted', 'Paid back']))

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\n", results_df.to_string(index=False))
results_df.to_csv("/home/claude/loan-payback-v2/images/model_comparison.csv", index=False)

# ---------------------------------------------------------------
# 7. ROC curves — all models
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
colors = ['#E63946', '#2E86AB', '#F4A261']
for (name, (fpr, tpr, auc)), c in zip(roc_data.items(), colors):
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=c, linewidth=2)
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/03_roc_curves.png")
plt.close()

# ---------------------------------------------------------------
# 8. Confusion matrix — best model (by ROC-AUC)
# ---------------------------------------------------------------
best_name = results_df.iloc[0]['Model']
best_model = fitted_models[best_name]
X_test_used = X_test_std if best_name == "Logistic Regression" else X_test
y_pred_best = best_model.predict(X_test_used)
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(5.5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Defaulted', 'Paid back'],
            yticklabels=['Defaulted', 'Paid back'], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_name} (Best Model)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/04_confusion_matrix_best.png")
plt.close()

# ---------------------------------------------------------------
# 9. Feature importance (Random Forest / XGBoost)
# ---------------------------------------------------------------
importance_model = fitted_models["XGBoost"]
importances = pd.Series(importance_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(8, 7))
importances.plot(kind='barh', ax=ax, color='#2E86AB')
ax.set_title("Top 15 Feature Importances (XGBoost)", fontsize=13, fontweight='bold')
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/05_feature_importance.png")
plt.close()

print("\nBest model:", best_name)
print("\nAll images saved to", IMG_DIR)

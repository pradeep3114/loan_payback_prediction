# 📱 Micro-Loan Payback Prediction

Predicting whether a telecom micro-loan customer will repay within 5 days, using 200K+ customer behavior records.

**Originally built:** 2022 · **Rebuilt & improved:** 2026

---

## 🎯 Business Problem

A telecom operator issues small, short-term "micro-loans" (recharge credit) to subscribers, expected to be repaid within 5 days. The goal: predict **before issuing a loan** whether a customer is likely to repay, so the company can reduce default losses, speed up approvals for low-risk customers, and set risk-based loan limits.

## 🔁 Why This Was Rebuilt

The original 2022 version had three issues that made its results unreliable:

| Issue | Original (2022) | Fixed (2026) |
|---|---|---|
| Training data size | Models trained on only 1,000 rows out of ~168K available | Trained on the **full training set** |
| Class imbalance | Not handled — dataset is ~87.5% repaid / 12.5% defaulted | **Class-weighted models** (`class_weight='balanced'`, `scale_pos_weight`) |
| Evaluation metric | Accuracy only (misleading on imbalanced data) | **ROC-AUC, precision/recall per class, F1** |

## 📊 Results

| Model | ROC-AUC | Recall (defaulters) | Precision (defaulters) |
|---|---|---|---|
| **XGBoost** | **0.908** | **0.80** | 0.42 |
| Random Forest | 0.892 | 0.74 | 0.43 |
| Logistic Regression | 0.831 | 0.81 | 0.28 |

XGBoost gives the best overall trade-off — catching 80% of actual defaulters while keeping strong performance on customers who repay on time.

## 🖼️ Key Visuals

- Class balance chart
- Feature correlation heatmap
- ROC curves comparing all 3 models
- Confusion matrix for the best model
- Top-15 feature importance (XGBoost)

All available in [`images/`](./images) and embedded directly in the notebook.

## 🔑 Key Insight

Recent recharge behavior (amount, frequency, recency) and account tenure (`aon`) are the strongest predictors of repayment — customers with frequent, recent top-ups and longer network history are more likely to repay on time. Past loan repayment history is also highly predictive, confirming that repayment behavior tends to repeat.

## 💡 Business Recommendations

1. Deploy the XGBoost model as the credit-scoring gate for new loan requests
2. Use recharge recency/frequency as a real-time risk signal / pre-filter
3. Weight past repayment history heavily in risk tiers for returning customers
4. Tune the decision threshold based on the actual business cost of a missed default vs. a declined good customer — not a default 0.5 cutoff

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Modeling:** scikit-learn (Logistic Regression, Random Forest), XGBoost
- **Visualization:** Matplotlib, Seaborn

## 📁 Project Structure

```
loan-payback-v2/
├── data/
│   ├── sample_data_intw.zip       # 209,593 loan records, 36 features (zipped)
│   └── Data_Description.xlsx      # Column definitions
├── notebooks/
│   └── loan_payback_prediction.ipynb   # Full analysis, executed with outputs
├── images/                        # Generated charts
├── src/
│   └── pipeline.py                # Standalone script version of the analysis
├── requirements.txt
└── README.md
```

## 🚀 How to Run Locally

```bash
git clone https://github.com/pradeep3114/loan-payback-prediction.git
cd loan-payback-prediction
unzip data/sample_data_intw.zip -d data/
pip install -r requirements.txt
jupyter notebook notebooks/loan_payback_prediction.ipynb
```

Or run the standalone script:
```bash
python src/pipeline.py
```

---
**Author:** Pradeep Sehrawat

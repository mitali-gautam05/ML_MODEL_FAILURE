# When ML Fails 🔍
### A Controlled Study of ML Failure Modes on Imbalanced Data

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dashboard-yellow)](https://huggingface.co/spaces/Mitali1234/when-ml-fails-app)
[![FastAPI](https://img.shields.io/badge/FastAPI-API%20Docs-009688)](https://mitali1234-when-ml-fails-api.hf.space/docs)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)

---

## What is this project?

Most ML tutorials focus on improving accuracy. This project does the opposite —  
it systematically **exposes how and why standard ML practices fail** on real-world imbalanced data.

Using the Credit Card Fraud dataset (284,807 transactions, 0.17% fraud), I ran 4 controlled  
experiments to demonstrate failure modes that cause production ML systems to silently break.

---

## Live Demo

| | Link |
|---|---|
| 📊 Interactive Dashboard | [when-ml-fails-app](https://huggingface.co/spaces/Mitali1234/when-ml-fails-app) |
| ⚡ FastAPI Backend | [when-ml-fails-api/docs](https://mitali1234-when-ml-fails-api.hf.space/docs) |

---

## The 4 Failure Modes

### Experiment 1 — The Accuracy Lie
A dummy classifier that **never detects fraud** scores **99.83% accuracy**.  
Accuracy is meaningless for imbalanced data. Recall = 0%.

### Experiment 2 — Validation Strategy Matters
Random KFold can produce folds with near-zero fraud cases.  
Stratified KFold is mandatory — even a small difference compounds in production pipelines.

### Experiment 3 — Complexity vs Imbalance
XGBoost without imbalance handling overfits the minority class.  
Logistic Regression sometimes outperforms complex models on Recall.

### Experiment 4 — Fixing It Right
Three strategies compared — `class_weight`, SMOTE, threshold tuning.  
No single fix wins universally. Business context decides:  
- Bank → maximize Recall (catch all fraud)  
- Insurance → maximize Precision (avoid false alarms)

---

## 7 Models Compared

| Model | F1 (No Fix) | F1 (SMOTE) |
|---|---|---|
| Logistic Regression | 0.7241 | 0.1089 |
| Decision Tree | 0.7487 | 0.5256 |
| Naive Bayes | 0.1099 | 0.1055 |
| KNN | 0.8587 | 0.5831 |
| SVM | 0.3077 | 0.1334 |
| Random Forest | 0.8743 | 0.5714 |
| XGBoost | 0.8495 | 0.3508 |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Research | Python, Scikit-learn, XGBoost, Pandas, Matplotlib |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit, Plotly |
| Deployment | HuggingFace Spaces (Docker + Streamlit SDK) |

---

## Project Structure
when-ml-fails/

├── backend/

│   ├── main.py        ← FastAPI endpoints

│   ├── models.py      ← Training + evaluation logic

│   └── schemas.py     ← Request/response models

├── frontend/

│   └── app.py         ← Streamlit dashboard

├── train.py           ← One-time model training script

└── requirements.txt


## Key Takeaway

> ML success depends more on **how we evaluate and reason about models**  
> than on which algorithm we choose.
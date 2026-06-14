import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE

os.makedirs("saved_models", exist_ok=True)

def get_all_models():
    return {
        "Logistic Regression": LogisticRegression(
            random_state=42, max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf", probability=True,
            class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=10,
            n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=50, max_depth=4,
            n_jobs=-1, random_state=42,
            eval_metric="logloss", verbosity=0
        ),
    }

def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba") else None
    )
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4)
                     if y_proba is not None else None,
    }

def train_and_save_all(data_path="data/creditcard.csv"):
    print("Loading data...")
    df = pd.read_csv(data_path)
    X  = df.drop("Class", axis=1)
    y  = df["Class"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "saved_models/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2,
        stratify=y, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    models  = get_all_models()
    results = {}
    SVM_SAMPLE = 10000

    for name, model in models.items():
        print(f"Training {name}...")
        try:
            if name == "SVM":
                idx = np.random.choice(len(X_train), SVM_SAMPLE, replace=False)
                model.fit(X_train[idx], y_train.iloc[idx])
            else:
                model.fit(X_train, y_train)

            metrics_base = evaluate(model, X_test, y_test)

            model_sm = type(model)(**model.get_params())
            if name == "SVM":
                sm_idx = np.random.choice(len(X_train_sm), SVM_SAMPLE, replace=False)
                model_sm.fit(X_train_sm[sm_idx], y_train_sm.iloc[sm_idx])
            else:
                model_sm.fit(X_train_sm, y_train_sm)

            metrics_smote = evaluate(model_sm, X_test, y_test)

            results[name] = {
                "without_fix": metrics_base,
                "with_smote":  metrics_smote,
            }

            joblib.dump(model, f"saved_models/{name.replace(' ', '_')}.pkl")
            print(f"  done — F1 (base): {metrics_base['f1']}  |  F1 (smote): {metrics_smote['f1']}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"error": str(e)}

    joblib.dump(results, "saved_models/all_results.pkl")
    print("\nAll models saved.")
    return results
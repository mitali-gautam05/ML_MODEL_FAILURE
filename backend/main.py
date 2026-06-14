import json
import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="When ML Fails — API",
    description="Exposes ML failure mode experiments and model comparisons",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

all_results  = {}
colab_data   = {}
scaler       = None

@app.on_event("startup")
def load_artifacts():
    global all_results, colab_data, scaler

    if os.path.exists("saved_models/all_results.pkl"):
        all_results = joblib.load("saved_models/all_results.pkl")
        print(f"Loaded {len(all_results)} model results")
    else:
        print("No trained models found. Run: python train.py")

    if os.path.exists("saved_models/scaler.pkl"):
        scaler = joblib.load("saved_models/scaler.pkl")

    if os.path.exists("data/colab_results.json"):
        with open("data/colab_results.json") as f:
            colab_data = json.load(f)
        print("Loaded Colab experiment results")
    else:
        print("No colab_results.json found in data/")


# ── Health ───────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "models_loaded": len(all_results),
        "colab_data_loaded": bool(colab_data),
    }


# ── Models ───────────────────────────────────────────
@app.get("/models")
def list_models():
    """All available model names"""
    return {"models": list(all_results.keys())}


# ── Comparison ───────────────────────────────────────
@app.get("/compare")
def compare_all():
    """Full comparison: all models, with and without SMOTE"""
    if not all_results:
        raise HTTPException(503, "Models not trained yet. Run: python train.py")
    return all_results


@app.get("/compare/{model_name}")
def compare_single(model_name: str):
    """Metrics for one model"""
    key = model_name.replace("-", " ").replace("_", " ")
    if key not in all_results:
        raise HTTPException(
            404,
            f"'{model_name}' not found. Available: {list(all_results.keys())}"
        )
    return {key: all_results[key]}


# ── Experiments ──────────────────────────────────────
@app.get("/experiments")
def get_experiment_descriptions():
    """4 failure mode experiment descriptions"""
    return {
        "experiments": [
            {
                "id": 1,
                "title": "Wrong metric trap",
                "finding": "High accuracy hides zero fraud recall. Accuracy is meaningless for imbalanced data.",
                "key_metric": "recall"
            },
            {
                "id": 2,
                "title": "Validation strategy matters",
                "finding": "Random KFold can produce folds with near-zero fraud cases. Stratified KFold is mandatory.",
                "key_metric": "f1"
            },
            {
                "id": 3,
                "title": "Complexity vs imbalance",
                "finding": "Complex models overfit without imbalance handling. Simpler models can outperform on recall.",
                "key_metric": "f1"
            },
            {
                "id": 4,
                "title": "Fixing it right",
                "finding": "SMOTE, class_weight, and threshold tuning each have tradeoffs. Context decides which fix is best.",
                "key_metric": "precision"
            }
        ]
    }


@app.get("/experiments/results")
def get_colab_results():
    """Actual numbers from Phase 1+2 Colab experiments"""
    if not colab_data:
        raise HTTPException(
            404,
            "colab_results.json not found. Export from Colab and place in data/"
        )
    return colab_data


# ── Predict ──────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Single transaction prediction using a trained model"""
    if scaler is None:
        raise HTTPException(503, "Scaler not loaded. Run: python train.py")

    model_file = f"saved_models/{req.model_name.replace(' ', '_')}.pkl"
    if not os.path.exists(model_file):
        raise HTTPException(
            404,
            f"Model file not found: {model_file}. Run: python train.py"
        )

    if len(req.features) != 30:
        raise HTTPException(
            400,
            f"Expected 30 features, got {len(req.features)}"
        )

    model          = joblib.load(model_file)
    features       = np.array(req.features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction     = int(model.predict(features_scaled)[0])
    confidence     = None

    if hasattr(model, "predict_proba"):
        confidence = round(
            float(model.predict_proba(features_scaled)[0][1]), 4
        )

    return PredictResponse(
        model      = req.model_name,
        prediction = prediction,
        label      = "FRAUD" if prediction == 1 else "Not Fraud",
        confidence = confidence,
    )